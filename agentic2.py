import os
import subprocess
import sys
import requests
import operator
import re
import json
import uuid
import yaml
from typing import Annotated, Sequence, TypedDict, Literal
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import litellm
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)

TOTAL_SESSION_COST = 0.0

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file '{path}' not found. "
            "Please create it based on the config.yaml template."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)

cfg     = load_config()
prompts = load_config("prompts.yaml")

# System-level constants
DEBUG_MODE = cfg["system"]["debug_mode"]

# LiteLLM connection
LITELLM_URL      = cfg["litellm"]["base_url"]
LITELLM_AUTH_KEY = cfg["litellm"]["api_key"]

# ---------------------------------------------------------------------------
# LLM factory + per-agent profile helpers
# ---------------------------------------------------------------------------

def get_profile(profile_name: str) -> dict:
    """Return the raw profile dict for a named LLM profile."""
    profiles = cfg.get("llm_profiles", {})
    if profile_name not in profiles:
        raise ValueError(
            f"LLM profile '{profile_name}' not found in config.yaml. "
            f"Available profiles: {list(profiles.keys())}"
        )
    return profiles[profile_name]

# Resolved agent → profile mapping (for buffer lookups at runtime)
_agent_profiles = {
    agent: get_profile(agent_cfg["llm"])
    for agent, agent_cfg in cfg.get("agents", {}).items()
}

def agent_limit(agent_name: str, key: str, default):
    """Look up a per-profile limit for a given agent, falling back to default."""
    return _agent_profiles.get(agent_name, {}).get(key, default)

def make_llm(profile_name: str) -> ChatOpenAI:
    """Instantiate a ChatOpenAI client from a named profile in config.yaml."""
    profile = get_profile(profile_name)
    return ChatOpenAI(
        model       = profile["model"],
        temperature = profile.get("temperature", 0),
        max_tokens  = profile.get("max_tokens", None),
        base_url    = LITELLM_URL,
        api_key     = LITELLM_AUTH_KEY,
    )

# Per-agent LLM instances
manager_llm    = make_llm(cfg["agents"]["manager"]["llm"])
researcher_llm = make_llm(cfg["agents"]["researcher"]["llm"])
writer_llm     = make_llm(cfg["agents"]["writer"]["llm"])
# Ensure you add 'summariser' to your config.yaml agents list!
summariser_llm = make_llm(cfg["agents"]["summariser"]["llm"])

# ---------------------------------------------------------------------------
# Worker Tools (Vault Actions)
# ---------------------------------------------------------------------------

@tool
def create_note(title: str, content: str) -> str:
    """Create a new note in the Obsidian vault."""
    safe_title = re.sub(r'[\\/*?:"<>|]', '-', title)
    print(f"\n[Writer Tool] ⚡ Executing CLI to create: '{safe_title}'...")
    try:
        subprocess.run(["obsidian", "create", f"name={safe_title}", f"content={content}"], capture_output=True, text=True, check=True)
        return f"Success: Note saved exactly as '{safe_title}'."
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def append_note(filename: str, content: str) -> str:
    """Append text content to an existing note."""
    safe_filename = re.sub(r'[\\/*?:"<>|]', '-', filename)
    try:
        subprocess.run(["obsidian", "append", f"file={safe_filename}", f"content={content}"], capture_output=True, text=True, check=True)
        return f"Success: Appended content to '{safe_filename}'."
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def read_note(filename: str, search_keyword: str = None) -> str:
    """Read a specific note. Optional: provide a 'search_keyword' to extract only relevant snippets from very long notes."""
    print(f"\n[Summariser Tool] 🔍 Reading note: '{filename}'" + (f" (Keyword: '{search_keyword}')" if search_keyword else "") + "...")
    try:
        result  = subprocess.run(["obsidian", "read", f"file={filename}"], capture_output=True, text=True, check=True)
        content = result.stdout.strip()

        # Limits now pull from the summariser profile
        max_len = agent_limit("summariser", "max_tool_response_length", 20000)
        buf     = agent_limit("summariser", "read_note_buffer_size", 5000)

        if len(content) <= max_len:
            return content

        if search_keyword:
            matches = [m.start() for m in re.finditer(re.escape(search_keyword), content, re.IGNORECASE)]

            if not matches:
                return (f"SYSTEM WARNING: The file '{filename}' is {len(content)} characters long, "
                        f"but the keyword '{search_keyword}' was NOT found. "
                        f"Here is the beginning of the file:\n\n{content[:3000]}...")

            snippets = []
            for match_idx in matches[:5]:
                start = max(0, match_idx - buf)
                end   = min(len(content), match_idx + len(search_keyword) + buf)
                snippets.append(f"...{content[start:end]}...")

            extracted_text  = f"--- EXTRACTED SNIPPETS FOR '{search_keyword}' IN '{filename}' (File too large to load entirely) ---\n\n"
            extracted_text += "\n\n[... SNIPPET BREAK ...]\n\n".join(snippets)
            return extracted_text

        else:
            return (content[:max_len] +
                    "\n\n[... TEXT TRUNCATED BY SYSTEM ...]\n" +
                    f"ACTION REQUIRED: This note is {len(content)} characters long and exceeds the {max_len} limit. " +
                    "To read deeper into this file, you MUST call `read_note` again and provide a specific `search_keyword`.")

    except Exception as e:
        return f"Error reading note: {str(e)}"

@tool
def search_vault(query: str) -> str:
    """Search the vault for filenames containing the query. Returns a list of matching file paths."""
    print(f"\n[Research Tool] 🔍 Searching vault for: '{query}'...")
    
    def _do_search(q, limit=10):
        try:
            result = subprocess.run(["obsidian", "search", f"query={q}", "format=text", f"limit={limit}"], capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            if "No matches found." in output:
                return ""
            return re.sub(r"2026-.*?https://obsidian\.md/download\n*", "", output, flags=re.DOTALL).strip()
        except Exception:
            return ""

    try:
        output = _do_search(query, 10)
        if output:
            return output
        
        # Fallback: split query if it has multiple words
        queries = [w.strip() for w in query.split() if len(w.strip()) > 2]
        if len(queries) <= 1:
            return "No results found. Try a different, maybe less specific search query."
            
        if DEBUG_MODE:
            print(f"No results for full query. Splitting into words: {queries[:5]}")
            
        results = []
        for q in queries[:5]:
            ans = _do_search(q, 5)
            if ans:
                results.append(f"--- Results for '{q}' ---\n{ans}")
                
        if results:
            return "\n\n".join(results)
        
        return "No results found. Try a different, maybe less specific search query."

    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------------------------------------------------------
# Communication & Planning Tools (State Routing)
# ---------------------------------------------------------------------------

@tool
def update_plan(plan_details: str) -> str:
    """Create or update your step-by-step plan to track progress."""
    return f"System: Plan successfully updated.\nCurrent Plan:\n{plan_details}"

@tool
def ask_clarifying_question(question: str) -> str:
    """Ask the user a clarifying question before proceeding with the plan."""
    return question

@tool
def delegate_to_researcher(task: str) -> str:
    """Assign a research task to the Researcher agent to look up information in the vault."""
    return f"System: Task delegated to Researcher -> {task}"

@tool
def delegate_to_summariser(file_path: str, research_question: str) -> str:
    """Assign a specific file to the Summariser to read and extract answers for your research question."""
    return f"System: Task delegated to Summariser -> Read '{file_path}' for question: '{research_question}'"

@tool
def delegate_to_writer(task: str, sources: list[str]) -> str:
    """Assign a saving/writing task to the Writer agent. Provide the task description and a list of sources that must be cited."""
    cleaned = [s.split('/')[-1][:-3] if s.lower().endswith(".md") else s.split('/')[-1] for s in sources]
    src_str = ", ".join(cleaned) if cleaned else "None provided"
    return f"System: Task delegated to Writer -> {task}\nSources to cite: {src_str}"

@tool
def respond_to_user(final_answer: str) -> str:
    """Respond directly to the user to conclude the conversation."""
    return final_answer

@tool
def submit_findings(summary: str, sources: list[str]) -> str:
    """Submit the final research findings back to the Manager. Must include a list of filenames used as sources."""
    cleaned = [s.split('/')[-1][:-3] if s.lower().endswith(".md") else s.split('/')[-1] for s in sources]
    src_str = ", ".join(cleaned) if cleaned else "None provided"
    return f"Researcher Findings: {summary}\nSources found: {src_str}"

@tool
def submit_summary(summary: str, source: str) -> str:
    """Submit the summary of the read note back to the Researcher."""
    return f"Summariser Findings for '{source}':\n{summary}"

@tool
def finish_writing(confirmation: str) -> str:
    """Confirm to the Manager that the writing task is complete."""
    return f"Writer Confirmation: {confirmation}"

manager_tools    = [delegate_to_researcher, delegate_to_writer, respond_to_user, ask_clarifying_question, update_plan]
researcher_tools = [search_vault, delegate_to_summariser, submit_findings]
summariser_tools = [read_note, submit_summary]
writer_tools     = [create_note, append_note, finish_writing]

# ---------------------------------------------------------------------------
# Orchestration Helpers
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def process_reasoning_output(response: AIMessage, name: str, tool_map: dict) -> AIMessage:
    raw_content = response.content or ""
    thoughts    = ""

    # Native extraction for proxy
    if "reasoning_content" in response.additional_kwargs:
        thoughts         = response.additional_kwargs["reasoning_content"]
        cleaned_content  = raw_content
    else:
        thought_match   = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL)
        if thought_match:
            thoughts    = thought_match.group(1).strip()
        cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

    response.content = cleaned_content

    if DEBUG_MODE and thoughts:
        print(f"\n🧠 [DEBUG - {name} Thoughts]:\n{thoughts}")

    # Cost tracking
    metadata          = response.response_metadata or {}
    actual_model      = metadata.get("model_name") or metadata.get("model", "unknown")
    clean_model       = actual_model.replace("openai/", "").replace("openrouter/", "")

    usage = getattr(response, "usage_metadata", None)
    if usage:
        in_tokens  = usage.get("input_tokens", 0)
        out_tokens = usage.get("output_tokens", 0)
        cost = 0.0
        try:
            cost = litellm.completion_cost(model=clean_model, prompt_tokens=in_tokens, completion_tokens=out_tokens)
        except Exception:
            pass

        global TOTAL_SESSION_COST
        TOTAL_SESSION_COST += cost
        
        print(f"💰 [Cost Tracker] {name} using {clean_model}: "
              f"{in_tokens} in, {out_tokens} out tokens. "
              f"Cost: ${cost:.6f} | Session Total: ${TOTAL_SESSION_COST:.6f}")

    # JSON Catcher - Upgraded to handle Qwen/OpenAI nested schemas
    if not response.tool_calls and (cleaned_content.startswith("{") or cleaned_content.startswith("[")):
        try:
            parsed = json.loads(cleaned_content)
            parsed = [parsed] if isinstance(parsed, dict) else parsed
            
            for item in parsed:
                # 1. Check for the nested OpenAI/Qwen function format
                if "function" in item and "name" in item["function"]:
                    tool_name = item["function"]["name"]
                    args_raw  = item["function"].get("arguments", {})
                    
                    # Sometimes the nested arguments are passed as a stringified JSON
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = args_raw
                        
                    if tool_name in tool_map:
                        print(f"[System] 🔧 Rescued tool call '{tool_name}' from nested OpenAI JSON format!")
                        response.tool_calls.append({"name": tool_name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"})
                        response.content = ""
                        continue

                # 2. Original fallback for flat dictionary formats
                item_keys = set(item.keys())
                for tool_name, tool_obj in tool_map.items():
                    try:
                        expected_keys = set(tool_obj.args_schema.model_json_schema()["properties"].keys()) if tool_obj.args_schema else set()
                    except AttributeError:
                        expected_keys = set(tool_obj.args_schema.schema()["properties"].keys()) if tool_obj.args_schema else set()
                    
                    if expected_keys and expected_keys.issubset(item_keys):
                        print(f"[System] 🔧 Rescued tool call '{tool_name}' from raw flat JSON!")
                        response.tool_calls.append({"name": tool_name, "args": item, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"})
                        response.content = ""
                        break
        except json.JSONDecodeError:
            pass

    if not response.tool_calls:
        if not response.content.strip():
            print(f"⚠️ [System] {name} provided empty output. Forcing a fallback tool call.")
            err_msg = "SYSTEM ERROR: You must output a valid tool call."
        else:
            err_msg = response.content

        print(f"⚠️ [System] {name} failed to output a tool call (or hit an API error). Wrapping plain text to prevent looping.")
        
        if name == "Manager":
            response.tool_calls.append({
                "name": "respond_to_user",
                "args": {"final_answer": err_msg},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call"
            })
        elif name == "Researcher":
            response.tool_calls.append({
                "name": "submit_findings",
                "args": {"summary": err_msg, "sources": []},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call"
            })
        elif name == "Summariser":
            response.tool_calls.append({
                "name": "submit_summary",
                "args": {"summary": err_msg, "source": "unknown"},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call"
            })
        elif name == "Writer":
            response.tool_calls.append({
                "name": "finish_writing",
                "args": {"confirmation": err_msg},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call"
            })
        
        response.content = ""

    return response

def safe_invoke(llm_bound, msgs, name, tool_map):
    """Wraps LLM invocation to catch empty-choice API crashes and handle them gracefully."""
    try:
        response = llm_bound.invoke(msgs)
        return process_reasoning_output(response, name, tool_map)
    except IndexError:
        print(f"\n🛡️ [System Shield] Caught API Empty Response Error (Safety Filter/Parse Failure). Forcing fallback.")
        err_msg = "SYSTEM ERROR: The underlying AI model refused to process the prompt (likely due to a safety filter or parsing bug)."
        return process_reasoning_output(AIMessage(content=err_msg), name, tool_map)
    except Exception as e:
        print(f"\n🛡️ [System Shield] Caught API Exception: {str(e)[:200]}... Forcing fallback.")
        err_msg = f"SYSTEM ERROR: The API call failed with error: {str(e)}"
        return process_reasoning_output(AIMessage(content=err_msg), name, tool_map)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def manager_node(state: AgentState):
    print("\n[Manager] Evaluating task...")
    msgs     = [SystemMessage(content=prompts["manager"])] + list(state["messages"])
    response = safe_invoke(manager_llm.bind_tools(manager_tools), msgs, "Manager", {t.name: t for t in manager_tools})
    return {"messages": [response]}

def researcher_node(state: AgentState):
    print("\n[Researcher] Investigating...")
    msgs     = [SystemMessage(content=prompts["researcher"])] + list(state["messages"])
    response = safe_invoke(researcher_llm.bind_tools(researcher_tools), msgs, "Researcher", {t.name: t for t in researcher_tools})
    return {"messages": [response]}

def summariser_node(state: AgentState):
    print("\n[Summariser] Reading and condensing source...")
    msgs     = [SystemMessage(content=prompts.get("summariser", "You are the Summariser. Read notes and extract information relevant to the research question."))] + list(state["messages"])
    response = safe_invoke(summariser_llm.bind_tools(summariser_tools), msgs, "Summariser", {t.name: t for t in summariser_tools})
    return {"messages": [response]}

def writer_node(state: AgentState):
    print("\n[Writer] Writing...")
    msgs     = [SystemMessage(content=prompts["writer"])] + list(state["messages"])
    response = safe_invoke(writer_llm.bind_tools(writer_tools), msgs, "Writer", {t.name: t for t in writer_tools})
    return {"messages": [response]}

# ---------------------------------------------------------------------------
# Shared Tool Executor
# ---------------------------------------------------------------------------

def tool_executor(state: AgentState):
    last_msg = state["messages"][-1]

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"messages": []}

    all_tools_map = {t.name: t for t in manager_tools + researcher_tools + summariser_tools + writer_tools}

    tool_to_agent = (
        {t.name: "manager"    for t in manager_tools} |
        {t.name: "researcher" for t in researcher_tools} |
        {t.name: "summariser" for t in summariser_tools} |
        {t.name: "writer"     for t in writer_tools}
    )

    outs = []

    ai_msgs     = [m for m in state["messages"] if isinstance(m, AIMessage)]
    prev_ai_msg = ai_msgs[-2] if len(ai_msgs) >= 2 else None

    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        # --- HARDENED ANTI-LOOP MECHANISM ---
        is_duplicate = False
        if prev_ai_msg and hasattr(prev_ai_msg, 'tool_calls') and prev_ai_msg.tool_calls:
            for prev_tc in prev_ai_msg.tool_calls:
                if prev_tc["name"] == tool_name and str(prev_tc["args"]).strip() == str(tool_args).strip():
                    is_duplicate = True
                    break

        if is_duplicate:
            print(f"\n🛑 [System Anti-Loop] Blocked repeated identical call: {tool_name}({tool_args})")
            error_msg = (
                f"SYSTEM ERROR: Loop detected! You just executed `{tool_name}` with these exact same arguments. "
                "DO NOT repeat identical tool calls. Look at the previous messages for the data you already retrieved, "
                "then take a DIFFERENT action."
            )
            outs.append(ToolMessage(content=error_msg, tool_call_id=tc["id"], name=tool_name))
            continue
        # ------------------------------------

        try:
            result     = all_tools_map[tool_name].invoke(tool_args)
            str_result = str(result)
        except Exception as e:
            err_str = str(e)
            print(f"\n⚠️ [System] Tool execution failed for {tool_name} (Bad args from LLM?): {err_str[:200]}...")
            error_msg = f"SYSTEM ERROR: Tool '{tool_name}' execution failed due to invalid arguments. Ensure you are providing the EXACT arguments specified in the tool documentation. Error details: {err_str}"
            outs.append(ToolMessage(content=error_msg, tool_call_id=tc["id"], name=tool_name))
            continue

        caller    = tool_to_agent.get(tool_name, "manager")
        max_len   = agent_limit(caller, "max_tool_response_length", 20000)

        if DEBUG_MODE and tool_name in ["search_vault", "read_note", "create_note", "append_note"]:
            print(f"\n📥 [DEBUG - Tool Result ({tool_name})] (cap: {max_len} chars):")
            if len(str_result) > 1000:
                print(f"{str_result[:800]}\n\n[... {len(str_result)-1000} characters truncated in console ...]\n\n{str_result[-200:]}")
            else:
                print(str_result)
            print("-" * 30)

        if len(str_result) > max_len:
            print(f"⚠️  [System] Global Safety Valve: Capping {tool_name} response to {max_len} chars.")
            if tool_name == "search_vault":
                final_content = (str_result[:max_len] +
                                 "\n\n[... SEARCH TRUNCATED ...]\n" +
                                 "ACTION REQUIRED: Use the `read_note` tool on a specific filename above.")
            else:
                final_content = str_result[:max_len] + "\n\n[... TRUNCATED BY GLOBAL SAFETY VALVE ...]"
        else:
            final_content = str_result

        outs.append(ToolMessage(content=final_content, tool_call_id=tc["id"], name=tool_name))

    return {"messages": outs}

# ---------------------------------------------------------------------------
# Router Logic
# ---------------------------------------------------------------------------

def route_from_node(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return "manager"

def route_after_tools(state: AgentState):
    last_msg  = state["messages"][-1]
    tool_name = getattr(last_msg, "name", "")

    routes = {
        "delegate_to_researcher":  "researcher",
        "delegate_to_writer":      "writer",
        "delegate_to_summariser":  "summariser",
        "respond_to_user":         END,
        "ask_clarifying_question": END,
        "update_plan":             "manager",
        "submit_findings":         "manager",
        "submit_summary":          "researcher",
        "finish_writing":          "manager",
        "search_vault":            "researcher",
        "read_note":               "summariser",
        "create_note":             "writer",
        "append_note":             "writer",
    }

    next_node = routes.get(tool_name, "manager")
    if DEBUG_MODE:
        print(f"🚦 [System Router]: Tool '{tool_name}' triggered routing to -> {next_node}")
    return next_node

# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("manager",    manager_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("summariser", summariser_node)
workflow.add_node("writer",     writer_node)
workflow.add_node("tools",      tool_executor)

workflow.add_edge(START, "manager")
workflow.add_conditional_edges("manager",    route_from_node)
workflow.add_conditional_edges("researcher", route_from_node)
workflow.add_conditional_edges("summariser", route_from_node)
workflow.add_conditional_edges("writer",     route_from_node)
workflow.add_conditional_edges("tools",      route_after_tools)

app = workflow.compile(checkpointer=MemorySaver())

# ---------------------------------------------------------------------------
# Chat Loop
# ---------------------------------------------------------------------------

def chat_loop():
    agent_cfg = cfg.get("agents", {})
    profiles  = cfg.get("llm_profiles", {})
    banner_parts = [
        f"Manager: {profiles[agent_cfg['manager']['llm']]['model']}",
        f"Researcher: {profiles[agent_cfg['researcher']['llm']]['model']}",
        f"Summariser: {profiles[agent_cfg['summariser']['llm']]['model']}",
        f"Writer: {profiles[agent_cfg['writer']['llm']]['model']}",
    ]
    print(f"\nTool-Driven Swarm Ready | {' | '.join(banner_parts)} | via LiteLLM Proxy")

    config = {"configurable": {"thread_id": "obsidian_manager_v3"}}
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break

        result   = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        last_msg = result["messages"][-1]

        tool_triggered = getattr(last_msg, "name", "")
        if tool_triggered == "respond_to_user":
            print(f"\nAI: {last_msg.content}")
        elif tool_triggered == "ask_clarifying_question":
            print(f"\nAI (Clarification Needed): {last_msg.content}")
        else:
            print(f"\nAI: [Workflow ended without final response]")

if __name__ == "__main__":
    chat_loop()
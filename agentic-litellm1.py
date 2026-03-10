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
    for agent, agent_cfg in cfg["agents"].items()
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

# ---------------------------------------------------------------------------
# Worker Tools (Vault Actions)
# ---------------------------------------------------------------------------

@tool
def create_note(title: str, content: str) -> str:
    """Create a new note in the Obsidian vault."""
    print(f"\n[Writer Tool] ⚡ Executing CLI to create: '{title}'...")
    try:
        subprocess.run(["obsidian", "create", f"name={title}", f"content={content}"], capture_output=True, text=True, check=True)
        return f"Success: Note saved exactly as '{title}'."
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def append_note(filename: str, content: str) -> str:
    """Append text content to an existing note."""
    try:
        subprocess.run(["obsidian", "append", f"file={filename}", f"content={content}"], capture_output=True, text=True, check=True)
        return f"Success: Appended content to '{filename}'."
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def read_note(filename: str, search_keyword: str = None) -> str:
    """Read a specific note. Optional: provide a 'search_keyword' to extract only relevant snippets from very long notes."""
    print(f"\n[Research Tool] 🔍 Reading note: '{filename}'" + (f" (Keyword: '{search_keyword}')" if search_keyword else "") + "...")
    try:
        result  = subprocess.run(["obsidian", "read", f"file={filename}"], capture_output=True, text=True, check=True)
        content = result.stdout.strip()

        max_len = agent_limit("researcher", "max_tool_response_length", 20000)
        buf     = agent_limit("researcher", "read_note_buffer_size", 5000)

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
    try:
        result = subprocess.run(["obsidian", "search", f"query={query}", "format=text", "limit=5"], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        output = re.sub(r"2026-.*?https://obsidian\.md/download\n*", "", output, flags=re.DOTALL).strip()
        return output if output else "No results found."
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------------------------------------------------------
# Communication Tools (State Routing)
# ---------------------------------------------------------------------------

@tool
def delegate_to_researcher(task: str) -> str:
    """Assign a research task to the Researcher agent to look up information in the vault."""
    return f"System: Task delegated to Researcher -> {task}"

@tool
def delegate_to_writer(task: str) -> str:
    """Assign a saving/writing task to the Writer agent."""
    return f"System: Task delegated to Writer -> {task}"

@tool
def respond_to_user(final_answer: str) -> str:
    """Respond directly to the user to conclude the conversation."""
    return final_answer

@tool
def submit_findings(summary: str) -> str:
    """Submit the final research findings back to the Manager."""
    return f"Researcher Findings: {summary}"

@tool
def finish_writing(confirmation: str) -> str:
    """Confirm to the Manager that the writing task is complete."""
    return f"Writer Confirmation: {confirmation}"

manager_tools    = [delegate_to_researcher, delegate_to_writer, respond_to_user]
researcher_tools = [search_vault, read_note, submit_findings]
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
        try:
            cost = litellm.completion_cost(model=clean_model, prompt_tokens=in_tokens, completion_tokens=out_tokens)
            print(f"💰 [Cost Tracker] {name} used {in_tokens} prompt tokens, {out_tokens} completion tokens. Est. Cost: ${cost:.6f}")
        except Exception:
            pass

    # JSON Catcher — rescue bare JSON tool calls from models that don't format properly
    if not response.tool_calls and (cleaned_content.startswith("{") or cleaned_content.startswith("[")):
        try:
            parsed = json.loads(cleaned_content)
            parsed = [parsed] if isinstance(parsed, dict) else parsed
            for item in parsed:
                item_keys = set(item.keys())
                for tool_name, tool_obj in tool_map.items():
                    expected_keys = set(tool_obj.args_schema.schema()["properties"].keys()) if tool_obj.args_schema else set()
                    if expected_keys and expected_keys.issubset(item_keys):
                        print(f"[System] 🔧 Rescued tool call '{tool_name}' from raw JSON!")
                        response.tool_calls.append({"name": tool_name, "args": item, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"})
                        response.content = ""
                        break
        except json.JSONDecodeError:
            pass

    if not response.tool_calls and not response.content.strip():
        print(f"⚠️ [System] {name} provided empty output. Forcing retry.")
        response.content = "SYSTEM ERROR: You must output a valid tool call."

    return response

def safe_invoke(llm_bound, msgs, name, tool_map):
    """Wraps LLM invocation to catch empty-choice API crashes and handle them gracefully."""
    try:
        response = llm_bound.invoke(msgs)
        return process_reasoning_output(response, name, tool_map)
    except IndexError:
        print(f"\n🛡️ [System Shield] Caught API Empty Response Error (Safety Filter/Parse Failure). Forcing fallback.")
        return AIMessage(content="SYSTEM ERROR: The underlying AI model refused to process the prompt (likely due to a safety filter or parsing bug). Please summarize what you know so far using `submit_findings` or `respond_to_user`.")
    except Exception as e:
        print(f"\n🛡️ [System Shield] Caught API Exception: {str(e)}. Forcing fallback.")
        return AIMessage(content=f"SYSTEM ERROR: The API call failed with error: {str(e)}. Please try a different approach.")

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

    # If safe_invoke returned plain text (no tool calls), bypass and return to Manager
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"messages": []}

    all_tools_map = {t.name: t for t in manager_tools + researcher_tools + writer_tools}

    # Map each tool to the agent that calls it, for per-profile limit lookups
    tool_to_agent = (
        {t.name: "manager"    for t in manager_tools} |
        {t.name: "researcher" for t in researcher_tools} |
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
                "then take a DIFFERENT action (e.g., use `read_note` on a file you found, or `submit_findings`)."
            )
            outs.append(ToolMessage(content=error_msg, tool_call_id=tc["id"], name=tool_name))
            continue
        # ------------------------------------

        result     = all_tools_map[tool_name].invoke(tool_args)
        str_result = str(result)

        # Resolve the response length cap from the receiving agent's LLM profile
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
        "delegate_to_researcher": "researcher",
        "delegate_to_writer":     "writer",
        "respond_to_user":        END,
        "submit_findings":        "manager",
        "finish_writing":         "manager",
        "search_vault":           "researcher",
        "read_note":              "researcher",
        "create_note":            "writer",
        "append_note":            "writer",
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
workflow.add_node("writer",     writer_node)
workflow.add_node("tools",      tool_executor)

workflow.add_edge(START, "manager")
workflow.add_conditional_edges("manager",    route_from_node)
workflow.add_conditional_edges("researcher", route_from_node)
workflow.add_conditional_edges("writer",     route_from_node)
workflow.add_conditional_edges("tools",      route_after_tools)

app = workflow.compile(checkpointer=MemorySaver())

# ---------------------------------------------------------------------------
# Chat Loop
# ---------------------------------------------------------------------------

def chat_loop():
    agent_cfg = cfg["agents"]
    profiles  = cfg["llm_profiles"]
    banner_parts = [
        f"Manager: {profiles[agent_cfg['manager']['llm']]['model']}",
        f"Researcher: {profiles[agent_cfg['researcher']['llm']]['model']}",
        f"Writer: {profiles[agent_cfg['writer']['llm']]['model']}",
    ]
    print(f"\nTool-Driven Swarm Ready | {' | '.join(banner_parts)} | via LiteLLM Proxy")

    config = {"configurable": {"thread_id": "obsidian_manager_v2"}}
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break

        result   = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        last_msg = result["messages"][-1]

        if getattr(last_msg, "name", "") == "respond_to_user":
            print(f"\nAI: {last_msg.content}")
        else:
            print(f"\nAI: [Workflow ended without final response]")

if __name__ == "__main__":
    chat_loop()
import subprocess
import sys
import requests
import operator
import re
import json
import uuid
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)

# --- Configuration ---
#DEFAULT_MODEL = "qwen3:8b"
DEFAULT_MODEL = "qwen3.5:9b"
OLLAMA_URL = "http://192.168.188.159:11434"
DEBUG_MODE = True
MAX_TOOL_RESPONSE_LENGTH = 20000

# --- Worker Tools (Vault Actions) ---

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

MAX_TOOL_RESPONSE_LENGTH = 20000

@tool
def read_note(filename: str, search_keyword: str = None) -> str:
    """Read a specific note. Optional: provide a 'search_keyword' to extract only relevant snippets from very long notes."""
    print(f"\n[Research Tool] 🔍 Reading note: '{filename}'" + (f" (Keyword: '{search_keyword}')" if search_keyword else "") + "...")
    try:
        result = subprocess.run(["obsidian", "read", f"file={filename}"], capture_output=True, text=True, check=True)
        content = result.stdout.strip()
        
        # CONDITION 1: Note is small enough to fit in the context window. Return the whole thing.
        if len(content) <= MAX_TOOL_RESPONSE_LENGTH:
            return content
            
        # CONDITION 2: Note is TOO LONG, and the AI provided a keyword. Extract snippets.
        if search_keyword:
            matches = [m.start() for m in re.finditer(re.escape(search_keyword), content, re.IGNORECASE)]
            
            if not matches:
                # Give the agent a taste of the top of the file so it can adjust its keyword
                return (f"SYSTEM WARNING: The file '{filename}' is {len(content)} characters long, "
                        f"but the keyword '{search_keyword}' was NOT found. "
                        f"Here is the beginning of the file:\n\n{content[:3000]}...")
                
            snippets = []
            
            # grab a buffer with total length of 5000 characters
            buffer_size = 5000
            
            # Limit to the first 5 matches to ensure the final string stays under 20k
            for match_idx in matches[:5]:
                start = max(0, match_idx - buffer_size)
                end = min(len(content), match_idx + len(search_keyword) + buffer_size)
                snippet = content[start:end]
                snippets.append(f"...{snippet}...")
                
            extracted_text = f"--- EXTRACTED SNIPPETS FOR '{search_keyword}' IN '{filename}' (File too large to load entirely) ---\n\n"
            extracted_text += "\n\n[... SNIPPET BREAK ...]\n\n".join(snippets)
            
            return extracted_text
            
        # CONDITION 3: Note is TOO LONG, and NO keyword was provided. Warn the AI.
        else:
            return (content[:MAX_TOOL_RESPONSE_LENGTH] + 
                    "\n\n[... TEXT TRUNCATED BY SYSTEM ...]\n" +
                    f"ACTION REQUIRED: This note is {len(content)} characters long and exceeds the {MAX_TOOL_RESPONSE_LENGTH} limit. " +
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
        # Clean out distracting Obsidian CLI update warnings
        output = re.sub(r"2026-.*?https://obsidian\.md/download\n*", "", output, flags=re.DOTALL).strip()
        return output if output else "No results found."
    except Exception as e:
        return f"Error: {str(e)}"

# --- Communication Tools (State Routing) ---

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

# Grouping tools for the agents
manager_tools = [delegate_to_researcher, delegate_to_writer, respond_to_user]
researcher_tools = [search_vault, read_note, submit_findings]
writer_tools = [create_note, append_note, finish_writing]

# --- Orchestration Helpers ---

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def process_reasoning_output(response: AIMessage, name: str, tool_map: dict) -> AIMessage:
    """Extracts thoughts and rescues raw JSON tool calls."""

    # temp get raw response
    raw_response = response.content or ""

    # temp print raw response
    print(f"\n[DEBUG - {name} Raw Response]:\n{raw_response}")

    raw_content = response.content or ""
    thought_match = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL)
    thoughts = thought_match.group(1).strip() if thought_match else ""
    cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
    
    response.content = cleaned_content
    response.name = name
    
    if DEBUG_MODE and thoughts:
        print(f"\n🧠 [DEBUG - {name} Thoughts]:\n{thoughts}")

    # JSON Catcher: Rescues tool calls if model outputs raw JSON instead of Langchain's format
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
            
    # SAFETY NET: Force a tool call if completely empty
    if not response.tool_calls and not response.content.strip():
        print(f"⚠️ [System] {name} provided empty output. Forcing retry.")
        response.content = "SYSTEM ERROR: You must output a valid tool call."

    return response

# --- Nodes ---

llm = ChatOllama(model=DEFAULT_MODEL, temperature=0, base_url=OLLAMA_URL)

def manager_node(state: AgentState):
    print("\n[Manager] Evaluating task...")
    prompt = """You are the Manager. You MUST use one of your tools to take action.
    You must ground your reponses in research.
    - Use `delegate_to_researcher` if we need information from the vault.
    - Use `delegate_to_writer` if we need to save information to the vault.
    - Use `respond_to_user` if you have enough information to satisfy the user's request.
    NEVER output plain text. ONLY output a tool call."""
    
    msgs = [SystemMessage(content=prompt)] + list(state["messages"])
    response = llm.bind_tools(manager_tools).invoke(msgs)
    response = process_reasoning_output(response, "Manager", {t.name: t for t in manager_tools})
    return {"messages": [response]}

def researcher_node(state: AgentState):
    print("\n[Researcher] Investigating...")
    prompt = """You are the Researcher. 
    1. Use `search_vault` to find relevant files.
    2. Use `read_note` to read specific files.
    3. IMPORTANT: If `read_note` returns a SYSTEM WARNING that the file is too long, you MUST call `read_note` again on the exact same file, but this time include a highly specific `search_keyword` (e.g., "Morana" or "theory") to extract the relevant paragraphs.
    4. When you are done gathering facts, you MUST use the `submit_findings` tool.
    NEVER output plain text. ONLY output a tool call."""
    
    msgs = [SystemMessage(content=prompt)] + [m for m in state["messages"] if isinstance(m, (HumanMessage, ToolMessage))]
    response = llm.bind_tools(researcher_tools).invoke(msgs)
    response = process_reasoning_output(response, "Researcher", {t.name: t for t in researcher_tools})
    return {"messages": [response]}

def writer_node(state: AgentState):
    print("\n[Writer] Writing...")
    prompt = """You are the Writer. 
    1. Use `Notes` or `append_note` to save the requested data.
    2. CRITICAL: When the file is saved successfully, you MUST use the `finish_writing` tool to confirm to the Manager.
    NEVER output plain text. ONLY output a tool call."""
    
    msgs = [SystemMessage(content=prompt)] + [m for m in state["messages"] if isinstance(m, (HumanMessage, ToolMessage))]
    response = llm.bind_tools(writer_tools).invoke(msgs)
    response = process_reasoning_output(response, "Writer", {t.name: t for t in writer_tools})
    return {"messages": [response]}

# --- Shared Tool Executor ---

def tool_executor(state: AgentState):
    """Executes all tools (both Vault actions and Communication triggers)."""
    last_msg = state["messages"][-1]
    all_tools_map = {t.name: t for t in manager_tools + researcher_tools + writer_tools}
    outs = []
    
    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        
        result = all_tools_map[tool_name].invoke(tool_args)
        str_result = str(result)
        
        # VERBOSE DEBUG (Only for actual Vault tools to avoid clutter)
        if DEBUG_MODE and tool_name in ["search_vault", "read_note", "create_note", "append_note"]:
            print(f"\n📥 [DEBUG - Tool Result ({tool_name})]:")
            if len(str_result) > 1000:
                print(f"{str_result[:800]}\n\n[... {len(str_result)-1000} characters truncated in console ...]\n\n{str_result[-200:]}")
            else:
                print(str_result)
            print("-" * 30)

        # GLOBAL CAPPING SAFETY VALVE
        # (read_note handles itself now, but this protects against runaway search_vaults)
        if len(str_result) > MAX_TOOL_RESPONSE_LENGTH:
            print(f"⚠️  [System] Global Safety Valve: Capping {tool_name} response to {MAX_TOOL_RESPONSE_LENGTH} chars.")
            
            if tool_name == "search_vault":
                final_content = (
                    str_result[:MAX_TOOL_RESPONSE_LENGTH] + 
                    "\n\n[... SEARCH TRUNCATED ...]\n" +
                    "ACTION REQUIRED: Use the `read_note` tool on a specific filename above."
                )
            else:
                # Fallback for any other tool that goes rogue
                final_content = str_result[:MAX_TOOL_RESPONSE_LENGTH] + "\n\n[... TRUNCATED BY GLOBAL SAFETY VALVE ...]"
        else:
            final_content = str_result
            
        # IMPORTANT: We pass the tool_name here so the Router knows what happened!
        outs.append(ToolMessage(content=final_content, tool_call_id=tc["id"], name=tool_name))
        
    return {"messages": outs}

# --- Router Logic ---

def route_from_node(state: AgentState):
    """Ensure nodes always jump to the Tool Executor."""
    if state["messages"][-1].tool_calls:
        return "tools"
    # If the model glitches and doesn't call a tool, force it back to Manager to fix it
    return "manager"

def route_after_tools(state: AgentState):
    """Mathematically route based on the exact tool that was just executed."""
    last_msg = state["messages"][-1]
    tool_name = getattr(last_msg, "name", "")

    # This is the "Brain" of the architecture. Hard-coded, deterministic paths.
    routes = {
        "delegate_to_researcher": "researcher",
        "delegate_to_writer": "writer",
        "respond_to_user": END,              # If manager responds, graph finishes
        "submit_findings": "manager",        # Researcher is done -> back to Manager
        "finish_writing": "manager",         # Writer is done -> back to Manager
        "search_vault": "researcher",        # Researcher tools loop back to Researcher
        "read_note": "researcher",
        "create_note": "writer",             # Writer tools loop back to Writer
        "append_note": "writer"
    }
    
    next_node = routes.get(tool_name, "manager")
    if DEBUG_MODE: print(f"🚦 [System Router]: Tool '{tool_name}' triggered routing to -> {next_node}")
    return next_node

# --- Graph Construction ---

workflow = StateGraph(AgentState)

workflow.add_node("manager", manager_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("tools", tool_executor)

workflow.add_edge(START, "manager")

# All agents send their logic into the Tools node
workflow.add_conditional_edges("manager", route_from_node)
workflow.add_conditional_edges("researcher", route_from_node)
workflow.add_conditional_edges("writer", route_from_node)

# The Tools node mathematically routes to the next agent based on the tool used
workflow.add_conditional_edges("tools", route_after_tools)

app = workflow.compile(checkpointer=MemorySaver())

# --- Chat Loop ---

def chat_loop():
    print(f"\nTool-Driven Swarm Ready (Model: '{DEFAULT_MODEL}')")
    config = {"configurable": {"thread_id": "obsidian_manager_v2"}}
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']: break
        
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        
        # Extract the final answer sent via the 'respond_to_user' tool
        last_msg = result["messages"][-1]
        if getattr(last_msg, "name", "") == "respond_to_user":
            print(f"\nAI: {last_msg.content}")
        else:
            print(f"\nAI: [Workflow ended without final response]")

if __name__ == "__main__":
    chat_loop()
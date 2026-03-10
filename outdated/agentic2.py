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
DEFAULT_MODEL = "qwen3:8b"
OLLAMA_URL = "http://192.168.188.159:11434"
DEBUG_MODE = True
MAX_TOOL_RESPONSE_LENGTH = 10000

# --- Tools ---

@tool
def create_note(title: str, content: str) -> str:
    """Create a new note in the Obsidian vault."""
    print(f"\n[Writer Tool] ⚡ Executing CLI to create: '{title}'...")
    try:
        subprocess.run(["obsidian", "create", f"name={title}", f"content={content}"], capture_output=True, text=True, check=True)
        print(f"\n[System] ✅ SUCCESS: File physically created at '{title}'")
        return f"Success: Note saved exactly as '{title}'."
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def append_note(filename: str, content: str) -> str:
    """Append text content to an existing note."""
    try:
        subprocess.run(["obsidian", "append", f"file={filename}", f"content={content}"], capture_output=True, text=True, check=True)
        print(f"\n[System] ✅ SUCCESS: File updated at '{filename}'")
        return f"Success: Appended content to '{filename}'."
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def read_note(filename: str) -> str:
    """Read a specific note."""
    print(f"\n[Research Tool] 🔍 Reading note: '{filename}'...")
    try:
        result = subprocess.run(["obsidian", "read", f"file={filename}"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def search_vault(query: str) -> str:
    """Search the vault for filenames containing the query. Returns a list of matching file paths."""
    print(f"\n[Research Tool] 🔍 Searching vault for: '{query}'...")
    try:
        # Use a simpler search that only returns file paths to avoid context bloat
        result = subprocess.run(["obsidian", "search", f"query={query}", "format=text", "limit=5"], capture_output=True, text=True, check=True)
        return result.stdout.strip() if result.stdout.strip() else "No results found."
    except Exception as e:
        return f"Error: {str(e)}"

# --- Orchestration Helpers ---

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_node: str  # Tracks which worker to go to next

def process_reasoning_output(response: AIMessage, name: str) -> AIMessage:
    """Extracts thoughts and cleans content for reasoning models."""
    raw_content = response.content or ""
    thought_match = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL)
    thoughts = thought_match.group(1).strip() if thought_match else ""
    cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
    
    response.content = cleaned_content
    response.name = name
    
    if DEBUG_MODE and thoughts:
        print(f"\n🧠 [DEBUG - {name} Thoughts]:\n{thoughts}")
    return response

# --- Nodes ---

llm = ChatOllama(model=DEFAULT_MODEL, temperature=0, base_url=OLLAMA_URL)

def manager_node(state: AgentState):
    print("\n[Manager] Evaluating task...")
    prompt = """You are the Manager. Your job is to decide the next step.
    1. If info is needed from the vault, reply ONLY with: ROUTE: researcher
    2. If info is ready to be saved, reply ONLY with: ROUTE: writer
    3. If the user's request is fully satisfied, reply with your final answer to the user."""
    
    messages = [SystemMessage(content=prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    response = process_reasoning_output(response, "Manager")

    if DEBUG_MODE:
        print(f"🎯 [DEBUG - Manager Route Decision]: {response.content}")

    if "ROUTE: researcher" in response.content:
        return {"next_node": "researcher", "messages": [response]}
    elif "ROUTE: writer" in response.content:
        return {"next_node": "writer", "messages": [response]}
    
    return {"next_node": END, "messages": [response]}

def researcher_node(state: AgentState):
    print("\n[Researcher] Investigating...")
    prompt = """You are the Researcher. 
    1. Start by using `search_vault` to get a list of relevant files.
    2. Once you see the list, use `read_note` to read the full content of the most relevant file.
    3. You may use tools multiple times in a row if needed.
    4. Once you have read the details and gathered the facts, output a final text summary and STOP calling tools. This will hand control back to the Manager."""
    
    msgs = [SystemMessage(content=prompt)] + [m for m in state["messages"] if isinstance(m, (HumanMessage, ToolMessage))]
    
    response = llm.bind_tools([search_vault, read_note]).invoke(msgs)
    response = process_reasoning_output(response, "Researcher")
    return {"messages": [response]}

def writer_node(state: AgentState):
    print("\n[Writer] Writing...")
    prompt = "You are the Writer. Use create_note or append_note to save data. When done, confirm and stop."
    msgs = [SystemMessage(content=prompt)] + [m for m in state["messages"] if isinstance(m, (HumanMessage, ToolMessage))]
    
    response = llm.bind_tools([create_note, append_note]).invoke(msgs)
    response = process_reasoning_output(response, "Writer")
    return {"messages": [response]}

# --- Tool Executor (Shared) ---

def tool_executor(state: AgentState):
    """Executes tool calls, logs them to console, and caps context size for performance."""
    last_msg = state["messages"][-1]
    tool_map = {t.name: t for t in [search_vault, read_note, create_note, append_note]}
    outs = []
    
    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        
        # 1. Execute the tool
        result = tool_map[tool_name].invoke(tool_args)
        str_result = str(result)
        
        # 2. VERBOSE DEBUG LOGGING (For your terminal)
        if DEBUG_MODE:
            print(f"\n📥 [DEBUG - Tool Result ({tool_name})]:")
            if len(str_result) > 1000:
                print(f"{str_result[:800]}\n\n[... {len(str_result)-1000} characters truncated in console ...]\n\n{str_result[-200:]}")
            else:
                print(str_result)
            print("-" * 30)

        # 3. CONTEXT SAFETY VALVE (For the LLM's brain)
        # We cap the content sent to the state at 5,000 chars. 
        # This prevents the "300k character" lag you encountered.
        if len(str_result) > MAX_TOOL_RESPONSE_LENGTH:
            print(f"⚠️  [System] Capping tool response for {tool_name} to {MAX_TOOL_RESPONSE_LENGTH} chars to prevent lag.")
            final_content = (
                str_result[:MAX_TOOL_RESPONSE_LENGTH] + 
                "\n\n[... TEXT TRUNCATED BY SYSTEM: Result too long for context window. ...]\n" +
                "ACTION REQUIRED: Use the 'read_note' tool on a specific filename above to see the full content."
            )
        else:
            final_content = str_result
            
        outs.append(ToolMessage(content=final_content, tool_call_id=tc["id"]))
        
    return {"messages": outs}
# --- Router Logic ---

def route_after_worker(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return "manager" # Go back to manager to see if more work is needed

def route_after_tools(state: AgentState):
    """Routes the graph back to the specific agent that called the tool."""
    # Look backwards to find the last AI message that made a tool call
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if msg.name == "Researcher":
                return "researcher"
            elif msg.name == "Writer":
                return "writer"
    
    # Fallback just in case
    return "manager"

# --- Graph Construction ---

workflow = StateGraph(AgentState)

workflow.add_node("manager", manager_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("tools", tool_executor)

workflow.add_edge(START, "manager")

# Manager routes to workers
workflow.add_conditional_edges(
    "manager", 
    lambda x: x["next_node"],
    {"researcher": "researcher", "writer": "writer", END: END}
)

# Workers go to tools or back to manager
workflow.add_conditional_edges("researcher", route_after_worker)
workflow.add_conditional_edges("writer", route_after_worker)
workflow.add_conditional_edges("tools", route_after_tools)

app = workflow.compile(checkpointer=MemorySaver())

# --- Chat Loop ---

def chat_loop():
    print(f"\nManager-Worker Swarm Ready (Model: '{DEFAULT_MODEL}')")
    config = {"configurable": {"thread_id": "obsidian_manager_v1"}}
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']: break
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        print(f"\nAI: {result['messages'][-1].content}")

if __name__ == "__main__":
    chat_loop()
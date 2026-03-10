import subprocess
import sys
import requests
from typing import Annotated, Sequence, TypedDict
import operator
import re        
import json      
import uuid

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

def check_ollama_status():
    print("[System] Checking connection to Ollama...")
    tags_url = f"{OLLAMA_URL}/api/tags"
    try:
        response = requests.get(tags_url)
        response.raise_for_status()
        models_data = response.json()
        available_models = [model['name'] for model in models_data.get('models', [])]
        
        if DEFAULT_MODEL not in available_models and f"{DEFAULT_MODEL}:latest" not in available_models:
            print(f"[System] ❌ Error: Model '{DEFAULT_MODEL}' is not available in Ollama.")
            return False
        else:
            print(f"[System] ✅ Model '{DEFAULT_MODEL}' is available in Ollama.")
        return True
    except Exception:
        print("[System] ❌ Error: Could not connect to Ollama.")
        return False

# --- LangChain Tools ---

@tool
def create_note(title: str, content: str) -> str:
    """Create a new note in the Obsidian vault."""
    print(f"\n[Writer Tool] ⚡ Executing CLI to create: '{title}'...")
    try:
        subprocess.run(["obsidian", "create", f"name={title}", f"content={content}"], capture_output=True, text=True, check=True)
        print(f"\n[System] ✅ SUCCESS: File physically created at '{title}'")
        return f"Success: Note saved exactly as '{title}'."
    except subprocess.CalledProcessError as e:
        return f"Error creating note: {e.stderr.strip() if e.stderr else str(e)}"

@tool
def append_note(filename: str, content: str) -> str:
    """Append text content to an existing note in the Obsidian vault."""
    try:
        subprocess.run(["obsidian", "append", f"file={filename}", f"content={content}"], capture_output=True, text=True, check=True)
        print(f"\n[System] ✅ SUCCESS: File physically updated at '{filename}'")
        return f"Success: Appended content to '{filename}'."
    except subprocess.CalledProcessError as e:
        return f"Error appending to note: {e.stderr.strip() if e.stderr else str(e)}"

@tool
def read_note(filename: str) -> str:
    """Read and return the text content of a specific note."""
    print(f"\n[Research Tool] 🔍 Reading note: '{filename}'...")
    try:
        result = subprocess.run(["obsidian", "read", f"file={filename}"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error reading note: {e.stderr.strip() if e.stderr else str(e)}"

@tool
def search_vault(query: str) -> str:
    """Search the entire Obsidian vault for specific text or keywords."""
    print(f"\n[Research Tool] 🔍 Searching vault for: '{query}'...")
    try:
        result = subprocess.run(["obsidian", "search:context", f"query={query}", "limit=10"], capture_output=True, text=True, check=True)
        return result.stdout.strip() if result.stdout.strip() else "No results found."
    except subprocess.CalledProcessError as e:
        return f"Error searching vault: {e.stderr.strip() if e.stderr else str(e)}"

research_tools = [search_vault, read_note]
writer_tools = [create_note, append_note]

# --- Interceptor Helper ---

def process_reasoning_output(response: AIMessage, tool_map: dict = None) -> AIMessage:
    """Strips <think> tags, extracts the thought process, and rescues raw JSON tool calls."""
    raw_content = response.content or ""
    
    thought_match = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL)
    thoughts = thought_match.group(1).strip() if thought_match else ""
    
    cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
    response.content = cleaned_content
    
    if DEBUG_MODE and thoughts:
        print(f"\n🧠 [DEBUG - {response.name} Thoughts]:\n{thoughts}")

    if response.tool_calls:
        return response

    if tool_map and (cleaned_content.startswith("{") or cleaned_content.startswith("[")):
        try:
            parsed = json.loads(cleaned_content)
            if isinstance(parsed, dict):
                parsed = [parsed]
                
            for item in parsed:
                item_keys = set(item.keys())
                for tool_name, tool_obj in tool_map.items():
                    expected_keys = set(tool_obj.args_schema.schema()["properties"].keys()) if tool_obj.args_schema else set()
                    
                    if expected_keys and expected_keys.issubset(item_keys):
                        print(f"[System] 🔧 Rescued tool call '{tool_name}' from raw JSON!")
                        response.tool_calls.append({
                            "name": tool_name,
                            "args": item,
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "tool_call"
                        })
                        response.content = "" 
                        break 
        except json.JSONDecodeError:
            pass 
        
    if not response.tool_calls and not response.content.strip():
        print("[System] ⚠️ Model output thoughts but forgot to provide a final answer. Forcing retry...")
        response.content = "SYSTEM ERROR: You provided no output outside of your <think> tags. You MUST provide your final text or tool call after closing the </think> tag."        
    
    return response

# --- LangGraph Setup ---

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def get_clean_messages(messages: list, new_system_prompt: str) -> list:
    cleaned_history = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue # We will add the fresh system prompt at the end
            
        if isinstance(m, AIMessage):
            # We keep the AI's actual answer but strip the <think> block
            # This allows the Writer to know what the Researcher 'found' 
            # without getting lost in the Researcher's internal monologue.
            clean_content = re.sub(r'<think>.*?</think>', '', m.content, flags=re.DOTALL).strip()
            if clean_content:
                cleaned_history.append(AIMessage(content=clean_content, name=getattr(m, "name", "Assistant")))
        else:
            # Always keep HumanMessages (user input) and ToolMessages (actual data)
            cleaned_history.append(m)
            
    return [SystemMessage(content=new_system_prompt)] + cleaned_history

llm_base = ChatOllama(model=DEFAULT_MODEL, temperature=0, base_url=OLLAMA_URL)
llm_researcher = llm_base.bind_tools(research_tools)
llm_writer = llm_base.bind_tools(writer_tools)

# --- Node Definitions ---

def planner_node(state: AgentState):
    print("\n[Planner] Drafting execution strategy...")
    prompt = """You are the Planner Agent. Analyze the user's request. Output a step-by-step plan detailing what information needs to be searched or read from the Obsidian vault, and what notes need to be created or appended. Do not use tools, just output the text plan. Keep thinking very short"""
    messages = get_clean_messages(state["messages"], prompt)
    
    response = llm_base.invoke(messages)
    response.name = "Planner"
    
    response = process_reasoning_output(response)
    response.content = f"**Plan:**\n{response.content}"
    print(f"\n[Planner] {response.content}")
    return {"messages": [response]}

def research_node(state: AgentState):
    print("\n[Researcher] Processing plan and investigating vault...")
    prompt = """You are the Researcher Agent. Your ONLY job is to execute the Planner's strategy using your tools.
    CRITICAL INSTRUCTIONS:
    1. You MUST immediately invoke the `search_vault` tool to search for the keywords requested. Output the request as raw JSON if necessary.
    2. Do NOT output hypothetical text. You must ACTUALLY use the tool to find out!
    3. If the search returns file paths, you MUST use the `read_note` tool to read their contents.
    4. Only AFTER you have successfully used the tools and gathered real data, output a final text summary for the Writer Agent.
    5. YOUR FINAL SUMMARY MUST BE WRITTEN IN PLAIN TEXT OUTSIDE OF ANY <think> TAGS.
    6. Keep thinking very short"""
    
    messages = get_clean_messages(state["messages"], prompt)
    
    response = llm_researcher.invoke(messages)
    response.name = "Researcher"
    
    tool_map = {t.name: t for t in research_tools}
    response = process_reasoning_output(response, tool_map)
    
    if DEBUG_MODE and response.tool_calls:
        print(f"\n🛠️ [DEBUG - Researcher Tool Calls]:\n{response.tool_calls}")
            
    return {"messages": [response]}

def writer_node(state: AgentState):
    print("\n[Writer] Preparing to update vault...")
    # FIXED: "Notes" changed back to "create_note"
    prompt = """You are the Writer Agent. Your ONLY job is to save the Researcher's findings into the vault.
    CRITICAL INSTRUCTIONS:
    1. You MUST invoke the `create_note` or `append_note` tool immediately. Output the request as raw JSON if necessary.
    2. NEVER output the raw summary text directly to the user.
    3. Once the tool executes successfully, state the EXACT filename where the information was saved.
    4. ALL TOOL CALLS OR FINAL CONFIRMATION TEXT MUST BE OUTSIDE OF ANY <think> TAGS.
    5. Keep thinking very short"""
    
    messages = get_clean_messages(state["messages"], prompt)
    
    response = llm_writer.invoke(messages)
    response.name = "Writer"
    
    tool_map = {t.name: t for t in writer_tools}
    response = process_reasoning_output(response, tool_map)
    
    if DEBUG_MODE and response.tool_calls:
        print(f"\n🛠️ [DEBUG - Writer Tool Calls]:\n{response.tool_calls}")
            
    return {"messages": [response]}

def research_tool_executor(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []
    tool_map = {tool.name: tool for tool in research_tools}
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        if tool_name in tool_map:
            result = tool_map[tool_name].invoke(tool_call["args"])
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name))
            if DEBUG_MODE:
                print(f"\n🛠️ [DEBUG - Researcher Tool Result]:\n{result}")
            
    return {"messages": tool_messages}

def writer_tool_executor(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []
    tool_map = {tool.name: tool for tool in writer_tools}
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        if tool_name == 'append_note':
            filename = tool_args.get('filename', 'Unknown')
            content = tool_args.get('content', '')
            
            print(f"\n⚠️  [Security Alert] The AI wants to modify an existing file.")
            print(f"Target File: {filename}")
            print("Content to Append:\n" + ("="*40) + f"\n{content}\n" + ("="*40))
            
            confirmation = input("Do you allow this action? (y/n): ")
            
            if confirmation.lower() != 'y':
                print("[System] 🛑 Action cancelled by user.")
                tool_messages.append(ToolMessage(content="Error: The user denied permission.", tool_call_id=tool_id, name=tool_name))
                continue
        
        if tool_name in tool_map:
            result = tool_map[tool_name].invoke(tool_args)
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name))
            
    return {"messages": tool_messages}

# --- Routing Functions ---

def research_router(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "research_tools"
    return "writer"

def writer_router(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "writer_tools"
    # End the graph if the Writer didn't use a tool (e.g., it is just giving final confirmation)
    return END

# --- Build the Graph ---

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", research_node)
workflow.add_node("research_tools", research_tool_executor)
workflow.add_node("writer", writer_node)
workflow.add_node("writer_tools", writer_tool_executor)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")

workflow.add_conditional_edges("researcher", research_router)
workflow.add_edge("research_tools", "researcher")

workflow.add_conditional_edges("writer", writer_router)
workflow.add_edge("writer_tools", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- Chat Application ---

def chat_loop():
    if not check_ollama_status():
        sys.exit(1)
        
    print(f"\nConnecting to Multi-Agent Swarm (Model: '{DEFAULT_MODEL}')")
    print("Type 'quit' or 'exit' to stop.")
    
    config = {"configurable": {"thread_id": "obsidian_multi_agent_1"}}
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        try:
            result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
            
            for msg in reversed(result["messages"]):
                if getattr(msg, "name", "") == "Writer" and msg.content:
                    if not DEBUG_MODE:
                        print(f"\nAI (Writer): {msg.content}")
                    break
                
        except Exception as e:
            print(f"\n[System] ❌ Communication error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    chat_loop()
import subprocess
import sys
from ollama import chat, ChatResponse

# --- Configuration ---
DEFAULT_MODEL = "qwen3:8b"

SYSTEM_PROMPT = """You are a helpful local AI assistant connected to the user's Obsidian vault. 
You can answer questions normally, and you have access to tools to interact with the vault.
When asked to manage notes, use the appropriate tools to read, search, create, or append to them.
"""

# --- Tool Definitions ---

def create_note(title: str, content: str) -> str:
    """Create a new note in the Obsidian vault.
    
    Args:
        title: The title or filename of the new note.
        content: The text content to write into the note.
        
    Returns:
        A string indicating the success or failure of the operation.
    """
    print(f"\n[System] ⚡ Creating new note: '{title}'...")
    try:
        subprocess.run(["obsidian", "create", f"name={title}", f"content={content}"], capture_output=True, text=True, check=True)
        return f"Success: Note '{title}' created."
    except subprocess.CalledProcessError as e:
        return f"Error creating note: {e.stderr.strip() if e.stderr else str(e)}"
    except FileNotFoundError:
        return "Error: 'obsidian' CLI tool not found."

def append_note(filename: str, content: str) -> str:
    """Append text content to an existing note in the Obsidian vault.
    
    Args:
        filename: The exact name of the file to modify.
        content: The text to add to the bottom of the note.
        
    Returns:
        A string indicating success or failure.
    """
    try:
        subprocess.run(["obsidian", "append", f"file={filename}", f"content={content}"], capture_output=True, text=True, check=True)
        print(f"[System] ✅ Successfully appended to '{filename}'.")
        return f"Success: Appended content to '{filename}'."
    except subprocess.CalledProcessError as e:
        return f"Error appending to note: {e.stderr.strip() if e.stderr else str(e)}"

def read_note(filename: str) -> str:
    """Read and return the text content of a specific note.
    
    Args:
        filename: The name of the file to read.
        
    Returns:
        The text content of the note, or an error message.
    """
    print(f"\n[System] 🔍 Reading note: '{filename}'...")
    try:
        result = subprocess.run(["obsidian", "read", f"file={filename}"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error reading note: {e.stderr.strip() if e.stderr else str(e)}"

def search_vault(query: str) -> str:
    """Search the entire Obsidian vault for specific text or keywords.
    
    Args:
        query: The keyword or phrase to search for.
        
    Returns:
        A string containing the search results and matching files.
    """
    print(f"\n[System] 🔍 Searching vault for: '{query}'...")
    try:
        result = subprocess.run(["obsidian", "search", f"query={query}", "format=text", "limit=10"], capture_output=True, text=True, check=True)
        return result.stdout.strip() if result.stdout.strip() else "No results found."
    except subprocess.CalledProcessError as e:
        return f"Error searching vault: {e.stderr.strip() if e.stderr else str(e)}"

# Map function names to the actual Python functions
AVAILABLE_FUNCTIONS = {
    'create_note': create_note,
    'append_note': append_note,
    'read_note': read_note,
    'search_vault': search_vault
}

TOOLS_LIST = list(AVAILABLE_FUNCTIONS.values())

def chat_loop():
    print(f"\nConnecting to Ollama model '{DEFAULT_MODEL}' via Python SDK...")
    print("Type 'quit' or 'exit' to stop.")
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        messages.append({"role": "user", "content": user_input})
        
        while True:
            try:
                response: ChatResponse = chat(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    tools=TOOLS_LIST
                )
                
                messages.append(response.message)
                
                if response.message.tool_calls:
                    for tool_call in response.message.tool_calls:
                        func_name = tool_call.function.name
                        
                        if func_name in AVAILABLE_FUNCTIONS:
                            func = AVAILABLE_FUNCTIONS[func_name]
                            kwargs = tool_call.function.arguments
                            
                            # --- SECURE CONFIRMATION FOR EDITS ONLY ---
                            if func_name == 'append_note':
                                filename = kwargs.get('filename', 'Unknown')
                                content = kwargs.get('content', '')
                                
                                print(f"\n⚠️  [Security Alert] The AI wants to modify an existing file.")
                                print(f"Target File: {filename}")
                                print("Content to Append:\n" + ("="*40) + f"\n{content}\n" + ("="*40))
                                
                                confirmation = input("Do you allow this action? (y/n): ")
                                
                                if confirmation.lower() != 'y':
                                    print("[System] 🛑 Action cancelled by user.")
                                    messages.append({
                                        'role': 'tool', 
                                        'tool_name': func_name, 
                                        'content': "Error: The user denied permission to modify the file."
                                    })
                                    continue # Move to the next tool call if there are multiple
                            # ------------------------------------------
                            
                            # Execute the tool and save the result
                            result = func(**kwargs)
                            messages.append({
                                'role': 'tool', 
                                'tool_name': func_name, 
                                'content': str(result)
                            })
                        else:
                            print(f"[System] Warning: Model tried to call unknown tool '{func_name}'")
                            messages.append({
                                'role': 'tool', 
                                'tool_name': func_name, 
                                'content': "Error: Unknown tool."
                            })
                else:
                    if response.message.content:
                        print(f"\nAI: {response.message.content}")
                    break
                    
            except Exception as e:
                print(f"\n[System] ❌ Communication error: {e}")
                sys.exit(1)

if __name__ == "__main__":
    chat_loop()
import requests
import json
import re
import subprocess
import sys

# --- Configuration ---
# Replace with the actual IP address of the machine hosting Ollama on your LAN
OLLAMA_URL = "http://localhost:11434/api/chat" 
DEFAULT_MODEL = "qwen3:8b"

# The system prompt instructs the model on how to interact with the user and the tools
SYSTEM_PROMPT = """You are a helpful local AI assistant connected to the user's Obsidian vault. 
You can answer questions normally, but you also have the ability to create notes.
If the user asks you to create a note, or if you think saving information to a note would be highly beneficial, you must output your request using the exact following format:
<CREATE_NOTE title="The Title of the Note">The content of the note goes here.</CREATE_NOTE>
You can include regular conversational text before or after this tag.
"""

def check_ollama_status():
    """Checks if Ollama is running and the specified model is available."""
    print("[System] Checking connection to Ollama...")
    
    # Extract the base URL to check the tags endpoint
    base_url = OLLAMA_URL.replace("/api/chat", "")
    tags_url = f"{base_url}/api/tags"
    
    try:
        response = requests.get(tags_url)
        response.raise_for_status()
        
        models_data = response.json()
        available_models = [model['name'] for model in models_data.get('models', [])]
        
        # Check if the exact model or the model with ':latest' exists
        if DEFAULT_MODEL not in available_models and f"{DEFAULT_MODEL}:latest" not in available_models:
            print(f"[System] ❌ Error: Model '{DEFAULT_MODEL}' is not available in Ollama.")
            print(f"[System] Please run 'ollama pull {DEFAULT_MODEL}' in your terminal first.")
            return False
            
        print(f"[System] ✅ Ollama is running and model '{DEFAULT_MODEL}' is ready.")
        return True
        
    except requests.exceptions.ConnectionError:
        print("[System] ❌ Error: Could not connect to Ollama.")
        print("[System] Please make sure the Ollama application is running.")
        return False
    except Exception as e:
        print(f"[System] ❌ An unexpected error occurred during the check: {e}")
        return False

def create_note_via_cli(title, content):
    """Executes the Obsidian CLI command to create a note."""
    print(f"\n[System] Attempting to create note: '{title}'...")
    try:
        # Updated to use the official Obsidian CLI syntax
        subprocess.run(["obsidian", "create", f"name={title}", f"content={content}"], check=True)
        print(f"[System] ✅ Successfully created note: '{title}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[System] ❌ CLI execution failed: {e}")
        return False
    except FileNotFoundError:
        print("[System] ❌ Error: 'obsidian' command not found. Please ensure it is in your PATH.")
        return False

def extract_and_execute_commands(text):
    """Parses the model's response for the CREATE_NOTE tag and executes it."""
    # Regex to find the title attribute and the content between the tags
    pattern = re.compile(r'<CREATE_NOTE title="(.*?)">(.*?)</CREATE_NOTE>', re.DOTALL)
    matches = pattern.findall(text)
    
    for match in matches:
        title = match[0].strip()
        content = match[1].strip()
        create_note_via_cli(title, content)
        
    # Return the cleaned text without the XML tags so the chat output looks natural
    cleaned_text = pattern.sub('\n[Note creation triggered]\n', text)
    return cleaned_text

def chat_loop():
    # Run the pre-check before doing anything else
    if not check_ollama_status():
        sys.exit(1)
        
    print(f"\nConnecting to Ollama model '{DEFAULT_MODEL}' at {OLLAMA_URL}...")
    print("Type 'quit' or 'exit' to stop.")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        messages.append({"role": "user", "content": user_input})
        
        payload = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "stream": False
        }
        
        try:
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            response_data = response.json()
            
            ai_message = response_data.get("message", {}).get("content", "")
            
            # Process any commands the model decided to output
            display_text = extract_and_execute_commands(ai_message)
            
            print(f"\nAI: {display_text}")
            
            # Save the raw AI response to history so the model remembers its own formatting
            messages.append({"role": "assistant", "content": ai_message})
            
        except requests.exceptions.RequestException as e:
            print(f"\n[System] ❌ Failed to communicate with Ollama: {e}")
            sys.exit(1)

if __name__ == "__main__":
    chat_loop()
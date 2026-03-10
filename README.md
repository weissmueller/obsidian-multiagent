# Obsidian LLM Swarm Agent

This project is a multi-agent system (Manager, Researcher, Writer) that operates on your local Obsidian vault. It uses LangChain, LangGraph, and LiteLLM to search, read, create, and append notes natively in Obsidian via the Obsidian CLI.

## Prerequisites
- **Python 3.9+**
- **Obsidian** installed locally with the CLI capabilities enabled.
- **LiteLLM / Ollama** running locally or a remote LLM provider (configured via proxy).

## Installation

1. **Clone the repository and enter the directory**
   ```bash
   git clone <repository-url>
   cd obsidian-ollama
   ```

2. **Set up a Python Virtual Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Environment Variables**
   Create a `.env` file in the root directory and add your LiteLLM proxy API key (or other provider keys):
   ```env
   LITELLM_API_KEY="your-api-key-here"
   ```

2. **System Config**
   Edit `config.yaml` to specify your LLM endpoints, profiles, context windows, and agent assignments.
   ```yaml
   litellm:
     base_url: "http://192.168.188.161:9027/v1"
     api_key: "your-api-key" # Or use the .env variable
   ```

## Pulling Local Models (Ollama)

If you are using local models via Ollama, you will need to pull them before running the script. Based on the default config, you might need to pull the models you plan to use:

```bash
# Example: Pulling a Qwen model
ollama pull qwen2.5

# Example: Pulling Llama 3
ollama pull llama3
```
*(Make sure the model names in Ollama match the `model` fields in your `config.yaml`'s `llm_profiles`, handling any custom tags or local naming conventions).*

## Usage

Ensure your virtual environment is activated and simply run the main script:

```bash
python agentic-litellm1.py
```

The system will initialize the Swarm (Manager, Researcher, Writer) and you can start typing natural language requests to interrogate your vault or have it write new notes!

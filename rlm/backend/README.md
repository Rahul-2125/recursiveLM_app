# RLM - Recursive Language Model

A Python implementation of Recursive Language Models that can process arbitrarily long contexts by storing them externally in a REPL environment.

## Project Structure

```
rlm/
├── main.py              # Entry point & CLI
├── requirements.txt     # Dependencies
├── .env                 # Environment variables (not committed)
│
├── config/              # Configuration
│   ├── __init__.py
│   └── settings.py      # Centralized settings
│
├── core/                # Core RLM logic
│   ├── __init__.py
│   ├── base.py          # Abstract RLM base class
│   ├── engine.py        # RLM_REPL implementation
│   └── repl_env.py      # REPL environment
│
├── services/            # External integrations
│   ├── __init__.py
│   └── llm_client.py    # LLM client wrapper
│
├── prompts/             # Prompt templates
│   ├── __init__.py
│   └── templates.py     # System & action prompts
│
├── utils/               # Utilities
│   ├── __init__.py
│   ├── logging.py       # Formatted logging
│   ├── parsing.py       # Text parsing
│   └── tracing.py       # Execution tracing
│
├── tests/               # Test suite
│   └── __init__.py
│
└── logs/                # Log output directory
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your credentials:

```env
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_API_KEY=your-api-key
NVIDIA_MODEL_NAME=qwen/qwen2.5-coder-32b-instruct
```

## Usage

### As a Library

```python
from rlm import run

result = run(
    context="Your long context here...",
    query="What is the main purpose of this code?",
    max_iterations=30,
)
print(result)
```

### Command Line

```bash
python main.py --query "What does this code do?" --context "def hello(): print('hi')"

# Or with a file
python main.py --query "Summarize this code" --context-file path/to/code.py
```

## How RLM Works

1. **Context Storage**: Context is stored externally in a REPL environment (not passed to the model directly)
2. **Code Generation**: LLM writes `\`\`\`repl\`\`\`` code blocks to access/analyze context
3. **Execution**: Code executes in a sandboxed Python REPL
4. **Recursive Calls**: LLM can call `llm_query()` for sub-LLM queries
5. **Iteration**: Loop continues until LLM returns `FINAL(answer)`

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     RLM Engine                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌─────────────┐    ┌────────────┐ │
│  │  ROOT LLM   │────▶│  REPL Env   │◀───│  SUB-LLM   │ │
│  └─────────────┘     └─────────────┘    └────────────┘ │
│        │                   │                   ▲       │
│        │              ┌────┴────┐              │       │
│        ▼              │ Context │        llm_query()   │
│   Code Blocks         │ Storage │              │       │
│   Extraction          └─────────┘              │       │
│        │                                       │       │
│        └───────────Execute──────────────────▶ │       │
└─────────────────────────────────────────────────────────┘
```

## License

MIT

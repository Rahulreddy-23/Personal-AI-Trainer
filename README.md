# Personal AI Trainer

**Personal AI Trainer** is a Retrieval-Augmented Generation (RAG) system built with Python and LangChain that allows you to query your personal documents (PDFs, text files) using natural language. This project enables private and secure document-based question answering over your own data, using free and local models (no OpenAI API required).

## Features

- Ingests `.txt` and `.pdf` documents
- Splits, embeds, and indexes text using HuggingFace embeddings (runs locally, free)
- Stores vectors in Chroma for semantic search
- Enables command-line querying via Retriever+LLM (local LLMs via Ollama, e.g., Llama 3, Mistral, Phi-3)
- Modular codebase for easy extensibility

## Use Cases

- Analyze and summarize resumes or reports
- Build a searchable knowledge base from notes or documentation
- Ask questions about personal or professional documents

## Installation

```bash
git clone https://github.com/Rahulreddy-23/Personal-AI-Trainer.git
cd Personal-AI-Trainer
pip install -r requirements.txt
```

## Setup

1. **Install Ollama** (for free local LLMs):  
   Download and install from [https://ollama.com/download](https://ollama.com/download)

2. **Pull a model (e.g., Llama 3):**
   ```bash
   ollama pull llama3
   ```

3. ** Add your documents:**  
   Place `.pdf` or `.txt` files in the `data/` directory.

## Usage

### 1. Ingest your documents

```bash
python src/ingest.py
```

### 2. Query your documents

```bash
python src/query.py
```

You can now ask questions about your documents in the terminal.  
Type `exit` to quit.

## Notes

- All embeddings and LLM inference run locally; no API keys or paid services required.
- Supports easy extension to other local LLMs via Ollama (e.g., `mistral`, `phi3`).
- For best results, keep your documents in plain text or PDF format.

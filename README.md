# Ollama RAG Application

A bunch of AI tests using retrieval-augmented generation (RAG) application using Ollama, LangChain, and Streamlit.

## Overview

All tests were made using ollama and Lancgchain, the main ideia of pdf*.py files was to allows users to upload PDF documents, which are then processed and stored in a vector database. Users can ask questions about the document content, and the application uses a large language model (LLM) to generate context-aware responses.

## Features

- PDF document upload and processing
- Vector storage for efficient document retrieval
- Semantic search using embeddings
- Multi-query retrieval for better search results
- Conversation history tracking
- Customizable LLM parameters
- Temperature control for response creativity
- Multiple implementation examples for learning purposes
- Specialized prompts for technical document analysis
- Example scripts demonstrating various Ollama API features

## Requirements

- Python 3.10+
- Ollama server running locally
- Required models:
  - An embedding model (default: `mxbai-embed-large`)
  - A RAG model (default: `llama3.2`)

## Installation

1. Clone the repository
2. Set up Virtual environment


## Setting Up Virtual Environment

It's recommended to use a virtual environment to avoid package conflicts:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
## On Linux/macOS
source venv/bin/activate
## On Windows
venv\Scripts\activate

# Install dependencies in the virtual environment
pip3 install -r requirements.txt
```

> ```bash
> # On Linux/macOS
> source venv_py310/bin/activate
> # On Windows
> venv_py310\Scripts\activate
> ```

When you're done working with the project, you can deactivate the virtual environment:

```bash
deactivate
```

4. Ensure Ollama is installed and running:

```bash
# Download necessary models
ollama pull mxbai-embed-large
ollama pull llama3.2
```

## Running the Application

> **Important:** Make sure you have activated the virtual environment before running any scripts.

### Main Streamlit Applications

Run the main application with:

```bash
streamlit run streamlit_app.py
```

Alternative version with temperature control:

```bash
streamlit run pdf-rag-streamlit.py
```

### Running Core RAG Scripts

For command-line usage without the Streamlit interface:

```bash
python3 pdf-rag.py
```

Or try the simplified version:

```bash
python3 pdf-rag-clean.py
```

### Example Scripts

Run the simple chat example:

```bash
python3 1-chat_llama.py
```

Text generation with streaming:

```bash
python3 2-generate_llama.py
```

Game categorization example:

```bash
python3 3-categorize.py
```

## Configuration

### Models
The application uses the following models by default:
- Embedding model: `mxbai-embed-large`
- RAG model: `llama3.2`

You can select different models through the UI dropdown menus or modify the model names in the scripts.

### Parameters
Various parameters can be customized:
- Temperature: Controls randomness in the responses (0.0-1.0)
- Chunk size: Size of text chunks (default: 1500)
- Chunk overlap: Overlap between chunks (default: 300)
- Collection name: Name used for vector database (adjustable in each script)

### Custom Models
You can create custom models using Modelfiles as shown in the `Modelfile` example:

```bash
# Build a custom model
ollama create mycustommodel -f Modelfile

# Use the custom model
ollama run mycustommodel
```

## Project Structure

- `streamlit_app.py`: Streamlit application with a user-friendly interface for PDF uploading and querying
- `pdf-rag-streamlit.py`: Alternative implementation of the Streamlit application with additional features like temperature control
- `pdf-rag.py`: Core implementation of the PDF RAG functionality without the Streamlit interface
- `pdf-rag-clean.py`: Simplified version of the PDF RAG implementation
- `1-chat_llama.py`: Simple example of using Ollama's chat API
- `2-generate_llama.py`: Example of streaming text generation with Ollama
- `3-categorize.py`: Example application that categorizes game titles using LLM
- `Modelfile`: Example configuration file for creating a custom Ollama model
- `requirements.txt`: List of Python dependencies required for the project
- `requirements_freeze.txt`: Comprehensive list of all dependencies with exact versions
- `chroma_db/`: Directory for storing vector database files
- `data/`: Directory containing sample data files for examples

## Troubleshooting

If you encounter errors related to PyTorch or Streamlit's file monitoring, you may need to adjust your environment setup. Common errors include RuntimeError with "no running event loop" which is typically a compatibility issue between PyTorch and Streamlit.

## License

MIT

## made buy @danilomarcus

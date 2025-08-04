# Multimodal RAG System with Qwen2.5-VL

A multimodal Retrieval-Augmented Generation (RAG) system that processes PDFs containing text, images, diagrams, and workflow images using Ollama, Qwen2.5-VL, LangChain, and ChromaDB.

## Features

- **Multimodal PDF Processing**: Extracts both text and images from PDF documents
- **Vision-Language Model**: Uses Qwen2.5-VL via Ollama for understanding images and diagrams
- **Vector Database**: ChromaDB for efficient similarity search
- **FastAPI Server**: RESTful API for document upload and querying
- **LangChain Integration**: Seamless workflow orchestration

## Prerequisites

1. **Python 3.11+**: Ensure you have Python 3.11 or higher installed
2. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
3. **Pull required models**:
   ```bash
   ollama pull qwen2.5-vl:7b
   ollama pull nomic-embed-text
   ```

## Installation

### Quick Setup

1. Clone or download this repository
2. Run the installation script:
   ```bash
   python install.py
   ```
3. Install and setup Ollama:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull qwen2.5-vl:7b
   ```
4. Start the system:
   ```bash
   python start.py
   ```

### Manual Installation

1. Ensure Python 3.11+ is installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or
   pip install -e .
   ```

## Usage

1. **Start the system**:
   ```bash
   python start.py
   ```
   This will check all requirements and start the FastAPI server.

2. **Alternative - Direct start**:
   ```bash
   python main.py
   ```

3. **Upload and query via web interface**:
   - Visit http://localhost:8000/docs for interactive API documentation
   - Upload PDFs and query the system through the web interface

4. **Test the system**:
   ```bash
   python test_system.py
   ```

## API Endpoints

- `POST /upload`: Upload PDF documents for processing
- `POST /query`: Query the multimodal RAG system
- `GET /health`: Health check endpoint
- `GET /docs`: Interactive API documentation

## Architecture

The system processes PDFs by:
1. Extracting text content using PyMuPDF
2. Extracting images and diagrams
3. Generating embeddings for text using Nomic Embed Text via Ollama
4. Analyzing images using Qwen2.5-VL via Ollama
5. Storing everything in ChromaDB
6. Retrieving relevant context for user queries
7. Generating responses using the multimodal model

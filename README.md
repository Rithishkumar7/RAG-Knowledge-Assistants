# Multi-Format Document RAG System

This repository contains a **Retrieval-Augmented Generation (RAG)** system built with LangChain and Python that can ingest and query documents in multiple formats (such as PDF, Word, text, and others) to provide intelligent, context-aware answers.

## Features

- Supports multiple document formats through pluggable loaders (for example, PDF, DOCX, text).
- Uses embeddings and a vector store to retrieve relevant document chunks.
- Integrates LangChain to connect retrieval with an LLM for answer generation.
- Configurable through Python code and project settings (see `app.py`, `main.py`, and `pyproject.toml`). [page:4]
- Dependency management with `requirements.txt` / `uv.lock` for reproducible environments. [page:4]

## Project structure

Key files and folders: [page:4]

- `src/` – Source code for the RAG pipeline, document loading, chunking, and query handling.
- `app.py` – Application entry-point (for example, FastAPI, Streamlit, or CLI) that exposes the RAG functionality. [page:4]
- `main.py` – Main script that initializes the RAG components and wiring. [page:4]
- `pyproject.toml` – Project configuration and packaging metadata.
- `requirements.txt` – List of Python dependencies needed to run the project. [page:4]
- `uv.lock` – Lock file ensuring deterministic dependency resolution.
- `.gitignore` – Git ignore rules to keep unnecessary files out of version control. [page:4]

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Rithishkumar7/Multi-Format-Document-RAG-System.git
   cd Multi-Format-Document-RAG-System
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or, if you use `uv` with `pyproject.toml` / `uv.lock`:

   ```bash
   uv sync
   ```

## Running the application

Depending on how you structured the app, use one of these approaches (update to match your real usage):

- If `app.py` runs a web app or API:

  ```bash
  python app.py
  ```

- If `main.py` is the CLI entry-point:

  ```bash
  python main.py
  ```

After starting the app, open the URL or follow the CLI prompts to upload documents and ask questions based on their content.

## How the RAG pipeline works

- Documents are loaded from multiple formats using appropriate loaders.
- Text is chunked and converted into embeddings using an embedding model.
- A vector store indexes the chunks for efficient similarity search.
- For each query, the system retrieves the most relevant chunks and passes them, along with the question, to an LLM through LangChain to generate an answer.

## Configuration and customization

- Modify `src/` modules to change chunk size, embedding model, or vector store.
- Adjust prompts and chains in the code to refine answer style and quality.
- Extend document loaders to support additional formats as needed.

## License

Specify your chosen license here (for example, MIT) and add a `LICENSE` file to the repository.

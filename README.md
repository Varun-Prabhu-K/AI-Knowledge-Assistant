# AI Knowledge Assistant

A lightweight Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their contents.

## Features

- Upload one or multiple PDF documents
- Extract text from PDFs
- Split documents into overlapping chunks
- Generate semantic embeddings
- Store and retrieve chunks using ChromaDB
- Generate answers using Gemini
- Display retrieved source pages
- Simple Streamlit interface

## Architecture

User uploads PDF  
↓  
PyMuPDF — Text Extraction  
↓  
Cleaning + Chunking — 250 words / 100 overlap  
↓  
Sentence Transformers — Embeddings  
↓  
ChromaDB — Vector Store + Metadata  
↓  
Query Embedding + Similarity Retrieval  
↓  
Retrieved Chunks + Neighboring Context  
↓  
Gemini 2.5 Flash  
↓  
Answer + Sources — Streamlit UI

## Tech Stack

- Python
- PyMuPDF
- Sentence Transformers
- ChromaDB
- Google Gemini API
- Streamlit

## How It Works

PDF → Text Extraction → Chunking → Embeddings → ChromaDB → Relevant Chunk Retrieval → Gemini → Answer

The user's question is converted into an embedding and used to retrieve semantically relevant document chunks. Neighboring chunks are also included to provide additional context. The retrieved content is then provided to Gemini so that the generated answer is based primarily on the uploaded documents.

## Setup

### 1. Clone the repository

    git clone https://github.com/Varun-Prabhu-K/AI-Knowledge-Assistant.git
    cd AI-Knowledge-Assistant

### 2. Create a virtual environment

    python -m venv venv

### 3. Activate the virtual environment

Windows:

    venv\Scripts\activate

### 4. Install dependencies

    pip install -r requirements.txt

### 5. Configure Gemini API key

Create a `.env` file in the project root:

    GEMINI_API_KEY=your_api_key_here

### 6. Run the application

    streamlit run app.py

## Project Structure

    AI-Knowledge-Assistant/
    ├── app.py
    ├── src/
    │   ├── ingestion.py
    │   ├── embeddings.py
    │   ├── vector_store.py
    │   ├── retrieval.py
    │   └── generation.py
    ├── documents/
    ├── requirements.txt
    ├── .gitignore
    └── README.md

## Limitations

The current implementation is primarily designed for text-based PDF content. Scanned PDFs and complex visual information such as tables or diagrams may require OCR or additional processing.

## Author

Developed as a mini AI Knowledge Assistant / RAG project using Python, open-source retrieval components, and Gemini for generation.
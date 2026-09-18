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

                 ┌─────────────────────┐
                 │   User uploads PDF  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      PyMuPDF        │
                 │    Text Extraction  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Cleaning + Chunking │
                 │   250 words / 100   │
                 │       overlap       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │Sentence Transformers│
                 │    Embeddings       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      ChromaDB       │
                 │ Vector Store + Meta │
                 └──────────┬──────────┘
                            │
                      User question
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Query Embedding +   │
                 │ Similarity Retrieval│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Retrieved Chunks +  │
                 │ Neighboring Context │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Gemini 2.5      │
                 │       Flash         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Answer + Sources   │
                 │    Streamlit UI     │
                 └─────────────────────┘
```

## Tech Stack

- Python
- PyMuPDF
- Sentence Transformers
- ChromaDB
- Google Gemini API
- Streamlit

## How It Works

PDF → Text Extraction → Chunking → Embeddings → ChromaDB → Relevant Chunk Retrieval → Gemini → Answer

The user's question is converted into an embedding and used to retrieve semantically relevant document chunks. These chunks are provided to Gemini as context so that the generated answer is based primarily on the uploaded
documents.

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd AI-Knowledge-Assistant

## Author

Developed as a mini AI Knowledge Assistant / RAG project using Python and open-source retrieval components with Gemini for generation.

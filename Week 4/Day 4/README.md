# Team Knowledge Base API

A production-grade RAG system for querying team documents.

## What it does

- Ingest PDF and plain text documents via API
- Query across all documents using hybrid search
- Returns grounded answers with source citations

## Tech stack

- FastAPI — REST API layer
- Qdrant — vector database
- sentence-transformers — embeddings (all-MiniLM-L6-v2)
- BM25 + dense vectors + RRF — hybrid search
- CrossEncoder — re-ranking
- Groq LLaMA 3.3 70B — answer generation

## Endpoints

- `POST /ingest` — upload and index a document (.txt or .pdf)
- `POST /query` — ask a question, get a grounded answer
- `GET /documents` — list all ingested documents
- `GET /health` — server status

## Authentication

All endpoints require header: `x-api-key: knowledge-base-2024`

## How to run

```bash
pip install -r requirements.txt
python novamind_capstone.py
```

Open http://localhost:8000/docs for interactive API documentation.

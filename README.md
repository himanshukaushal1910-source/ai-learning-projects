# AI Learning Projects 🤖

> 4-week intensive AI engineering program — building production-grade AI systems from scratch.
> Every project is built concept-first: deep understanding before code, then deployed.

---

## About This Repository

This repository documents a structured 4-week journey from AI fundamentals to production deployment.
The focus is on building real, working AI systems — not just following tutorials.

Each project is built concept-first: deep understanding before code, then applied to real problems.
This repository contains production-grade AI projects covering semantic search, RAG pipelines,
hybrid retrieval, prompt security, full API deployment, Docker containerisation, and cloud deployment.

---

## Live Demo

🔗 **NovaMind Knowledge Base (live):** https://himanshukaushal1910-novamind-rag-api.hf.space

Ask questions about NovaMind's internal documents. Powered by RAG — retrieval augmented generation
with hybrid search, Qdrant, and LLaMA 3.3 70B via Groq.

---

## Projects

| #   | Project                          | Description                                                                                                                                     | Tech Stack                                    | Status             |
| --- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------ |
| 1   | **NovaMind Semantic Search**     | Semantic search over NovaMind internal documents using vector embeddings and cosine similarity. Metadata filtering by source.                   | sentence-transformers, ChromaDB               | ✅ Complete        |
| 2   | **NovaMind Qdrant Search**       | Production-grade semantic search using Qdrant with Docker server mode. HNSW indexing, persistent storage.                                       | sentence-transformers, Qdrant                 | ✅ Complete        |
| 3   | **NovaMind RAG Pipeline**        | Full RAG pipeline from scratch — chunk, embed, retrieve, generate. Answers grounded in NovaMind documents with inline citations.                | Qdrant, Groq, LLaMA 3.3 70B                   | ✅ Complete        |
| 4   | **PDF RAG Pipeline**             | Upload any PDF and ask questions. Page-level citations showing exactly which page the answer came from. Full error handling.                    | PyMuPDF, Qdrant, Groq                         | ✅ Portfolio ready |
| 5   | **Hybrid Search RAG**            | BM25 sparse search + dense semantic search combined with Reciprocal Rank Fusion. Handles exact codes and semantic queries simultaneously.       | Qdrant, rank-bm25, Groq                       | ✅ Complete        |
| 6   | **Re-ranking + LangChain**       | Cross-encoder re-ranking on top of hybrid retrieval. Three pipelines compared side by side: dense, reranking, LangChain LCEL.                   | sentence-transformers CrossEncoder, LangChain | ✅ Complete        |
| 7   | **Parent-Child RAG + RAGAS**     | Parent-child chunking to fix split-chunk retrieval failures. RAGAS evaluation with 10-question golden dataset. 10/10 accuracy.                  | Qdrant, RAGAS, Groq                           | ✅ Complete        |
| 8   | **Secure RAG — Injection Audit** | Four-layer prompt injection defence: input sanitisation, chunk sanitisation, injection-resistant prompt, output filtering. Full attack battery. | Qdrant, Groq, regex                           | ✅ Complete        |
| 9   | **Classical ML Literacy**        | Churn prediction model comparing Logistic Regression vs Random Forest. Feature importance analysis, precision/recall evaluation.                | scikit-learn, pandas                          | ✅ Complete        |
| 10  | **FastAPI RAG API**              | RAG pipeline exposed as a REST API with Pydantic validation, API key auth, health endpoint, async endpoints, CORS.                              | FastAPI, uvicorn, Pydantic, Qdrant, Groq      | ✅ Complete        |
| 11  | **Docker Full App**              | FastAPI RAG app + persistent Qdrant running together via docker-compose. Single command starts everything. Vectors survive restarts.            | Docker, docker-compose, Qdrant volumes        | ✅ Complete        |
| 12  | **NovaMind Knowledge Base**      | Live Gradio UI deployed to Hugging Face Spaces. Ask questions via browser — no API knowledge required. Public URL, free hosting.               | Gradio, Qdrant, Groq, Hugging Face Spaces     | ✅ Live            |
| 13  | **Team Knowledge Base API**      | Production-grade multi-document knowledge base. Ingest PDFs and text, hybrid search (BM25 + dense + RRF), cross-encoder reranking, 4 endpoints. | FastAPI, Qdrant, rank-bm25, CrossEncoder, Groq | ✅ Complete       |

---

## Tech Stack

### AI and ML

![sentence-transformers](https://img.shields.io/badge/sentence--transformers-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.3-blue)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

### Vector Databases

![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C)

### Deployment and Infrastructure

![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)

---

## Repository Structure

```
ai-learning-projects/
  Week 1/
    Day 1/    — Embeddings, cosine similarity, vector space intuition
    Day 2/    — Embedding models, tokenisation, mean pooling, model comparison
    Day 3/    — Chunking strategies, Git setup
    Day 4/    — ChromaDB, metadata filtering, NovaMind semantic search
  Week 2/
    Day 1/    — Qdrant internals, HNSW indexing, Docker server mode
    Day 2/    — LLM API setup, Groq adoption (see API decision below)
    Day 3/    — Full RAG pipeline v1 and v2 with citations
    Day 4/    — PDF ingestion, page fingerprinting, error handling
  Week 3/
    Day 1/    — Hybrid search: BM25 + dense + RRF fusion
    Day 2/    — Cross-encoder reranking + LangChain LCEL pipeline
    Day 3/    — Parent-child chunking + RAGAS evaluation
    Day 4/    — Prompt injection audit: 4-layer security architecture
    Day 5/    — Classical ML literacy: sklearn churn prediction
  Week 4/
    Day 1/    — FastAPI RAG API with auth, Pydantic validation, health endpoint
    Day 2/    — Docker full app: FastAPI + persistent Qdrant via docker-compose
    Day 3/    — Gradio UI deployed to Hugging Face Spaces (live demo)
    Day 4/    — Capstone: Team Knowledge Base API — multi-doc, hybrid search, reranking
    Day 5/    — Portfolio polish, interview prep, program complete
```

---

## API Decision — Why Groq Instead of OpenAI

OpenAI requires paid credits — not available at the start of this program.

Gemini free tier was attempted but abandoned: deprecated `google.generativeai` library,
wrong model names, and daily quota exhausted during debugging.

**Groq was adopted as the primary LLM API:**

- Free tier with generous rate limits — no quota issues during development
- OpenAI-compatible syntax — one line change to switch: `Groq()` → `OpenAI()`
- Model: `llama-3.3-70b-versatile` — comparable to GPT-4o-mini for RAG tasks

---

## Key Concepts Covered

**Week 1 — Foundations**

- Vector embeddings and semantic similarity
- Chunking strategies: fixed, recursive, semantic, document structure, sentence window
- ChromaDB and Qdrant for vector storage and metadata filtering
- Git and GitHub for version control

**Week 2 — RAG Systems**

- HNSW indexing and approximate nearest neighbour search
- LLM fundamentals: tokens, context window, temperature, hallucination
- RAG pipeline from scratch: chunk → embed → retrieve → generate
- PDF ingestion with page-level citations
- Prompt engineering: 5 rules for RAG system prompts

**Week 3 — Advanced RAG + Security**

- Hybrid search: BM25 sparse + dense semantic, fused with RRF
- Two-stage retrieval: bi-encoder retrieval + cross-encoder reranking
- LangChain LCEL pipeline — modern pipe-based chain construction
- Parent-child chunking: small chunks for retrieval, large for context
- RAGAS evaluation: faithfulness, answer relevancy, context precision, recall
- Prompt injection: direct and indirect attacks, 4-layer defence architecture
- Classical ML literacy: supervised/unsupervised, classification/regression, sklearn

**Week 4 — Production**

- FastAPI REST API with Pydantic validation, authentication, async endpoints
- Docker containerisation: Dockerfile, docker-compose, persistent volumes
- Hugging Face Spaces deployment with Gradio UI — live public URL
- Capstone: multi-document ingestion, hybrid search, reranking, full API

---

## Learning Approach

Every topic follows the same structure:

1. **Concept** — deep intuition before any code
2. **Tool** — see the concept in a real working tool
3. **Build** — apply it to the NovaMind project context
4. **Depth** — edge cases, failure modes, tradeoffs
5. **Review** — self-test and curriculum check every day

---

## About

**Himanshu Kaushal**

AI Engineer. JEE Advanced qualified. Built 13 production-grade AI projects across a structured
4-week intensive program covering RAG pipelines, vector databases, LLM integration, prompt security,
and cloud deployment.

- 📧 himanshukaushal1910@gmail.com
- 🐙 https://github.com/himanshukaushal1910-source/
- 🤗 https://huggingface.co/himanshukaushal1910

---

## Progress

- [x] Week 1 — Foundations + Semantic Search
- [x] Week 2 — Vector DBs + LLMs + RAG Pipeline
- [x] Week 3 — Advanced RAG + Security + ML Literacy
- [x] Week 4 — Deployment + Production + Portfolio ✅

---

_4-week intensive program complete._

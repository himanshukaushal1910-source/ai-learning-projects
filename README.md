# AI Learning Projects 🤖

> 4-week intensive AI engineering program — building production-grade AI systems from scratch.
> Every project is deployed, documented and portfolio-ready.

---

## About This Repository

This repository documents a structured 4-week journey from AI fundamentals to production deployment.
The focus is on building real, working AI systems — not just following tutorials.

Each project is built concept-first: deep understanding before code, then deployed to a live URL.
By end of week 4 this repository will contain 5 production-grade AI projects covering semantic search,
RAG pipelines, recommendation systems, AI agents and full API deployment.

---

## Projects

| # | Project | Description | Tech Stack | Status |
|---|---------|-------------|------------|--------|
| 1 | **Mini Semantic Search** | Semantic search over text data using vector embeddings and cosine similarity. Supports metadata filtering and CLI interface. | sentence-transformers, ChromaDB, numpy | 🔨 In Progress |
| 2 | **PDF Q&A Chatbot** | Upload any PDF and ask questions. Answers include citations showing exactly which page the answer came from. | OpenAI, ChromaDB, LangChain | 📅 Week 2 |
| 3 | **Product Recommendation Engine** | Content-based recommendation system using embedding similarity. Given any product, returns top 5 most similar products with scores. | sentence-transformers, Qdrant | 📅 Week 3 |
| 4 | **AI Research Agent** | Autonomous research agent that searches the web and queries a local knowledge base to synthesise answers from multiple sources. | LangChain, LangGraph, Qdrant | 📅 Week 3 |
| 5 | **Team Knowledge Base API** | Production-grade knowledge base API. Ingest PDFs and documents, search by meaning, get answers with sources. Deployed with FastAPI and Docker. | Qdrant, FastAPI, Docker, OpenAI | 📅 Week 4 |

---

## Tech Stack

### AI and ML
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C)

### Vector Databases
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C)

### Deployment and Infrastructure
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)

---

## Repository Structure

```
ai-learning-projects/
  Week1/
    Day1/               — Python for data, vectors, cosine similarity
    Day2/               — Embedding models, tokenisation, mean pooling
    Day3/               — Git, chunking strategy, semantic search
    Day4/               — ChromaDB, vector storage, metadata filtering
    Day5/               — Project polish, mini semantic search complete
  Week2/                — Coming: Vector DB internals, LLMs, RAG pipeline
  Week3/                — Coming: Hybrid search, agents, prompt security
  Week4/                — Coming: FastAPI, Docker, cloud deployment
  novamind_story.md     — Running narrative of every concept learned
```

---

## Key Concepts Covered

**Week 1 — Foundations**
- Vector embeddings and semantic similarity
- How transformer models convert text to vectors (tokenisation → attention → pooling)
- Chunking strategies for RAG systems
- Vector databases — ChromaDB and Qdrant

**Week 2 — RAG Systems**
- LLM fundamentals and OpenAI API
- RAG pipeline design from scratch
- LangChain abstractions and re-ranking
- Classical ML literacy

**Week 3 — Advanced Patterns**
- Hybrid search (BM25 + dense vectors)
- AI agents and the ReAct loop
- Prompt security and injection defence
- RAG evaluation with RAGAS

**Week 4 — Production**
- FastAPI for AI system deployment
- Docker containerisation
- Cloud deployment (Railway / Render)
- Portfolio and interview preparation

---

## Learning Approach

Every topic follows the same structure:
1. **Concept** — deep intuition before any code
2. **Tool** — see the concept in a real working tool
3. **Build** — apply it to the week's project
4. **Depth** — edge cases, failure modes, tradeoffs
5. **Deploy** — ship something real every week

---

## About the Developer

**[Himanshu Kaushal]**

[Add 2-3 sentences about yourself — your background, what you're transitioning from or into, and what excites you about AI engineering.]

- 📧 [himanshukaushal1910@gmail.com]
- 💼 [LinkedIn URL]
- 🐙 [https://github.com/himanshukaushal1910-source/]

---

## Progress

- [x] Week 1 — Foundations + Mini Semantic Search
- [ ] Week 2 — Vector DBs + LLMs + RAG Pipeline
- [ ] Week 3 — Advanced RAG + Agents + Security
- [ ] Week 4 — Deployment + Production + Portfolio

---

*Updated weekly as projects are completed and deployed.*

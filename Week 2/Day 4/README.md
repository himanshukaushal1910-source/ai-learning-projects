# PDF RAG Pipeline

A retrieval-augmented generation (RAG) system that lets you ask questions about any PDF document and get grounded, cited answers. Built with Qdrant, sentence-transformers, and Groq (LLaMA 3.3 70B).

---

## What it does

- Drop in any PDF document
- Ask questions in plain English
- Get answers grounded strictly in the document — no hallucination
- Every answer includes citations showing which page the information came from
- If the answer isn't in the document, the system says so instead of making something up

---

## How it works

```
PDF → extract text (PyMuPDF) → clean → chunk (LangChain)
                                              ↓
                                    embed (all-MiniLM-L6-v2)
                                              ↓
                                    store in Qdrant (in-memory)
                                              ↓
                              user question → embed → search Qdrant
                                              ↓
                                    top 3 chunks retrieved
                                              ↓
                              build prompt with context + question
                                              ↓
                                  Groq API (LLaMA 3.3 70B)
                                              ↓
                              grounded answer + page citations
```

---

## Requirements

- Python 3.10+
- A free Groq API key — get one at [console.groq.com](https://console.groq.com)
- A digital native PDF (not scanned/image-only)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/ai-learning-projects.git
cd ai-learning-projects/Week\ 2/Day\ 4

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key (Windows)
set GROQ_API_KEY=your-key-here

# Set your Groq API key (Mac/Linux)
export GROQ_API_KEY=your-key-here
```

---

## Usage

**Default — uses novamind_sample.pdf in the same folder:**
```bash
python pdf_rag.py
```

**With a specific PDF:**
```bash
python pdf_rag.py path/to/your/document.pdf
```

**Example session:**
```
PDF RAG READY — report.pdf (12 pages, 47 chunks)
Type your question. Type 'exit' to quit.

Ask a question: What are the main revenue streams?

QUESTION: What are the main revenue streams?
──────────────────────────────────────────────────────────────
ANSWER:

According to [Chunk 2], the three main revenue streams are...

── SOURCES ───────────────────────────────────────────────────
  [Chunk 2]  Page: 3  Score: 0.8123
             "The company generates revenue through three primary..."
─────────────────────────────────────────────────────────────

Ask a question: exit
Session ended.
```

---

## Models used

| Component | Model | Notes |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 | 384-dimensional, runs locally |
| LLM | LLaMA 3.3 70B (via Groq) | Free API, fast inference |
| Vector DB | Qdrant in-memory | Resets on script exit |

---

## Limitations

- **Scanned PDFs not supported** — image-only PDFs return no extractable text. OCR is required for scanned documents (not included).
- **In-memory storage** — Qdrant runs in-memory. The index is rebuilt every time the script runs. For persistent storage, switch to Qdrant server mode via Docker.
- **Single document** — one PDF per session. Multi-document support would require a persistent Qdrant collection with source metadata.
- **English only** — all-MiniLM-L6-v2 works best with English text. Other languages may have reduced retrieval quality.
- **Complex tables** — table content may be extracted in a degraded format. Structured table extraction requires additional tooling.
- **Context window** — top 3 chunks retrieved per query. For complex questions spanning many sections, increase `top_k` in the `retrieve()` function.

---

## Project structure

```
Week 2/Day 4/
├── pdf_rag.py           # main RAG pipeline
├── requirements.txt     # dependencies
├── README.md            # this file
└── novamind_sample.pdf  # sample document for testing
```

---

## Built during

Week 2, Day 4 of a 4-week intensive AI engineering program.
Part of the NovaMind AI application developer learning track.

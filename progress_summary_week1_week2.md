# AI Application Developer — Progress Summary
## Week 1 + Week 2 Complete

**Program:** Lane 1 — AI Application Developer (4-week intensive)
**Learner:** Harsh Kaushal
**Approach:** Concept-first, intent-driven. No syntax memorisation. Understand → generate → understand.
**Project thread:** NovaMind (fictional AI productivity company) — used as real-world context across all builds.

---

## WEEK 1 — Foundations: Embeddings, Vectors, Semantic Search

### Day 1 — Embeddings + Vectors
**Concepts learned:**
- What embeddings are — converting meaning into numbers (vectors)
- Why vectors enable semantic search — similar meaning = similar direction in vector space
- Cosine similarity — measuring angle between vectors, not distance
- The library analogy — librarian who understands meaning vs keyword matching

**Tools used:**
- sentence-transformers — all-MiniLM-L6-v2 (384 dimensions)
- First embedding generated and understood

**Build:**
- First semantic similarity script — embedded sentences, compared cosine similarity manually

**Key insight:** Embeddings don't store words, they store meaning. Two sentences can use different words and have high similarity if they mean the same thing.

---

### Day 2 — Chunking Strategy
**Concepts learned:**
- Why chunking exists — LLMs and vector DBs work on chunks, not whole documents
- Chunk size vs chunk overlap tradeoff
- Why overlap exists — preserves context across chunk boundaries
- Minimum length filtering — removes header-only junk chunks

**Tools used:**
- LangChain RecursiveCharacterTextSplitter
- chunk_size=400, chunk_overlap=80, minimum 100 chars

**Build:**
- Chunked novamind_sample.txt — 13 raw chunks → 11 after filter
- Understood why bad chunking = bad retrieval downstream

**Key insight:** Chunking is the most underrated decision in RAG. Perfect embeddings on bad chunks = bad retrieval.

---

### Day 3 — Chroma Vector Database
**Concepts learned:**
- What a vector database is — stores vectors + metadata, enables similarity search
- Collections — like tables in a relational DB
- Metadata filtering — retrieving by both similarity AND structured attributes
- Distance vs similarity scores — Chroma returns distance (lower = better)

**Tools used:**
- ChromaDB — in-memory mode
- First semantic search: embed query → search collection → return top chunks

**Build:**
- NovaMind semantic search on Chroma
- 4 test queries, live query loop
- Understood retrieval quality vs retrieval failure

**Key insight:** Chroma is a great local scratchpad. It hides the embedding step — you pass text, it embeds internally. Good for learning, not for production.

---

### Day 4 — Advanced Retrieval + Metadata
**Concepts learned:**
- Metadata filtering — combining vector search with structured filters
- Multi-source retrieval — tagging chunks with source document
- Score thresholds — filtering results below a minimum similarity

**Build:**
- NovaMind search with 4 sources + metadata filters
- Source-aware retrieval — "only search engineering documents"

---

### Day 5 — Friday Review
- Full Week 1 concepts consolidated
- Git + GitHub — portfolio repository created
- All Week 1 projects pushed to GitHub
- Self-test completed across all concepts

**Week 1 portfolio output:**
- `novamind_chroma_search.py` — semantic search on Chroma with live query loop
- GitHub repo live with Week 1 commits

---

## WEEK 2 — LLMs, RAG Pipeline, PDF Ingestion

### Day 1 — Qdrant Vector Database
**Concepts learned:**
- HNSW indexing — Hierarchical Navigable Small World
- The city navigation analogy — layered graph, big jumps at top, fine steps at bottom
- ANN (Approximate Nearest Neighbour) vs exact search — why ANN exists at scale
- ef parameter — speed vs recall tradeoff
- Vector DB decision framework:
  - Chroma → prototyping
  - Qdrant → production, self-hosted, full control
  - Pinecone → managed cloud, no infrastructure
  - pgvector → already on PostgreSQL, want simplicity

**Tools used:**
- Qdrant in-memory mode — QdrantClient(":memory:")
- PointStruct — id + vector + payload
- query_points() — HNSW search
- Docker — Qdrant as real server, localhost:6333

**Build:**
- NovaMind search rebuilt on Qdrant (was Chroma in Week 1)
- Proved persistence — data survived between script runs on Docker server

**Key differences from Chroma:**
- You embed manually — Qdrant only stores/searches vectors
- Distance metric declared at collection creation, not query time
- Similarity score (higher = better) vs Chroma distance (lower = better)

**Key insight:** Chroma was your scratchpad. Qdrant is your production-grade vector store. The embedding logic doesn't change — only the database API changes.

---

### Day 2 — LLMs + Prompt Engineering
**Concepts learned:**
- Tokens — chunks of text with unique IDs, everything measured in tokens
- Context window — the desk analogy, fixed capacity, explains why RAG exists
- Hallucination — autocomplete at scale, model predicts most likely token not correct token
- Temperature — 0 = deterministic/factual, 1 = creative, >1 = increasingly random
- Prompt engineering — 5 rules for RAG:
  1. Ground the model explicitly with ONLY
  2. Give the model an escape hatch
  3. Define persona and tone specifically
  4. Control output format
  5. Use clear delimiters (XML-style tags)

**Tools used:**
- Groq API — free tier, LLaMA 3.3 70B
- OpenAI-compatible syntax — transferable to OpenAI with one line change
- GROQ_API_KEY set as Windows environment variable (permanent)

**API setup journey:**
- Tried Gemini free tier → quota exhausted
- Switched to Groq → working, generous free tier, OpenAI-compatible

**Key insight:** The system prompt is the highest-leverage variable in your RAG system. Better prompt = 40% quality improvement. Better embedding model = 15%.

---

### Day 3 — Complete RAG Pipeline
**Concepts learned:**
- Three RAG failure modes:
  1. Retrieval failure — wrong chunks returned (hardest to debug — silent failure)
  2. Context failure — poor prompt structure confuses the model
  3. Generation failure — wrong temperature or format instructions
- Context assembly — labelling chunks, adding delimiters, staying within context window
- The complete RAG flow: document → chunk → embed → store → retrieve → assemble → generate → answer

**Build — novamind_rag.py (v1):**
- Full pipeline: Qdrant retrieval + Groq generation
- System prompt with escape hatch
- Debug chunk printing before every answer
- Live query loop
- Injection attack test — passed
- Boundary test ("2025 pricing?") — passed

**Build — novamind_rag_v2.py:**
- Added inline citations — model cites [Chunk 1], [Chunk 2] in answer
- Added code-based SOURCES section — always runs, always reliable
- Output format instructions added to system prompt

**Key insight:** Retrieval failure is hardest to debug because it's invisible — no error, just a wrong answer. Always log retrieved chunks in production.

---

### Day 4 — PDF Ingestion
**Concepts learned:**
- PDF internals — coordinate rendering instructions, not text files
- Three PDF types:
  1. Digital native — extractable text, PyMuPDF works perfectly
  2. Scanned — image only, needs OCR, PyMuPDF returns empty
  3. Hybrid — mix of both, needs combined approach
- PDF extraction artifacts:
  1. Line-wrap artifact — "Slack \nintegration" split mid-word
  2. Leading period artifact — ". Following three months..." chunk starts with period
- Five chunking strategies:
  1. Fixed size — general purpose, simple
  2. Recursive character — smart default, tries paragraph → sentence → word
  3. Semantic — splits when meaning changes, best quality, slowest
  4. Document structure — uses headings as boundaries, best for structured docs
  5. Sentence window — indexes sentences, retrieves surrounding window

**Tools used:**
- PyMuPDF (fitz) — fastest, most reliable for digital native PDFs
- Page number fingerprinting — first 80 chars as fingerprint to map chunks to pages

**Build — pdf_rag.py:**
- Accepts any PDF via command line argument
- Page-by-page extraction + text cleaning
- Page numbers stored in Qdrant payload
- Citations show page numbers — auditable answers
- Error handling: file not found, empty PDF, no chunks, API failure
- requirements.txt + README.md — portfolio ready

**Key insight:** A PDF is not a document. It's rendering instructions. The quality of your text extraction determines the quality of everything downstream.

---

### Day 5 — Friday Review
- Full Week 2 consolidated
- Interview Q&A completed — all 5 questions answered
- Gap analysis: chunking strategy naming needs reinforcement, evaluation metrics missing (Week 3)
- All projects pushed to GitHub

**Week 2 portfolio output:**
- `novamind_qdrant_search.py` — NovaMind search on Qdrant
- `novamind_qdrant_docker.py` — Qdrant server mode with persistence
- `groq_first_call.py` — first LLM API call
- `novamind_rag.py` — complete RAG pipeline v1
- `novamind_rag_v2.py` — RAG with citations
- `pdf_rag.py` — PDF RAG with page citations + error handling
- `requirements.txt` + `README.md` — portfolio ready

---

## Cumulative Concepts Mastered

| Concept | Week | Status |
|---|---|---|
| Embeddings + vector space | 1 | ✅ Solid |
| Cosine similarity | 1 | ✅ Solid |
| Chunking strategy | 1 | ✅ Solid |
| ChromaDB | 1 | ✅ Solid |
| Metadata filtering | 1 | ✅ Solid |
| Git + GitHub | 1 | ✅ Solid |
| HNSW indexing | 2 | ✅ Solid |
| Qdrant in-memory + server | 2 | ✅ Solid |
| Docker basics | 2 | ✅ Solid |
| Vector DB selection framework | 2 | ✅ Solid |
| Tokens + context window | 2 | ✅ Solid |
| Hallucination mechanics | 2 | ✅ Solid |
| Temperature | 2 | ✅ Solid |
| Groq API | 2 | ✅ Solid |
| Prompt engineering (5 rules) | 2 | ✅ Solid |
| RAG pipeline architecture | 2 | ✅ Solid |
| Three RAG failure modes | 2 | ✅ Solid |
| Context assembly | 2 | ✅ Solid |
| Citations (prompt + code) | 2 | ✅ Solid |
| PDF internals | 2 | ✅ Solid |
| PyMuPDF extraction | 2 | ✅ Solid |
| Five chunking strategies | 2 | ⚠️ Needs reinforcement |
| RAG evaluation metrics | — | ❌ Week 3 |
| Multi-document RAG | — | ❌ Week 3 |
| Hybrid search | — | ❌ Week 3 |
| Reranking | — | ❌ Week 3 |
| FastAPI | — | ❌ Week 3-4 |
| Frontend | — | ❌ Week 4 |
| Deployment | — | ❌ Week 4 |

---

## What Week 3 Adds

- Hybrid search — vector + keyword combined
- Multi-document RAG — thousands of documents, source tracking
- RAG evaluation — measure accuracy, precision, recall, faithfulness
- Reranking — second-pass quality improvement
- Begin API layer — FastAPI wrapper around RAG pipeline

---

*Last updated: End of Week 2, Day 5*
*Next session: Week 3, Day 1 — Hybrid Search*

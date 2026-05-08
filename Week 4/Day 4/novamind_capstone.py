"""
Team Knowledge Base — Capstone Project (Week 4, Day 4)
=======================================================
This is the anchor portfolio piece. It demonstrates every skill from all 4 weeks:

Week 1: Embeddings, chunking, vector search
Week 2: Qdrant, LLM integration, RAG pipeline, PDF ingestion
Week 3: Hybrid search (BM25 + dense + RRF), re-ranking, prompt engineering
Week 4: FastAPI, Pydantic, async, authentication, deployment-ready structure

Architecture:
    POST /ingest  → upload document → chunk → embed → store in Qdrant
    POST /query   → embed query → hybrid search → rerank → Groq → answer
    GET  /health  → server status
    GET  /documents → list all ingested documents

Two endpoints. Clean separation of concerns.
Ingest once. Query many times.
"""

import os
import io
import uuid
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF support — PyMuPDF
try:
    import fitz  # PyMuPDF
    PDF_SUPPORTED = True
except ImportError:
    PDF_SUPPORTED = False
    print("PyMuPDF not installed — PDF ingestion disabled. Install with: pip install pymupdf")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIMS = 384
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K_RETRIEVE = 10     # how many chunks to retrieve before reranking
TOP_K_RERANK = 3        # how many chunks to keep after reranking
BM25_WEIGHT = 0.3       # weight for sparse BM25 scores in RRF
DENSE_WEIGHT = 0.7      # weight for dense vector scores in RRF
RRF_K = 60              # RRF constant — standard value

API_KEY = "knowledge-base-2024"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a helpful knowledge base assistant.
Your job is to answer questions using ONLY the context provided below.
The context contains excerpts from documents in the knowledge base.

Rules you must follow:
1. Answer ONLY from information explicitly stated in the context.
2. Do not infer, extrapolate, or fill gaps with general knowledge.
3. If the answer is not in the context, say exactly:
   "I don't have enough information to answer this question."
4. Cite which chunk your answer came from using [Chunk N] notation.
5. Keep answers concise — 2-4 sentences unless the question requires more.
6. If multiple documents are relevant, synthesise information across them.
"""


# ==============================================================================
# GLOBAL STATE
# Loaded once at startup, reused for every request.
# ==============================================================================

app_state = {
    "qdrant": None,
    "embedder": None,
    "reranker": None,
    "groq": None,
    "documents": {},    # tracks ingested documents: {doc_id: {name, chunk_count, ingested_at}}
    "all_chunks": [],   # all chunk texts for BM25 index
    "bm25": None,       # BM25 index — rebuilt after every ingestion
    "chunk_ids": [],    # maps BM25 index position to Qdrant point id
    "ready": False
}


# ==============================================================================
# LIFESPAN — STARTUP
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and connect to Qdrant at startup."""
    print("Starting Team Knowledge Base API...")

    print("Loading embedding model...")
    app_state["embedder"] = SentenceTransformer(EMBEDDING_MODEL)

    # Cross-encoder for re-ranking — same model used in Week 3
    print("Loading re-ranking model...")
    app_state["reranker"] = CrossEncoder(RERANK_MODEL)

    print("Setting up Qdrant in-memory...")
    app_state["qdrant"] = QdrantClient(":memory:")
    app_state["qdrant"].create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIMS, distance=Distance.COSINE)
    )

    app_state["groq"] = Groq()
    app_state["ready"] = True

    print("Team Knowledge Base API ready.")
    print("Docs: http://localhost:8000/docs")

    yield

    print("Shutting down...")
    app_state["ready"] = False


# ==============================================================================
# FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="Team Knowledge Base API",
    description="""
    A production-grade RAG system for querying your team's documents.

    **Features:**
    - Multi-document ingestion (PDF and plain text)
    - Hybrid search: BM25 sparse + dense vectors + RRF fusion
    - Cross-encoder re-ranking for precision
    - Grounded answers with source citations

    **Authentication:** Include header `x-api-key: knowledge-base-2024`
    """,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class QueryRequest(BaseModel):
    """Shape of a query request."""
    question: str
    top_k: Optional[int] = TOP_K_RERANK
    document_filter: Optional[str] = None  # filter by specific document name


class SourceChunk(BaseModel):
    """A single retrieved and re-ranked chunk."""
    chunk_index: int
    document_name: str
    preview: str
    rerank_score: float


class QueryResponse(BaseModel):
    """Shape of a query response."""
    answer: str
    sources: List[SourceChunk]
    question: str
    documents_searched: int


class IngestResponse(BaseModel):
    """Shape of an ingest response."""
    document_id: str
    document_name: str
    chunks_created: int
    total_chunks_in_kb: int
    message: str


class DocumentInfo(BaseModel):
    """Info about a single ingested document."""
    document_id: str
    document_name: str
    chunk_count: int
    ingested_at: str


class HealthResponse(BaseModel):
    """Shape of health check response."""
    status: str
    documents_ingested: int
    total_chunks: int
    models_loaded: bool


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using PyMuPDF.
    Returns concatenated text from all pages.
    """
    if not PDF_SUPPORTED:
        raise HTTPException(status_code=400, detail="PDF support not available. Install pymupdf.")

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    page_count = doc.page_count  # save before closing
    for page in doc:
        pages.append(page.get_text())
    doc.close()

    if not any(pages):
        raise HTTPException(status_code=400, detail="PDF appears to be empty or scanned (no extractable text).")

    return "\n\n".join(pages)


def chunk_text(text: str) -> List[str]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter.
    Filters out chunks that are too short to be meaningful.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [c for c in chunks if len(c) > 100]


def rebuild_bm25_index():
    """
    Rebuild the BM25 index from all chunks currently in the knowledge base.

    BM25 requires tokenised text — we split on whitespace for simplicity.
    This runs after every ingestion to include new chunks.

    Why rebuild completely: BM25Okapi computes IDF (inverse document frequency)
    across the entire corpus. Adding new documents changes IDF scores for all terms.
    Rebuilding ensures scores are accurate across all documents.
    """
    if not app_state["all_chunks"]:
        app_state["bm25"] = None
        return

    tokenised = [chunk.lower().split() for chunk in app_state["all_chunks"]]
    app_state["bm25"] = BM25Okapi(tokenised)
    print(f"BM25 index rebuilt with {len(app_state['all_chunks'])} chunks")


def hybrid_search(question: str, top_k: int, doc_filter: Optional[str] = None) -> List[dict]:
    """
    Two-stage retrieval using hybrid search + re-ranking.

    Stage 1: Retrieve TOP_K_RETRIEVE candidates using RRF fusion of:
        - Dense vector search (semantic similarity)
        - BM25 sparse search (keyword matching)

    Stage 2: Re-rank candidates with cross-encoder, return top_k.

    Args:
        question: the user's question
        top_k: number of chunks to return after re-ranking
        doc_filter: optional document name to filter results

    Returns:
        list of dicts with chunk text, metadata and rerank score
    """
    # --- STAGE 1A: Dense vector search ---
    query_vector = app_state["embedder"].encode(question).tolist()

    # Build metadata filter if document_filter specified
    qdrant_filter = None
    if doc_filter:
        qdrant_filter = Filter(
            must=[FieldCondition(key="document_name", match=MatchValue(value=doc_filter))]
        )

    dense_results = app_state["qdrant"].query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_RETRIEVE,
        query_filter=qdrant_filter
    ).points

    # Map point_id → dense rank for RRF
    dense_ranks = {result.id: i for i, result in enumerate(dense_results)}

    # --- STAGE 1B: BM25 sparse search ---
    bm25_ranks = {}
    if app_state["bm25"] is not None:
        query_tokens = question.lower().split()
        bm25_scores = app_state["bm25"].get_scores(query_tokens)

        # Get top TOP_K_RETRIEVE indices by BM25 score
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:TOP_K_RETRIEVE]

        # Map chunk_id → BM25 rank
        for rank, idx in enumerate(top_bm25_indices):
            if idx < len(app_state["chunk_ids"]):
                chunk_id = app_state["chunk_ids"][idx]
                bm25_ranks[chunk_id] = rank

    # --- STAGE 1C: RRF fusion ---
    # Collect all candidate IDs from both lists
    all_candidate_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())

    # RRF score: 1/(rank + K) for each list, summed
    rrf_scores = {}
    for point_id in all_candidate_ids:
        score = 0.0
        if point_id in dense_ranks:
            score += DENSE_WEIGHT * (1.0 / (dense_ranks[point_id] + RRF_K))
        if point_id in bm25_ranks:
            score += BM25_WEIGHT * (1.0 / (bm25_ranks[point_id] + RRF_K))
        rrf_scores[point_id] = score

    # Sort by RRF score, take top TOP_K_RETRIEVE
    top_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:TOP_K_RETRIEVE]

    # Fetch full payloads for top candidates
    # Build a lookup from dense results
    dense_lookup = {result.id: result for result in dense_results}

    candidates = []
    for point_id in top_ids:
        if point_id in dense_lookup:
            result = dense_lookup[point_id]
            candidates.append({
                "id": point_id,
                "text": result.payload["text"],
                "document_name": result.payload["document_name"],
                "chunk_index": result.payload["chunk_index"],
                "preview": result.payload["preview"],
                "rrf_score": rrf_scores[point_id]
            })

    if not candidates:
        return []

    # --- STAGE 2: Cross-encoder re-ranking ---
    # Cross-encoder evaluates query + document together — much more accurate than
    # bi-encoder similarity. Slower but only runs on TOP_K_RETRIEVE candidates.
    pairs = [[question, c["text"]] for c in candidates]
    rerank_scores = app_state["reranker"].predict(pairs)

    # Attach rerank scores and sort
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(rerank_scores[i])

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return candidates[:top_k]


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """GET /health — server status and knowledge base stats."""
    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="Server starting up")

    return HealthResponse(
        status="ok",
        documents_ingested=len(app_state["documents"]),
        total_chunks=len(app_state["all_chunks"]),
        models_loaded=True
    )


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents(x_api_key: str = Header(...)):
    """
    GET /documents — list all ingested documents.
    Shows document name, chunk count and ingestion time.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return [
        DocumentInfo(
            document_id=doc_id,
            document_name=info["name"],
            chunk_count=info["chunk_count"],
            ingested_at=info["ingested_at"]
        )
        for doc_id, info in app_state["documents"].items()
    ]


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    """
    POST /ingest — upload and index a document.

    Accepts: .txt and .pdf files
    Process: read → extract text → chunk → embed → store in Qdrant → rebuild BM25

    The document is given a unique ID so multiple documents can coexist
    in the knowledge base and be filtered independently.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="Server not ready")

    # Validate file type
    filename = file.filename or "unknown"
    if not (filename.endswith(".txt") or filename.endswith(".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .pdf files are supported"
        )

    # Read file bytes
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # Extract text based on file type
    if filename.endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_bytes)
    else:
        try:
            raw_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = file_bytes.decode("latin-1")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from file")

    # Chunk the text
    chunks = chunk_text(raw_text)

    if not chunks:
        raise HTTPException(status_code=400, detail="Document too short to create meaningful chunks")

    # Generate unique document ID
    doc_id = str(uuid.uuid4())[:8]
    ingested_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # Embed all chunks
    embeddings = app_state["embedder"].encode(chunks)

    # Store in Qdrant with document metadata in payload
    # Each chunk carries its document name so we can filter by document later
    start_id = len(app_state["all_chunks"])  # offset to avoid ID collisions

    points = [
        PointStruct(
            id=start_id + i,
            vector=embeddings[i].tolist(),
            payload={
                "text": chunks[i],
                "document_name": filename,
                "document_id": doc_id,
                "chunk_index": i,
                "preview": chunks[i][:80]
            }
        )
        for i in range(len(chunks))
    ]

    app_state["qdrant"].upsert(collection_name=COLLECTION_NAME, points=points)

    # Update BM25 corpus and chunk ID map
    app_state["all_chunks"].extend(chunks)
    app_state["chunk_ids"].extend([start_id + i for i in range(len(chunks))])

    # Rebuild BM25 index to include new chunks
    rebuild_bm25_index()

    # Track document metadata
    app_state["documents"][doc_id] = {
        "name": filename,
        "chunk_count": len(chunks),
        "ingested_at": ingested_at
    }

    print(f"Ingested: {filename} → {len(chunks)} chunks (doc_id: {doc_id})")

    return IngestResponse(
        document_id=doc_id,
        document_name=filename,
        chunks_created=len(chunks),
        total_chunks_in_kb=len(app_state["all_chunks"]),
        message=f"Successfully ingested {filename}. Knowledge base now contains {len(app_state['all_chunks'])} total chunks across {len(app_state['documents'])} documents."
    )


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    x_api_key: str = Header(...)
):
    """
    POST /query — ask a question against the knowledge base.

    Uses hybrid search (BM25 + dense + RRF) followed by cross-encoder re-ranking.
    Returns a grounded answer with source citations.

    Optional: filter by document_name to query a specific document only.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="Server not ready")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(app_state["all_chunks"]) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Use POST /ingest to add documents first."
        )

    # Run hybrid search + reranking
    try:
        candidates = hybrid_search(
            question=request.question,
            top_k=request.top_k or TOP_K_RERANK,
            doc_filter=request.document_filter
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

    if not candidates:
        return QueryResponse(
            answer="I don't have enough information to answer this question.",
            sources=[],
            question=request.question,
            documents_searched=len(app_state["documents"])
        )

    # Assemble context from top candidates
    context_parts = []
    for i, candidate in enumerate(candidates):
        context_parts.append(
            f"[Chunk {i+1}] (from: {candidate['document_name']})\n{candidate['text']}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    # Build prompt
    user_prompt = f"""<context>
{context_block}
</context>

<question>
{request.question}
</question>

Answer the question using ONLY the information in the context above.
If the answer is not in the context, say exactly: "I don't have enough information to answer this question."
"""

    # Call Groq
    try:
        response = app_state["groq"].chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Build source list
    sources = [
        SourceChunk(
            chunk_index=c["chunk_index"],
            document_name=c["document_name"],
            preview=c["preview"],
            rerank_score=round(c["rerank_score"], 4)
        )
        for c in candidates
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        question=request.question,
        documents_searched=len(app_state["documents"])
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run("novamind_capstone:app", host="0.0.0.0", port=8000, reload=False)

"""
NovaMind RAG API — Week 4, Day 1
================================
This file wraps the RAG pipeline we built in Weeks 2-3 inside a FastAPI server.
The goal: give the outside world a real URL to send questions to and get answers back.

What changes from the terminal script:
- Nothing in the RAG logic itself changes
- We add a "door" (FastAPI) around it so HTTP requests can trigger it
- Input is validated automatically before it reaches the pipeline
- Output is returned as structured JSON instead of printed to terminal

Run this file, then open http://localhost:8000/docs in your browser.
FastAPI auto-generates an interactive UI where you can test every endpoint.
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==============================================================================
# CONFIGURATION
# All settings in one place — easy to change without hunting through the code
# ==============================================================================

DOCUMENT_PATH = r"D:\AI learning roadmap state\ai-learning-projects\Week 1\Day 3\novamind_sample.txt"
COLLECTION_NAME = "novamind_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384 dimensions, fast, free, local
EMBEDDING_DIMS = 384
TOP_K = 3                                # number of chunks to retrieve per query
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# API key for protecting our endpoints.
# In production this would come from an environment variable or secrets manager.
# For this demo we hardcode it — change this to anything you want.
API_KEY = "novamind-secret-2024"

# Groq model — same one we've used all program
GROQ_MODEL = "llama-3.3-70b-versatile"

# System prompt — the hardened version from Week 3 Day 3
# Explicitly restricts the LLM to only use retrieved context
SYSTEM_PROMPT = """You are NovaMind's internal knowledge assistant.
Your job is to answer questions using ONLY the context provided below.
The context contains excerpts from NovaMind's internal documents.

Rules you must follow without exception:
1. Answer ONLY from information explicitly stated word-for-word in the context.
2. Do not infer, extrapolate, or fill gaps with general knowledge.
3. If the answer is not explicitly in the context, say exactly:
   "I don't have enough information to answer this question."
4. Always cite which chunk your answer came from using [Chunk N] notation.
5. Keep answers concise — 2-3 sentences maximum unless the question requires more.
"""


# ==============================================================================
# GLOBAL STATE
# These objects are created once when the server starts and reused for every
# request. Creating them per-request would be extremely slow — embedding model
# alone takes several seconds to load.
# ==============================================================================

# We use a dict as a simple container for shared state.
# FastAPI's lifespan pattern (below) populates this on startup.
app_state = {
    "qdrant": None,
    "embedder": None,
    "groq": None,
    "ready": False   # flag — False until startup fully completes
}


# ==============================================================================
# LIFESPAN — STARTUP AND SHUTDOWN LOGIC
# This runs once when the server starts and once when it stops.
# We load all heavy resources here so they're ready before any request arrives.
# asynccontextmanager turns this into a context manager FastAPI understands.
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything before 'yield' runs at startup.
    Everything after 'yield' runs at shutdown.
    This is where we load models, connect to databases, and ingest documents.
    """
    print("Starting NovaMind RAG API...")

    # Step 1: Load the embedding model
    # This downloads ~90MB on first run, then caches locally.
    print("Loading embedding model...")
    app_state["embedder"] = SentenceTransformer(EMBEDDING_MODEL)

    # Step 2: Connect to Qdrant in-memory
    # For this demo we use in-memory Qdrant — vectors reset on restart.
    # In production you'd connect to a persistent Qdrant server.
    print("Connecting to Qdrant...")
    app_state["qdrant"] = QdrantClient(":memory:")

    # Step 3: Create the collection
    # This is the "table" that holds our vectors.
    app_state["qdrant"].create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIMS, distance=Distance.COSINE)
    )

    # Step 4: Load and chunk the NovaMind document
    print(f"Loading document: {DOCUMENT_PATH}")
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # RecursiveCharacterTextSplitter tries paragraph breaks first,
    # then line breaks, then sentences — preserves meaning better than fixed split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(raw_text)

    # Filter out chunks that are too short to be meaningful
    chunks = [c for c in chunks if len(c) > 100]
    print(f"Created {len(chunks)} chunks from document")

    # Step 5: Embed all chunks and store in Qdrant
    # We do this once at startup — not on every query.
    print("Embedding and indexing chunks...")
    embeddings = app_state["embedder"].encode(chunks)

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "text": chunks[i],
                "chunk_index": i,
                # Store first 80 chars as a preview for source display
                "preview": chunks[i][:80]
            }
        )
        for i in range(len(chunks))
    ]

    app_state["qdrant"].upsert(collection_name=COLLECTION_NAME, points=points)

    # Step 6: Connect to Groq
    # Groq reads GROQ_API_KEY from environment variable automatically
    app_state["groq"] = Groq()

    # Mark as ready — health endpoint uses this flag
    app_state["ready"] = True
    print(f"NovaMind RAG API ready. {len(chunks)} chunks indexed.")
    print("Interactive docs: http://localhost:8000/docs")

    yield  # Server is now running and handling requests

    # Shutdown cleanup — runs when server stops (Ctrl+C)
    print("Shutting down NovaMind RAG API...")
    app_state["ready"] = False


# ==============================================================================
# FASTAPI APP INSTANCE
# We pass our lifespan function so FastAPI knows what to run on start/stop.
# ==============================================================================

app = FastAPI(
    title="NovaMind Knowledge API",
    description="RAG-powered API for querying NovaMind's internal documents",
    version="1.0.0",
    lifespan=lifespan
)


# ==============================================================================
# CORS MIDDLEWARE
# CORS = Cross-Origin Resource Sharing.
# Without this, a browser on a different domain cannot call our API.
# allow_origins=["*"] means any frontend can call us — fine for a demo.
# In production you'd restrict this to your actual frontend domain.
# ==============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# PYDANTIC SCHEMAS
# These define the shape of data coming IN and going OUT.
# FastAPI automatically validates against these — malformed requests are
# rejected before they ever reach our RAG logic.
# ==============================================================================

class QueryRequest(BaseModel):
    """
    Shape of a valid query request.
    question: the user's question — must be a non-empty string
    top_k: how many chunks to retrieve — optional, defaults to 3
    """
    question: str
    top_k: Optional[int] = TOP_K


class SourceChunk(BaseModel):
    """
    A single retrieved chunk returned as a source citation.
    Lets the caller see exactly what context the answer was based on.
    """
    chunk_index: int
    preview: str        # first 80 chars — enough to identify the source
    score: float        # similarity score — how relevant this chunk was


class QueryResponse(BaseModel):
    """
    Shape of a successful query response.
    answer: the LLM's answer grounded in retrieved context
    sources: the chunks that were used to generate the answer
    question: echo back the original question for clarity
    """
    answer: str
    sources: List[SourceChunk]
    question: str


class HealthResponse(BaseModel):
    """Shape of the health check response."""
    status: str
    chunks_indexed: int
    model: str


# ==============================================================================
# HELPER FUNCTION — THE RAG PIPELINE
# This is the core logic. It's a regular function called by the endpoint.
# Separating it from the endpoint makes it easier to test and reuse.
# ==============================================================================

def run_rag_pipeline(question: str, top_k: int) -> tuple[str, list]:
    """
    Execute the full RAG pipeline for a given question.

    Returns:
        answer (str): the LLM's grounded answer
        results (list): the retrieved Qdrant results for source display

    Flow: embed question → retrieve chunks → assemble context → call Groq → return answer
    """
    # Step 1: Embed the question using the same model used at indexing time.
    # Critical — you must use the same model for query and documents.
    # Different models produce incompatible vector spaces.
    query_vector = app_state["embedder"].encode(question).tolist()

    # Step 2: Search Qdrant for the most similar chunks
    results = app_state["qdrant"].query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    # Step 3: Assemble retrieved chunks into a labelled context block
    # Labels [Chunk 1], [Chunk 2] let the LLM cite sources in its answer
    # --- separators make chunk boundaries visually clear in the prompt
    context_parts = []
    for i, result in enumerate(results):
        context_parts.append(f"[Chunk {i+1}]\n{result.payload['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    # Step 4: Build the full prompt
    # XML-style delimiters separate context from question clearly.
    # This reduces the chance of the LLM treating context as instructions.
    user_prompt = f"""<context>
{context_block}
</context>

<question>
{question}
</question>

Answer the question using ONLY the information in the context above.
If the answer is not in the context, say exactly: "I don't have enough information to answer this question."
"""

    # Step 5: Call Groq with temperature=0
    # Temperature 0 = deterministic, factual. Never use higher for RAG.
    response = app_state["groq"].chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content
    return answer, results


# ==============================================================================
# ENDPOINTS
# These are the actual URLs the outside world can call.
# ==============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    GET /health — is the server alive and ready?

    Used by:
    - Load balancers to decide whether to send traffic here
    - Deployment platforms to know when startup is complete
    - You, to verify the server started correctly

    Returns 503 if startup hasn't finished yet.
    """
    if not app_state["ready"]:
        raise HTTPException(
            status_code=503,
            detail="Server is still starting up — try again in a moment"
        )

    # Count how many chunks are indexed
    collection_info = app_state["qdrant"].get_collection(COLLECTION_NAME)
    chunks_count = collection_info.points_count

    return HealthResponse(
        status="ok",
        chunks_indexed=chunks_count,
        model=GROQ_MODEL
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    x_api_key: str = Header(...)   # ... means this header is REQUIRED, no default
):
    """
    POST /query — ask a question, get an answer grounded in NovaMind documents.

    Requires header: x-api-key: novamind-secret-2024

    Request body:
        question (str): your question
        top_k (int, optional): how many chunks to retrieve, default 3

    Returns:
        answer: the LLM's grounded answer
        sources: the chunks used to generate the answer
        question: your original question echoed back
    """
    # Step 1: Authenticate — check API key before doing any expensive work
    # This prevents random people from using your Groq quota
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Include header: x-api-key: novamind-secret-2024"
        )

    # Step 2: Validate the server is ready
    if not app_state["ready"]:
        raise HTTPException(
            status_code=503,
            detail="Server not ready yet"
        )

    # Step 3: Validate the question is not empty
    # Pydantic already checked it's a string — we check it's not blank
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # Step 4: Run the RAG pipeline
    # Any exception inside is caught and returned as a 500 error
    try:
        answer, results = run_rag_pipeline(request.question, request.top_k)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(e)}"
        )

    # Step 5: Build the structured response
    # Convert Qdrant results into our SourceChunk schema
    sources = [
        SourceChunk(
            chunk_index=result.payload["chunk_index"],
            preview=result.payload["preview"],
            score=round(result.score, 4)
        )
        for result in results
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        question=request.question
    )


# ==============================================================================
# ENTRY POINT
# When you run this file directly, uvicorn starts the server.
# host="0.0.0.0" = accept connections from any network interface (required for deployment)
# port=8000 = standard port for local development
# reload=True = auto-restart when you save changes (development only, remove in production)
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run("novamind_fastapi:app", host="0.0.0.0", port=8000, reload=True)

"""
NovaMind RAG API — Docker Version (Week 4, Day 2)
==================================================
This is the same API as Day 1 with one critical change:
Qdrant connection switches from in-memory to a persistent container.

Day 1: QdrantClient(":memory:")  — vectors lost on every restart
Day 2: QdrantClient("qdrant", port=6333) — vectors persist in a Docker volume

"qdrant" is not a URL or IP address — it is the service name defined in
docker-compose.yml. Inside the Docker network, containers find each other
by service name. This is how the app container talks to the Qdrant container.

Everything else in this file is identical to Day 1. The RAG logic does not
change — only the infrastructure connection does.
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
# ==============================================================================

# Document path inside the container.
# When Docker builds the image, it copies everything from the build directory
# into /app (our WORKDIR). So novamind_sample.txt must be in the same folder
# as this script when you run docker-compose up.
DOCUMENT_PATH = "novamind_sample.txt"

COLLECTION_NAME = "novamind_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384
TOP_K = 3
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# API key — in production this would come from a secrets manager.
# For this demo it is hardcoded. Never commit real secrets to GitHub.
API_KEY = "novamind-secret-2024"

# Groq model — read from environment variable injected by docker-compose.
# os.environ.get() reads from the container's environment.
# If not set, falls back to the default model name.
GROQ_MODEL = "llama-3.3-70b-versatile"

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
# GLOBAL STATE — loaded once at startup, reused for every request
# ==============================================================================

app_state = {
    "qdrant": None,
    "embedder": None,
    "groq": None,
    "ready": False
}


# ==============================================================================
# LIFESPAN — STARTUP AND SHUTDOWN
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load models, connect to Qdrant container, ingest documents.
    Shutdown: cleanup.

    Key difference from Day 1: Qdrant connection uses service name "qdrant"
    instead of ":memory:". This connects to the Qdrant container running
    on the Docker internal network created by docker-compose.
    """
    print("Starting NovaMind RAG API (Docker version)...")

    # Load embedding model — same as Day 1
    print("Loading embedding model...")
    app_state["embedder"] = SentenceTransformer(EMBEDDING_MODEL)

    # Connect to Qdrant CONTAINER — this is the key change from Day 1.
    # "qdrant" resolves to the Qdrant container's IP on the Docker network.
    # Docker handles the DNS resolution automatically — we just use the name.
    print("Connecting to Qdrant container...")
    app_state["qdrant"] = QdrantClient("qdrant", port=6333)

    # Check if collection already exists — important for persistence.
    # When the app container restarts but Qdrant container keeps running,
    # the collection already exists with all vectors. No need to re-ingest.
    existing = [c.name for c in app_state["qdrant"].get_collections().collections]

    if COLLECTION_NAME not in existing:
        # First run — create collection and ingest documents
        print(f"Collection not found. Creating and ingesting documents...")

        app_state["qdrant"].create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMS, distance=Distance.COSINE)
        )

        # Load document — path is relative to /app inside the container
        print(f"Loading document: {DOCUMENT_PATH}")
        with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
            raw_text = f.read()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(raw_text)
        chunks = [c for c in chunks if len(c) > 100]
        print(f"Created {len(chunks)} chunks")

        embeddings = app_state["embedder"].encode(chunks)

        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={
                    "text": chunks[i],
                    "chunk_index": i,
                    "preview": chunks[i][:80]
                }
            )
            for i in range(len(chunks))
        ]

        app_state["qdrant"].upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Indexed {len(chunks)} chunks into Qdrant.")

    else:
        # Collection already exists — vectors are persisted in the Docker volume.
        # This happens when the app restarts but Qdrant keeps running.
        # No re-ingestion needed — just reconnect and use existing vectors.
        count = app_state["qdrant"].get_collection(COLLECTION_NAME).points_count
        print(f"Collection already exists with {count} chunks. Skipping ingestion.")

    # Connect to Groq — reads GROQ_API_KEY from container environment
    app_state["groq"] = Groq()
    app_state["ready"] = True

    print("NovaMind RAG API ready.")
    print("Docs: http://localhost:8000/docs")

    yield

    print("Shutting down...")
    app_state["ready"] = False


# ==============================================================================
# FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="NovaMind Knowledge API",
    description="RAG-powered API for querying NovaMind's internal documents",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# PYDANTIC SCHEMAS — identical to Day 1
# ==============================================================================

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K


class SourceChunk(BaseModel):
    chunk_index: int
    preview: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    question: str


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    model: str
    storage: str   # new field — shows whether using memory or persistent storage


# ==============================================================================
# RAG PIPELINE — identical to Day 1
# ==============================================================================

def run_rag_pipeline(question: str, top_k: int) -> tuple[str, list]:
    """Execute the full RAG pipeline. Logic unchanged from Day 1."""

    query_vector = app_state["embedder"].encode(question).tolist()

    results = app_state["qdrant"].query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    context_parts = []
    for i, result in enumerate(results):
        context_parts.append(f"[Chunk {i+1}]\n{result.payload['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""<context>
{context_block}
</context>

<question>
{question}
</question>

Answer the question using ONLY the information in the context above.
If the answer is not in the context, say exactly: "I don't have enough information to answer this question."
"""

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
# ==============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    GET /health — server status check.
    Now includes storage type so you can confirm persistent Qdrant is connected.
    """
    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="Server starting up")

    collection_info = app_state["qdrant"].get_collection(COLLECTION_NAME)
    chunks_count = collection_info.points_count

    return HealthResponse(
        status="ok",
        chunks_indexed=chunks_count,
        model=GROQ_MODEL,
        storage="persistent (qdrant container)"  # confirms we're not using memory
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    x_api_key: str = Header(...)
):
    """
    POST /query — ask a question, get a grounded answer with sources.
    Requires header: x-api-key: novamind-secret-2024
    """
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Include header: x-api-key: novamind-secret-2024"
        )

    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="Server not ready")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer, results = run_rag_pipeline(request.question, request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

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
# Note: reload=False here — inside Docker, file watching doesn't work the same way.
# Rebuild the image if you change code.
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run("novamind_fastapi_docker:app", host="0.0.0.0", port=8000, reload=False)

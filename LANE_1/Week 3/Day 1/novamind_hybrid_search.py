"""
NovaMind Hybrid Search RAG — Week 3, Day 1
Demonstrates dense-only vs hybrid (dense + sparse BM25) retrieval side by side.

Uses rank-bm25 for sparse scoring (pure Python, works on all Python versions).
Hybrid fusion is done manually with Reciprocal Rank Fusion (RRF).

Requirements:
    pip install qdrant-client rank-bm25 sentence-transformers groq langchain-text-splitters

Run Qdrant first:
    docker run -p 6333:6333 qdrant/qdrant
"""

import os
import sys
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ─────────────────────────────────────────────────────────────

COLLECTION_NAME = "novamind_hybrid"
DENSE_SIZE = 384
GROQ_MODEL = "llama-3.3-70b-versatile"
RRF_K = 60  # RRF constant — standard value, smooths rank scores

# ── NovaMind document sources ──────────────────────────────────────────────────
# Three sources deliberately chosen:
#   pricing      → pure prose, semantic search handles this well
#   integrations → contains exact product codes: MX-4400, LLM-7B, GHB-3310
#   hr_policy    → prose with specific policy codes: DSG-2024
# This mix clearly shows when hybrid beats dense-only

DOCUMENTS = {
    "pricing": """
NovaMind pricing is structured across three tiers. The Starter plan costs $29 per month
and supports up to 5 team members with 10GB of storage. The Professional plan costs $99
per month and supports up to 25 team members with 100GB of storage and priority support.
The Enterprise plan is custom-priced and includes unlimited team members, dedicated
infrastructure, SSO integration, and a dedicated account manager.
All plans include a 14-day free trial. Annual billing provides a 20% discount.
NovaMind does not offer refunds after the first 30 days of a paid plan.
""",

    "integrations": """
NovaMind supports the following integrations as of Q2 2024.
Slack integration: module code SLK-2200. Enables real-time notifications and
slash commands directly in Slack channels. Requires Slack workspace admin permissions.
GitHub integration: module code GHB-3310. Syncs pull request status and CI pipeline
results into NovaMind dashboards. Supports GitHub Enterprise via the GHB-3310-ENT variant.
Jira integration: module code JRA-4400. Bidirectional sync of tickets and sprint boards.
The MX-4400 data connector handles custom data source ingestion for enterprise clients.
MX-4400 supports REST, GraphQL, and webhook-based data pipelines.
The LLM-7B fine-tuning module allows enterprise customers to fine-tune NovaMind's
internal language model on their proprietary data. Minimum dataset size: 10,000 samples.
""",

    "hr_policy": """
NovaMind employee leave policy is as follows.
Annual leave: 25 days per year for all full-time employees. Part-time employees
receive leave on a pro-rata basis. Leave must be approved by line manager at least
two weeks in advance for periods longer than 3 consecutive days.
Sick leave: up to 10 days per year with no approval required. A doctor's certificate
is required for sick leave exceeding 3 consecutive days.
Parental leave: 16 weeks fully paid for primary caregiver. 4 weeks fully paid for
secondary caregiver. Applies to birth, adoption, and surrogacy arrangements.
Remote work policy: employees may work remotely up to 3 days per week. Exceptions
require VP-level approval. All remote work must comply with the NovaMind data security
guidelines document DSG-2024.
"""
}

# ── Step 1: Dense model ────────────────────────────────────────────────────────

print("Loading dense model (sentence-transformers)...")
dense_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Dense model loaded.")

# ── Step 2: Qdrant connection ──────────────────────────────────────────────────

print("Connecting to Qdrant...")
try:
    client = QdrantClient("localhost", port=6333)
    client.get_collections()
    print("Connected to Qdrant Docker server.")
except Exception as e:
    print(f"ERROR: Cannot connect to Qdrant: {e}")
    print("Run this in a separate terminal first:")
    print("  docker run -p 6333:6333 qdrant/qdrant")
    sys.exit(1)

# ── Step 3: Collection setup ───────────────────────────────────────────────────

existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")

# Dense-only collection — BM25 runs in Python memory, not inside Qdrant
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=DENSE_SIZE, distance=Distance.COSINE)
)
print(f"Created collection: {COLLECTION_NAME}")

# ── Step 4: Chunking ───────────────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
)

all_chunks = []  # list of (text, source)
for source, text in DOCUMENTS.items():
    chunks = splitter.split_text(text.strip())
    for chunk in chunks:
        if len(chunk) > 80:
            all_chunks.append((chunk, source))

print(f"Total chunks after filtering: {len(all_chunks)}")

# ── Step 5: Build BM25 index in Python memory ─────────────────────────────────

# BM25 works on tokenised text — split each chunk into lowercase words
texts = [chunk for chunk, _ in all_chunks]
tokenised = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenised)
print("BM25 index built.")

# ── Step 6: Dense embedding + Qdrant ingestion ────────────────────────────────

print("Generating dense vectors and inserting into Qdrant...")
dense_vectors = dense_model.encode(texts, show_progress_bar=False)

points = []
for i, ((text, source), dense_vec) in enumerate(zip(all_chunks, dense_vectors)):
    points.append(
        PointStruct(
            id=i,
            vector=dense_vec.tolist(),
            payload={
                "text": text,
                "source": source,
                "chunk_index": i,
            }
        )
    )

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Inserted {len(points)} points into Qdrant.\n")

# ── Search helpers ─────────────────────────────────────────────────────────────

def dense_search(query: str, top_k: int = 10):
    """
    Pure dense vector search — semantic only.
    Returns list of (chunk_index, score, payload) tuples.
    """
    query_vector = dense_model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
    return [(p.id, p.score, p.payload) for p in results.points]


def bm25_search(query: str, top_k: int = 10):
    """
    BM25 sparse keyword search — exact term matching.
    Returns list of (chunk_index, score, payload) tuples, sorted by score descending.
    """
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Pair each chunk index with its BM25 score, sort descending
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top = ranked[:top_k]

    results = []
    for chunk_idx, score in top:
        text, source = all_chunks[chunk_idx]
        results.append((chunk_idx, score, {"text": text, "source": source}))
    return results


def reciprocal_rank_fusion(dense_results, sparse_results, top_k: int = 3):
    """
    Combine dense and sparse ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score = 1/(rank_in_dense + K) + 1/(rank_in_sparse + K)

    Documents appearing highly in BOTH lists get the highest combined score.
    Documents in only one list still score — just lower.
    K=60 is the standard constant that prevents top-ranked docs from dominating.
    """
    rrf_scores = {}  # chunk_index → combined RRF score
    chunk_data = {}  # chunk_index → payload for display

    for rank, (chunk_idx, score, payload) in enumerate(dense_results, start=1):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1 / (rank + RRF_K)
        chunk_data[chunk_idx] = payload

    for rank, (chunk_idx, score, payload) in enumerate(sparse_results, start=1):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1 / (rank + RRF_K)
        chunk_data[chunk_idx] = payload

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top = ranked[:top_k]

    return [(chunk_idx, rrf_score, chunk_data[chunk_idx]) for chunk_idx, rrf_score in top]


def hybrid_search(query: str, top_k: int = 3):
    """Full hybrid pipeline: dense(10) + BM25(10) → RRF → top_k."""
    d_results = dense_search(query, top_k=10)
    s_results = bm25_search(query, top_k=10)
    return reciprocal_rank_fusion(d_results, s_results, top_k=top_k)


def build_context(results) -> str:
    """Assemble retrieved chunks into a labelled context block for the LLM."""
    parts = []
    for i, (chunk_idx, score, payload) in enumerate(results, 1):
        source = payload.get("source", "unknown")
        text = payload.get("text", "")
        parts.append(f"[Chunk {i} | Source: {source}]\n{text}")
    return "\n\n---\n\n".join(parts)


def ask_groq(query: str, context: str) -> str:
    """Send query + retrieved context to Groq and return the answer."""
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    system_prompt = """You are NovaMind's internal knowledge assistant.
Answer ONLY using the information provided in the <context> tags below.
If the context does not contain enough information, say exactly:
"I don't have enough information to answer this."
Cite which chunk your answer comes from (e.g. [Chunk 2]).
Keep answers to 3 sentences maximum."""

    user_message = f"""<context>
{context}
</context>

<question>
{query}
</question>"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq API error: {e}"


# ── Comparison runner ──────────────────────────────────────────────────────────

def compare(query: str):
    """
    Run the same query through dense-only and hybrid.
    Print top 3 results from each side by side.
    Then show the LLM answer generated from hybrid context.
    """
    print("\n" + "═" * 70)
    print(f"QUERY: {query}")
    print("═" * 70)

    # Dense only — top 3
    dense_results = dense_search(query, top_k=3)
    print("\n── DENSE ONLY (semantic, top 3) ────────────────────────────")
    for i, (chunk_idx, score, payload) in enumerate(dense_results, 1):
        source = payload.get("source", "unknown")
        text = payload.get("text", "")[:120].replace("\n", " ")
        print(f"  [{i}] source={source} score={score:.3f} | {text}...")

    # Hybrid — top 3
    hybrid_results = hybrid_search(query, top_k=3)
    print("\n── HYBRID (dense + BM25 → RRF, top 3) ─────────────────────")
    for i, (chunk_idx, rrf_score, payload) in enumerate(hybrid_results, 1):
        source = payload.get("source", "unknown")
        text = payload.get("text", "")[:120].replace("\n", " ")
        print(f"  [{i}] source={source} rrf={rrf_score:.4f} | {text}...")

    # LLM answer from hybrid context
    context = build_context(hybrid_results)
    answer = ask_groq(query, context)
    print(f"\n── ANSWER (from hybrid context) ─────────────────────────────")
    print(f"  {answer}")


# ── Demo queries ───────────────────────────────────────────────────────────────

print("\n" + "█" * 70)
print("  NOVAMIND HYBRID SEARCH — DENSE vs HYBRID COMPARISON")
print("█" * 70)

compare("What is the parental leave policy?")
compare("What does the MX-4400 connector support?")
compare("How does the LLM-7B fine-tuning module work?")

# ── Interactive mode ───────────────────────────────────────────────────────────

print("\n" + "═" * 70)
print("INTERACTIVE MODE — type your own queries (or 'quit' to exit)")
print("═" * 70)

while True:
    try:
        query = input("\nYour query: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if not query:
        continue
    if query.lower() in ("quit", "exit", "q"):
        print("Done.")
        break

    compare(query)

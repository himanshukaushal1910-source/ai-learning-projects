# novamind_qdrant_search.py
# Week 2, Day 1 — Slot 3: Build
# NovaMind semantic search rebuilt on Qdrant (was Chroma in Week 1)

# ── Imports ───────────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. Load the source document ───────────────────────────────────────────────
# Same novamind_sample.txt from Week 1 — nothing changes about the data.
# Update this path to wherever your file lives.
FILE_PATH = r"D:\AI learning roadmap state\ai-learning-projects\Week 2\Day 1\novamind_sample.txt"
print("Loading document...")
with open(FILE_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Document loaded — {len(raw_text)} characters.\n")

# ── 2. Chunk the document ─────────────────────────────────────────────────────
# Exact same settings as Week 1:
#   chunk_size=400     → max characters per chunk
#   chunk_overlap=80   → overlap between consecutive chunks to preserve context
#   separators         → tries paragraph breaks first, then sentences, then words
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)

raw_chunks = splitter.split_text(raw_text)

# Minimum length filter — drops header-only chunks (the Week 1 fix)
# Any chunk under 100 characters is almost certainly a title or stray line.
chunks = [c for c in raw_chunks if len(c) >= 100]

print(f"Chunking complete — {len(raw_chunks)} raw chunks → {len(chunks)} after filter.\n")

# ── 3. Load embedding model ───────────────────────────────────────────────────
# all-MiniLM-L6-v2 produces 384-dimensional vectors.
# Must match vector_size declared in the Qdrant collection below.
# Same model as Week 1 — embeddings are directly comparable.
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── 4. Embed all chunks ───────────────────────────────────────────────────────
# model.encode() returns shape (num_chunks, 384).
# .tolist() converts numpy array → plain Python list (Qdrant expects this).
print("Embedding chunks...")
embeddings = model.encode(chunks).tolist()
print(f"Embedded {len(embeddings)} chunks.\n")

# ── 5. Create Qdrant in-memory client and collection ─────────────────────────
# ":memory:" = no disk, no Docker — resets when script ends.
# Distance.COSINE declared here at collection level (not per-query like Chroma).
client = QdrantClient(":memory:")

client.create_collection(
    collection_name="novamind",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
print("Qdrant collection 'novamind' created.\n")

# ── 6. Build and insert points ────────────────────────────────────────────────
# PointStruct = one record in Qdrant.
# payload stores the original text + metadata so we can display results.
# Key difference from Chroma: YOU embed, YOU pass the vector.
# Qdrant only stores and searches vectors — it has no concept of "text".
points = [
    PointStruct(
        id=i,
        vector=embeddings[i],
        payload={
            "chunk_text":   chunks[i],
            "chunk_index":  i,
            "chunk_length": len(chunks[i]),
        }
    )
    for i in range(len(chunks))
]

client.upsert(collection_name="novamind", points=points)
print(f"Inserted {len(points)} points into Qdrant.\n")

# ── 7. Helper: embed a query and return top-k results ─────────────────────────
# Centralised so both preset queries and the live loop use identical logic.
def search(query: str, top_k: int = 3):
    """Embed query and retrieve top_k most similar chunks from Qdrant."""
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name="novamind",
        query=query_vector,
        limit=top_k,
        using=None,   # use the default (only) vector for this collection
    ).points
    return results

# ── 8. Helper: print results ──────────────────────────────────────────────────
def print_results(query: str, results):
    print(f"\nQuery: '{query}'")
    print("-" * 65)
    for rank, r in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"  Score        : {r.score:.4f}")
        print(f"  Chunk index  : {r.payload['chunk_index']}")
        print(f"  Chunk length : {r.payload['chunk_length']} chars")
        print(f"  Text         : {r.payload['chunk_text']}")
        print()

# ── 9. Run 4 preset queries ───────────────────────────────────────────────────
# These are the same queries from Week 1 so you can directly compare
# how Qdrant results differ (or match) the Chroma results.
preset_queries = [
    "What was decided about pricing?",
    "What are the engineering priorities for Q3?",
    "How many people is NovaMind hiring?",
    "What is the customer satisfaction trend?",
]

print("=" * 65)
print("PRESET QUERIES")
print("=" * 65)

for q in preset_queries:
    results = search(q)
    print_results(q, results)

# ── 10. Live query loop ───────────────────────────────────────────────────────
# User can keep entering questions until they type 'exit'.
# Each query is embedded fresh — no caching between turns.
print("=" * 65)
print("LIVE QUERY MODE  (type 'exit' to quit)")
print("=" * 65)

while True:
    user_query = input("\nEnter your query: ").strip()
    if user_query.lower() == "exit":
        print("Exiting. Session ended.")
        break
    if not user_query:
        print("Empty query — please type something.")
        continue
    results = search(user_query)
    print_results(user_query, results)

# novamind_qdrant_docker.py
# Week 2, Day 1 — Slot 5: Deploy/Extra
# Same as novamind_qdrant_search.py but connects to Qdrant running via Docker
#
# BEFORE RUNNING — start Qdrant server in a separate terminal:
#   docker run -p 6333:6333 qdrant/qdrant
#
# Then run this script normally. Data will persist between runs.

# ── Imports ───────────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. Load the source document ───────────────────────────────────────────────
FILE_PATH = r"D:\AI learning roadmap state\ai-learning-projects\Week 2\Day 1\novamind_sample.txt"

print("Loading document...")
with open(FILE_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()
print(f"Document loaded — {len(raw_text)} characters.\n")

# ── 2. Chunk the document ─────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)
raw_chunks = splitter.split_text(raw_text)
chunks = [c for c in raw_chunks if len(c) >= 100]
print(f"Chunking complete — {len(raw_chunks)} raw → {len(chunks)} after filter.\n")

# ── 3. Load embedding model ───────────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── 4. Embed all chunks ───────────────────────────────────────────────────────
print("Embedding chunks...")
embeddings = model.encode(chunks).tolist()
print(f"Embedded {len(embeddings)} chunks.\n")

# ── 5. Connect to Qdrant server running via Docker ────────────────────────────
# THE ONLY CHANGE from the in-memory version:
#   BEFORE:  QdrantClient(":memory:")
#   AFTER:   QdrantClient("localhost", port=6333)
#
# In-memory vs Server — key differences:
#
#   ":memory:"                     │  "localhost", port=6333
#   ───────────────────────────────┼──────────────────────────────────
#   Data lost when script ends     │  Data persists between runs
#   No Docker needed               │  Requires Docker running first
#   Only this script can access it │  Multiple scripts can connect
#   Good for prototyping           │  Good for real projects
#
# Everything else — collections, upsert, query_points — is IDENTICAL.
# This is by design. Qdrant's API doesn't change based on where it runs.
client = QdrantClient("localhost", port=6333)
print("Connected to Qdrant server on localhost:6333\n")

# ── 6. Create collection if it doesn't already exist ─────────────────────────
# On first run: collection is created and vectors are inserted.
# On second run: collection already exists on the server — we skip creation.
# This is the persistence benefit — your data survives between script runs.
existing_collections = [c.name for c in client.get_collections().collections]

if "novamind" not in existing_collections:
    client.create_collection(
        collection_name="novamind",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print("Collection 'novamind' created on server.\n")

    # ── 7. Insert points (only on first run) ─────────────────────────────────
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
    print(f"Inserted {len(points)} points into Qdrant server.\n")

else:
    # Collection already exists from a previous run — data is still there
    print("Collection 'novamind' already exists on server — skipping insert.\n")
    print("This is persistence in action: data survived between script runs.\n")

# ── 8. Helper: search ────────────────────────────────────────────────────────
def search(query: str, top_k: int = 3):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name="novamind",
        query=query_vector,
        limit=top_k,
        using=None,
    ).points
    return results

# ── 9. Helper: print results ─────────────────────────────────────────────────
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

# ── 10. Run 4 preset queries ──────────────────────────────────────────────────
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

# ── 11. Live query loop ───────────────────────────────────────────────────────
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

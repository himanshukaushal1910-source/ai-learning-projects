# qdrant_inmemory_demo.py
# Week 2, Day 1 — Slot 2: Tool Demo
# Qdrant in-memory mode: create collection, insert vectors, run similarity search

# ── Imports ──────────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ── 1. Sample text chunks ─────────────────────────────────────────────────────
# These are the documents we want to search over.
# In a real project these would come from PDFs, databases, etc.
chunks = [
    "NovaMind raised a Series A round of $12 million in funding.",
    "The engineering team is targeting 200ms latency for all search queries.",
    "Redis caching will be implemented in Q3 to reduce repeated embedding calls.",
    "NovaMind is hiring three senior engineers with a September start date.",
    "The pricing model is $49 per seat per month for enterprise customers.",
    "Slack integration is the top feature request from existing customers.",
    "The machine learning team is evaluating hybrid search for better recall.",
    "Customer satisfaction scores increased by 18% after the last product update.",
    "The data pipeline processes over 2 million documents per day.",
    "NovaMind's churn rate dropped to 3.2% following the onboarding redesign.",
]

# ── 2. Load embedding model ───────────────────────────────────────────────────
# all-MiniLM-L6-v2 produces 384-dimensional vectors.
# This must match the vector_size we declare in the collection.
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── 3. Embed all chunks ───────────────────────────────────────────────────────
# model.encode() returns a numpy array of shape (num_chunks, 384).
# We convert to a plain Python list because Qdrant expects lists.
print("Embedding chunks...")
embeddings = model.encode(chunks).tolist()

# ── 4. Create Qdrant in-memory client ────────────────────────────────────────
# ":memory:" means no disk, no Docker — data lives in RAM for this session.
# Identical API to a real Qdrant server — swap ":memory:" for a URL to go prod.
client = QdrantClient(":memory:")

# ── 5. Create a collection ───────────────────────────────────────────────────
# A collection is like a table in a relational DB — holds one type of vectors.
# vector_size must match the embedding model's output dimension (384 here).
# Distance.COSINE tells Qdrant to use cosine similarity for all searches.
client.create_collection(
    collection_name="novamind",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
print("Collection 'novamind' created.\n")

# ── 6. Insert vectors + payloads ─────────────────────────────────────────────
# PointStruct = one record in Qdrant.
#   id      → unique integer ID for this point
#   vector  → the embedding (list of 384 floats)
#   payload → metadata dict — store anything you want (text, source, date, etc.)
# 
# Key difference from Chroma: Qdrant does NOT store or understand text.
# You embed first, then store the vector + the original text in the payload.
points = [
    PointStruct(
        id=i,
        vector=embeddings[i],
        payload={"text": chunks[i], "chunk_index": i}
    )
    for i in range(len(chunks))
]

client.upsert(collection_name="novamind", points=points)
print(f"Inserted {len(points)} vectors into 'novamind'.\n")

# ── 7. Define a search query ─────────────────────────────────────────────────
# The query must be embedded with the SAME model used for the chunks.
# Mixing models = vectors in different spaces = meaningless similarity scores.
query = "What are the engineering priorities for Q3?"
query_vector = model.encode(query).tolist()

# ── 8. Run similarity search (updated API — query_points replaces search) ────
# In newer versions of qdrant-client, .search() was removed.
# query_points() is the current method. using=None means use the default vector.
# .points extracts the list of results from the response object.
results = client.query_points(
    collection_name="novamind",
    query=query_vector,
    limit=3,
    using=None,
).points

# ── 9. Print results ──────────────────────────────────────────────────────────
# result.score  → cosine similarity (0 to 1, higher = more similar)
# result.payload → the metadata dict we stored with each point
print(f"Query: '{query}'\n")
print("Top 3 results:")
print("-" * 60)
for rank, result in enumerate(results, start=1):
    print(f"Rank {rank}")
    print(f"  Score       : {result.score:.4f}")
    print(f"  Chunk index : {result.payload['chunk_index']}")
    print(f"  Text        : {result.payload['text']}")
    print()

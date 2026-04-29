# chroma_metadata_demo.py
# NovaMind — Chroma with cosine distance + metadata filtering

# Import Chroma client
import chromadb

# Import sentence-transformers to generate embeddings
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# STEP 1 — Load the embedding model
# Same model you used in Days 2 and 3
# --------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# STEP 2 — Create an in-memory Chroma client
# Nothing saved to disk — resets every run
# --------------------------------------------------
client = chromadb.Client()

# --------------------------------------------------
# STEP 3 — Create a collection with COSINE distance
# hnsw:space = "cosine" fixes scores to sit 0 to 1
# Without this, Chroma defaults to L2 (larger numbers)
# --------------------------------------------------
collection = client.create_collection(
    name="novamind_docs",
    metadata={"hnsw:space": "cosine"}
)

# --------------------------------------------------
# STEP 4 — Define 5 NovaMind chunks with metadata
# Each chunk has: source, chunk_index, topic
# --------------------------------------------------
chunks = [
    {
        "id": "chunk_0",
        "text": "The team agreed on per-seat pricing after reviewing competitor models. Flat rate was considered but rejected due to enterprise scalability concerns.",
        "source": "meeting_notes",
        "topic": "pricing",
        "chunk_index": 0
    },
    {
        "id": "chunk_1",
        "text": "Q3 engineering priorities include reducing API latency below 200ms, implementing Redis caching, and completing the Slack integration for NovaMind alerts.",
        "source": "engineering_spec",
        "topic": "engineering_roadmap",
        "chunk_index": 1
    },
    {
        "id": "chunk_2",
        "text": "NovaMind is hiring three roles starting September: a senior backend engineer, a machine learning engineer, and a product designer. 47 applications received so far.",
        "source": "hr_document",
        "topic": "hiring",
        "chunk_index": 2
    },
    {
        "id": "chunk_3",
        "text": "The infrastructure team will migrate from AWS EC2 to containerised workloads using Docker and Kubernetes to improve deployment reliability and reduce costs.",
        "source": "engineering_spec",
        "topic": "infrastructure",
        "chunk_index": 3
    },
    {
        "id": "chunk_4",
        "text": "Q2 revenue exceeded targets by 12%. The board approved an additional budget allocation for engineering headcount in Q3 to accelerate product delivery.",
        "source": "meeting_notes",
        "topic": "financials",
        "chunk_index": 4
    },
]

# --------------------------------------------------
# STEP 5 — Embed all chunks and insert into Chroma
# embeddings — the vectors (list of 384 numbers each)
# documents — the raw text stored alongside the vector
# metadatas — source, topic, chunk_index per chunk
# ids — unique identifier for each chunk
# --------------------------------------------------
collection.add(
    embeddings=model.encode([c["text"] for c in chunks]).tolist(),
    documents=[c["text"] for c in chunks],
    metadatas=[{
        "source": c["source"],
        "topic": c["topic"],
        "chunk_index": c["chunk_index"]
    } for c in chunks],
    ids=[c["id"] for c in chunks]
)

# --------------------------------------------------
# STEP 6 — Run an unfiltered similarity query
# Returns top 3 most similar chunks from ALL sources
# --------------------------------------------------
query = "What are the engineering priorities for Q3?"
query_embedding = model.encode([query]).tolist()

print("\n=== UNFILTERED QUERY ===")
print(f"Query: {query}\n")

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for i in range(len(results["ids"][0])):
    print(f"Rank {i+1}")
    print(f"  Source     : {results['metadatas'][0][i]['source']}")
    print(f"  Topic      : {results['metadatas'][0][i]['topic']}")
    print(f"  Distance   : {results['distances'][0][i]:.4f}")
    print(f"  Text       : {results['documents'][0][i][:80]}...")
    print()

# --------------------------------------------------
# STEP 7 — Run a FILTERED query
# where= restricts search to engineering_spec only
# Similarity search still runs — but only over
# chunks where source = "engineering_spec"
# --------------------------------------------------
print("\n=== FILTERED QUERY (engineering_spec only) ===")
print(f"Query: {query}\n")

filtered_results = collection.query(
    query_embeddings=query_embedding,
    n_results=2,
    where={"source": "engineering_spec"},
    include=["documents", "metadatas", "distances"]
)

for i in range(len(filtered_results["ids"][0])):
    print(f"Rank {i+1}")
    print(f"  Source     : {filtered_results['metadatas'][0][i]['source']}")
    print(f"  Topic      : {filtered_results['metadatas'][0][i]['topic']}")
    print(f"  Distance   : {filtered_results['distances'][0][i]:.4f}")
    print(f"  Text       : {filtered_results['documents'][0][i][:80]}...")
    print()

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# Step 1 — Load the NovaMind document
# Read the raw text file into a Python string
# ─────────────────────────────────────────────

with open("novamind_sample.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

print("")
print("=" * 60)
print("NovaMind Ingestion Pipeline")
print("=" * 60)
print("Document loaded: {} characters".format(len(text)))
print("")

# ─────────────────────────────────────────────
# Step 2 — Chunk the document
# RecursiveCharacterTextSplitter tries to split
# on paragraph breaks first, then sentences,
# then words, then characters as last resort
# chunk_size=400: each chunk up to 400 chars
# chunk_overlap=80: 80 chars shared between
# adjacent chunks to avoid boundary cutoffs
# ─────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(text)
chunks = [c for c in chunks if len(c) > 100]
print("Chunking complete:")
print("  Total chunks    : {}".format(len(chunks)))
print("  Avg chunk length: {:.0f} characters".format(
    sum(len(c) for c in chunks) / len(chunks)))
print("")

# ─────────────────────────────────────────────
# Step 3 — Load embedding model
# Same model used throughout Week 1
# Loads once here — used for both documents
# and queries so embeddings are compatible
# ─────────────────────────────────────────────

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.")
print("")

# ─────────────────────────────────────────────
# Embed all chunks at once
# model.encode() accepts a list of strings
# returns numpy array of shape (num_chunks, 384)
# .tolist() converts to plain Python lists
# Chroma requires plain lists not numpy arrays
# ─────────────────────────────────────────────

print("Embedding {} chunks...".format(len(chunks)))
embeddings = model.encode(chunks).tolist()
print("Embeddings complete. Shape: ({}, 384)".format(len(embeddings)))
print("")

# ─────────────────────────────────────────────
# Step 4 — Store in Chroma
# Create in-memory client and collection
# cosine space gives more intuitive distances
# add all chunks with embeddings and metadata
# ─────────────────────────────────────────────

client = chromadb.Client()

collection = client.create_collection(
    name="novamind",
    metadata={"hnsw:space": "cosine"}  # use cosine distance
)

# Build metadata list — one dict per chunk
# chunk_index: position in document (0-based)
# chunk_length: number of characters in chunk
metadatas = [
    {
        "chunk_index": i,
        "chunk_length": len(chunk)
    }
    for i, chunk in enumerate(chunks)
]

# Build unique IDs for each chunk
ids = ["chunk_{}".format(i) for i in range(len(chunks))]

# Add everything to Chroma in one call
collection.add(
    documents=chunks,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print("Stored {} chunks in Chroma collection 'novamind'".format(
    collection.count()))
print("")

# ─────────────────────────────────────────────
# Step 5 — Run 3 test queries
# For each query:
#   1. embed the query text
#   2. search Chroma for top 2 similar chunks
#   3. print ranked results with metadata
# ─────────────────────────────────────────────

queries = [
    "What was decided about pricing?",
    "What are the engineering priorities for Q3?",
    "How many people is NovaMind hiring?",
]

print("=" * 60)
print("SEARCH RESULTS")
print("=" * 60)

for query in queries:
    print("")
    print("Query: '{}'".format(query))
    print("-" * 60)

    # embed the query — must use same model as documents
    query_embedding = model.encode(query).tolist()

    # search Chroma for top 2 most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2,
        include=["documents", "distances", "metadatas"]
    )

    # print each result with full details
    for rank, (doc, distance, metadata) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ), start=1):
        print("  Rank            : {}".format(rank))
        print("  Chunk index     : {}".format(metadata["chunk_index"]))
        print("  Chunk length    : {} characters".format(metadata["chunk_length"]))
        print("  Distance score  : {:.4f}  (lower = more similar)".format(distance))
        print("  Content preview :")
        # print full chunk text with indentation
        for line in doc.split("\n"):
            if line.strip():
                print("    {}".format(line.strip()))
        print("")

# ─────────────────────────────────────────────
# Summary — what this pipeline demonstrated
# ─────────────────────────────────────────────

print("=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print("  1. Document loaded    : novamind_sample.txt")
print("  2. Chunks created     : {}".format(len(chunks)))
print("  3. Embeddings created : {} x 384 dimensions".format(len(embeddings)))
print("  4. Stored in Chroma   : {} documents".format(collection.count()))
print("  5. Queries run        : {}".format(len(queries)))
print("")
print("This is the ingestion half of a RAG system.")
print("Missing piece: an LLM that reads retrieved chunks")
print("and generates a final answer. That is Week 2.")
print("=" * 60)

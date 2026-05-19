import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Load the embedding model
# Same model you have been using all day
# ─────────────────────────────────────────────

print("")
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.")
print("")

# ─────────────────────────────────────────────
# Create an in-memory Chroma client
# in-memory means data lives in RAM only
# nothing is saved to disk
# perfect for learning and quick experiments
# ─────────────────────────────────────────────

client = chromadb.Client()

# ─────────────────────────────────────────────
# Create a collection — like a table in a
# regular database, but stores vectors
# give it a name that describes what it holds
# ─────────────────────────────────────────────

collection = client.create_collection(name="novamind_docs")

# ─────────────────────────────────────────────
# Define 3 documents to add
# These are sample NovaMind internal documents
# ─────────────────────────────────────────────

documents = [
    "The team decided on per-seat pricing in Q2 after reviewing competitor models.",
    "Engineering roadmap for Q3 focuses on API performance and reducing latency.",
    "We are hiring two senior engineers to expand the backend infrastructure team.",
]

# ─────────────────────────────────────────────
# Embed all 3 documents using sentence-transformers
# model.encode() returns a numpy array of shape (3, 384)
# .tolist() converts to plain Python lists
# Chroma expects plain lists, not numpy arrays
# ─────────────────────────────────────────────

embeddings = model.encode(documents).tolist()

# ─────────────────────────────────────────────
# Add documents to the collection
# documents = the raw text (stored for display)
# embeddings = the vectors (used for search)
# ids        = unique identifier for each doc
# metadatas  = optional extra info per document
# ─────────────────────────────────────────────

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=["doc_1", "doc_2", "doc_3"],
    metadatas=[
        {"source": "pricing_meeting", "date": "2024-06-01"},
        {"source": "engineering_planning", "date": "2024-07-01"},
        {"source": "hr_update", "date": "2024-07-15"},
    ]
)

print("Added {} documents to collection.".format(len(documents)))
print("")

# ─────────────────────────────────────────────
# Define a search query — this simulates the
# CEO typing a question into NovaMind's system
# ─────────────────────────────────────────────

query = "What was decided about the pricing model?"

# ─────────────────────────────────────────────
# Embed the query using the same model
# CRITICAL: must use the same model as documents
# different model = wrong map = wrong results
# ─────────────────────────────────────────────

query_embedding = model.encode(query).tolist()

# ─────────────────────────────────────────────
# Query the collection
# n_results = how many similar docs to return
# Chroma ranks results by similarity automatically
# most similar document comes back first
# ─────────────────────────────────────────────

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,  # return top 2 most similar documents
    include=["documents", "distances", "metadatas"]
)

# ─────────────────────────────────────────────
# Print the results
# distances = how far each result is from query
# lower distance = more similar
# (Chroma uses distance not similarity by default)
# ─────────────────────────────────────────────

print("=" * 55)
print("Query: '{}'".format(query))
print("=" * 55)
print("")

for i, (doc, distance, metadata) in enumerate(zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
)):
    # convert distance to similarity score (0 to 1)
    # distance of 0 = identical, distance of 2 = opposite
    similarity = 1 - (distance / 2)

    print("Result {}:".format(i + 1))
    print("  Document : {}".format(doc))
    print("  Source   : {}".format(metadata["source"]))
    print("  Date     : {}".format(metadata["date"]))
    print("  Distance : {:.4f}  (lower = more similar)".format(distance))
    print("  Approx similarity: {:.4f}".format(similarity))
    print("")

# ─────────────────────────────────────────────
# Show collection size
# ─────────────────────────────────────────────

print("=" * 55)
print("Collection '{}' contains {} documents".format(
    collection.name,
    collection.count()
))
print("")
print("WHAT JUST HAPPENED:")
print("  1. Embedded 3 NovaMind documents and stored in Chroma")
print("  2. Embedded the search query")
print("  3. Chroma found the most similar documents instantly")
print("  4. No search algorithm written -- Chroma handled it")
print("  5. Results came back ranked -- best match first")
print("")
print("This is the foundation of NovaMind's knowledge base.")
print("Scale this to 300 documents and it works exactly the same way.")

import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------
# STEP 1: Load the embedding model
# all-MiniLM-L6-v2 is small, fast, and free — good for learning
# It converts any text into a 384-dimensional vector
# -----------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------------------------
# STEP 2: Create a Chroma client
# PersistentClient saves your vectors to disk
# So if you restart the script, your data is still there
# -----------------------------------------------------------
client = chromadb.PersistentClient(path="./chroma_db")

# -----------------------------------------------------------
# STEP 3: Create a collection
# A collection is like a table in SQL — but for vectors
# get_or_create means: use existing one if it already exists
# -----------------------------------------------------------
collection = client.get_or_create_collection(name="wikipedia_chunks")

# -----------------------------------------------------------
# STEP 4: Your 20 text chunks
# In a real system these would come from real documents
# Here we're using hand-written examples covering different topics
# -----------------------------------------------------------
documents = [
    "Python is a high-level programming language known for its simplicity.",
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by the structure of the human brain.",
    "The Eiffel Tower is located in Paris, France.",
    "Photosynthesis is the process by which plants convert sunlight into food.",
    "The Amazon rainforest produces 20% of the world's oxygen.",
    "Albert Einstein developed the theory of general relativity.",
    "DNA carries the genetic instructions for living organisms.",
    "The Roman Empire was one of the largest empires in ancient history.",
    "Black holes are regions of spacetime where gravity is extremely strong.",
    "Deep learning uses multiple layers of neural networks.",
    "The stock market reflects the economic health of a country.",
    "Volcanoes are formed by the movement of tectonic plates.",
    "Shakespeare wrote 37 plays and 154 sonnets.",
    "The human brain contains approximately 86 billion neurons.",
    "Quantum computing uses quantum mechanical phenomena to process information.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Climate change is caused by the accumulation of greenhouse gases.",
    "The internet was originally developed for military communication.",
    "Antibiotics are used to treat bacterial infections.",
]

# -----------------------------------------------------------
# STEP 5: Generate embeddings for all 20 documents
# model.encode() returns a numpy array — we convert to list
# because Chroma expects plain Python lists, not numpy arrays
# -----------------------------------------------------------
embeddings = model.encode(documents).tolist()

# -----------------------------------------------------------
# STEP 6: Insert documents into Chroma
# documents = the raw text (Chroma stores this for you to read back)
# embeddings = the vectors (Chroma uses these to search)
# ids = unique string identifier for each document (required)
# -----------------------------------------------------------
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"Inserted {collection.count()} documents into the collection.")

# -----------------------------------------------------------
# STEP 7: Run a similarity query
# We embed the query using the SAME model (critical)
# Then ask Chroma for the top 3 most similar documents
# -----------------------------------------------------------
query = "how do computers learn from data?"

query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

# -----------------------------------------------------------
# STEP 8: Print results clearly
# results["documents"] contains the matched text
# results["distances"] contains the similarity scores
# Lower distance = more similar in Chroma's default metric
# -----------------------------------------------------------
print(f"\nQuery: '{query}'")
print("\nTop 3 results:")
for i, (doc, distance) in enumerate(zip(
    results["documents"][0],
    results["distances"][0]
)):
    print(f"\n#{i+1} (distance: {distance:.4f})")
    print(f"  {doc}")
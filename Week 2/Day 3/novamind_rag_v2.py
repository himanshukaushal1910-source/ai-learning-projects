# novamind_rag_v2.py
# Week 2, Day 3 — Slot 5: Deploy/Extra
# RAG pipeline v2 — adds citations after every answer
# Changes from v1:
#   1. System prompt updated — model instructed to cite [Chunk 1], [Chunk 2] etc.
#   2. rag_query() prints a clean SOURCES section after every answer
#   3. Sources show chunk index, similarity score, and first 100 chars of text

# ── Imports ───────────────────────────────────────────────────────────────────
import os
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. Load the source document ───────────────────────────────────────────────
FILE_PATH = os.path.join(os.path.dirname(__file__), "novamind_sample.txt")

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

# ── 5. Set up Qdrant in-memory collection ─────────────────────────────────────
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="novamind",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
print("Qdrant collection 'novamind' created.\n")

# ── 6. Insert chunks as points ────────────────────────────────────────────────
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

# ── 7. Set up Groq client ─────────────────────────────────────────────────────
groq_client = Groq()

# ── 8. RAG system prompt v2 ───────────────────────────────────────────────────
# CHANGE FROM V1: Added citation instruction — Rule 5.
# The model now knows chunk labels [Chunk 1], [Chunk 2] etc. are source references.
# It is instructed to cite them inline in its answer.
# Code-based citation (the SOURCES section) runs regardless — belt and suspenders.
SYSTEM_PROMPT = """You are NovaMind's internal knowledge assistant.
You help employees find accurate information from official company documents.
You are concise, professional, and never speculate beyond what is stated.

STRICT RULES:
1. Answer ONLY using the information provided in the <context> block.
2. Do NOT use any knowledge from your training data or outside sources.
3. If the answer is not present in the context, respond with exactly:
   "I don't have enough information to answer this based on the available documents."
4. Be concise and factual. Do not speculate or infer beyond what is stated.
5. When referencing information, cite the source chunk inline using its label,
   for example: "According to [Chunk 1], the pricing model is per-seat."
6. Maximum 3 sentences unless the answer requires a list.
7. If the answer is a list of items, use bullet points."""

# ── 9. Helper: retrieve top-k chunks from Qdrant ─────────────────────────────
def retrieve(query: str, top_k: int = 3) -> list:
    """Embed query and retrieve top_k most similar chunks from Qdrant."""
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name="novamind",
        query=query_vector,
        limit=top_k,
        using=None,
    ).points
    return results

# ── 10. Helper: assemble context from retrieved chunks ────────────────────────
# Labels [Chunk 1], [Chunk 2] etc. are included so the model can cite them.
# The model's system prompt (Rule 5) instructs it to use these labels.
def assemble_context(results: list) -> str:
    """Build a labelled, delimited context block from retrieved chunks."""
    parts = []
    for i, result in enumerate(results, start=1):
        chunk_text = result.payload["chunk_text"]
        score = result.score
        parts.append(f"[Chunk {i}] (similarity: {score:.4f})\n{chunk_text}")
    return "\n\n---\n\n".join(parts)

# ── 11. Helper: build the full user message ───────────────────────────────────
def build_user_message(context: str, question: str) -> str:
    """Wrap context and question in clear XML-style delimiters."""
    return f"""Answer the question below using ONLY the context provided.
Cite the relevant chunk labels inline in your answer.

<context>
{context}
</context>

<question>
{question}
</question>"""

# ── 12. Helper: call Groq and get answer ──────────────────────────────────────
def generate_answer(context: str, question: str) -> str:
    """Send context + question to Groq and return the generated answer."""
    user_message = build_user_message(context, question)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content

# ── 13. Helper: print citations ───────────────────────────────────────────────
# CODE-BASED CITATIONS — printed after every answer programmatically.
# This is reliable — it always runs regardless of what the model does.
# Shows: chunk rank, similarity score, first 100 characters of chunk text.
# First 100 chars gives enough context to identify the source without overwhelming.
def print_citations(results: list):
    """Print a clean sources section after the answer."""
    print("\n── SOURCES ───────────────────────────────────────────────")
    for i, result in enumerate(results, start=1):
        chunk_preview = result.payload["chunk_text"][:100].replace("\n", " ")
        print(f"  [Chunk {i}]  Score: {result.score:.4f}")
        print(f"            \"{chunk_preview}...\"")
    print("─" * 65)

# ── 14. Helper: full RAG query v2 — retrieve + assemble + generate + cite ─────
# CHANGE FROM V1: citations printed after the answer via print_citations().
# Debug chunk print removed from production output — sources section replaces it.
# To re-enable debug chunks, uncomment the block below.
def rag_query(question: str):
    """Run full RAG pipeline v2: retrieve → assemble → generate → cite."""
    print("\n" + "=" * 65)
    print(f"QUESTION: {question}")
    print("=" * 65)

    # Step 1: Retrieve
    results = retrieve(question, top_k=3)

    # Step 2: Assemble context
    context = assemble_context(results)

    # Step 3: Generate answer
    print("\n── ANSWER ────────────────────────────────────────────────\n")
    answer = generate_answer(context, question)
    print(answer)

    # Step 4: Print citations — always runs, regardless of model behaviour
    # This is code-based citation — reliable and consistent.
    print_citations(results)
    print()

# ── 15. Run 2 test queries ────────────────────────────────────────────────────
print("=" * 65)
print("TEST QUERIES")
print("=" * 65)

test_queries = [
    "What pricing model did NovaMind decide on?",
    "What is the customer satisfaction score?",   # escape hatch test
]

for q in test_queries:
    rag_query(q)

# ── 16. Live query loop ───────────────────────────────────────────────────────
print("=" * 65)
print("LIVE QUERY MODE  (type 'exit' to quit)")
print("=" * 65)

while True:
    user_input = input("\nAsk a question: ").strip()
    if user_input.lower() == "exit":
        print("Session ended.")
        break
    if not user_input:
        print("Empty question — please type something.")
        continue
    rag_query(user_input)

# novamind_rag.py
# Week 2, Day 3 — Slot 3: Build
# Complete RAG pipeline: retrieval (Qdrant) + generation (Groq)
# This is where everything from Week 1 and Day 1-2 connects.

# ── Imports ───────────────────────────────────────────────────────────────────
import os
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. Load the source document ───────────────────────────────────────────────
# novamind_sample.txt — the same document from Week 1.
# os.path.dirname(__file__) ensures we look in the same folder as this script.
FILE_PATH = os.path.join(os.path.dirname(__file__), "novamind_sample.txt")

print("Loading document...")
with open(FILE_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()
print(f"Document loaded — {len(raw_text)} characters.\n")

# ── 2. Chunk the document ─────────────────────────────────────────────────────
# Same settings as Week 1 and Day 1:
#   chunk_size=400    → max characters per chunk
#   chunk_overlap=80  → overlap preserves context across chunk boundaries
# Minimum length filter drops header-only or near-empty chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)
raw_chunks = splitter.split_text(raw_text)
chunks = [c for c in raw_chunks if len(c) >= 100]
print(f"Chunking complete — {len(raw_chunks)} raw → {len(chunks)} after filter.\n")

# ── 3. Load embedding model ───────────────────────────────────────────────────
# all-MiniLM-L6-v2 → 384-dimensional vectors.
# Same model used for both chunks AND queries — they must be in the same space.
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── 4. Embed all chunks ───────────────────────────────────────────────────────
# model.encode() returns shape (num_chunks, 384).
# .tolist() converts numpy array → Python list for Qdrant.
print("Embedding chunks...")
embeddings = model.encode(chunks).tolist()
print(f"Embedded {len(embeddings)} chunks.\n")

# ── 5. Set up Qdrant in-memory collection ─────────────────────────────────────
# In-memory mode — no Docker needed, resets when script ends.
# Distance.COSINE → cosine similarity (higher score = more similar).
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="novamind",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
print("Qdrant collection 'novamind' created.\n")

# ── 6. Insert chunks as points ────────────────────────────────────────────────
# PointStruct = one record: id + vector + payload.
# Payload stores the original text so we can display it in results.
# Qdrant stores vectors only — text lives in the payload.
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
# Groq() automatically reads GROQ_API_KEY from environment.
# Uses OpenAI-compatible API — same message structure, same response format.
groq_client = Groq()

# ── 8. RAG system prompt ──────────────────────────────────────────────────────
# This is the instruction layer — tells the model:
#   - Who it is
#   - Where to get its answers from (ONLY the provided context)
#   - What to do when the answer isn't in the context (escape hatch)
# The escape hatch is critical — without it the model hallucinates.
SYSTEM_PROMPT = """You are NovaMind's internal knowledge assistant.
You help employees find accurate information from company documents.

STRICT RULES:
1. Answer ONLY using the information provided in the <context> block below.
2. Do NOT use any knowledge from your training data or outside sources.
3. If the answer is not present in the context, respond with exactly:
   "I don't have enough information to answer this based on the available documents."
4. Be concise and factual. Do not speculate or infer beyond what is stated."""

# ── 9. Helper: retrieve top-k chunks from Qdrant ─────────────────────────────
# Embeds the query using the same model as the chunks.
# Returns top_k most similar chunks as a list of payload dicts.
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
# Joins chunks with clear labels and separators.
# Labels like [Chunk 1] help the model reason about multiple sources.
# Separator "---" visually separates chunks for the model.
# This structure directly reduces context failure (failure mode 2).
def assemble_context(results: list) -> str:
    """Build a labelled, delimited context block from retrieved chunks."""
    parts = []
    for i, result in enumerate(results, start=1):
        chunk_text = result.payload["chunk_text"]
        score = result.score
        parts.append(f"[Chunk {i}] (similarity: {score:.4f})\n{chunk_text}")
    return "\n\n---\n\n".join(parts)

# ── 11. Helper: build the full prompt ─────────────────────────────────────────
# Uses XML-style tags to clearly separate context from question.
# Models respond well to structured delimiters — reduces context confusion.
# This is the prompt template from Day 2 Slot 4 put into practice.
def build_user_message(context: str, question: str) -> str:
    """Wrap context and question in clear XML-style delimiters."""
    return f"""Answer the question below using ONLY the context provided.

<context>
{context}
</context>

<question>
{question}
</question>"""

# ── 12. Helper: call Groq and get answer ──────────────────────────────────────
# Sends system prompt + assembled user message to LLaMA 3.3 70B.
# temperature=0  → deterministic, factual answers
# max_tokens=500 → prevents runaway responses
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

# ── 13. Helper: full RAG query — retrieve + assemble + generate ───────────────
# This is the complete RAG pipeline in one function.
# Debug print shows retrieved chunks BEFORE the answer.
# If the answer is wrong, you check the chunks first — that's retrieval debugging.
def rag_query(question: str):
    """Run full RAG pipeline: retrieve → assemble → generate → print."""
    print("\n" + "=" * 65)
    print(f"QUESTION: {question}")
    print("=" * 65)

    # Step 1: Retrieve
    results = retrieve(question, top_k=3)

    # Step 2: Debug — print retrieved chunks BEFORE generating answer
    # This is your window into retrieval quality.
    # If the answer is wrong, look here first — retrieval failure is invisible.
    print("\n── RETRIEVED CHUNKS (debug) ──────────────────────────────")
    for i, result in enumerate(results, start=1):
        print(f"\n[Chunk {i}] Score: {result.score:.4f}")
        print(f"{result.payload['chunk_text']}")
    print("\n── END RETRIEVED CHUNKS ──────────────────────────────────")

    # Step 3: Assemble context
    context = assemble_context(results)

    # Step 4: Generate answer
    print("\n── ANSWER ────────────────────────────────────────────────\n")
    answer = generate_answer(context, question)
    print(answer)
    print()

# ── 14. Run 4 preset test queries ────────────────────────────────────────────
# Query 4 is the escape hatch test — customer satisfaction is NOT in the doc.
# If prompt engineering works, the model should say "I don't have enough info."
# If it hallucinates an answer — your system prompt needs strengthening.
print("=" * 65)
print("PRESET TEST QUERIES")
print("=" * 65)

preset_queries = [
    "What pricing model did NovaMind decide on?",
    "What are the engineering priorities for Q3?",
    "How many people is NovaMind hiring and for which roles?",
    "What is the customer satisfaction score?",   # escape hatch test
]

for q in preset_queries:
    rag_query(q)

# ── 15. Live query loop ───────────────────────────────────────────────────────
# User types questions, gets grounded answers, types 'exit' to quit.
# Each query runs the full RAG pipeline fresh.
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

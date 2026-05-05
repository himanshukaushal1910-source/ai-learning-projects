"""
NovaMind RAG — Re-ranking + LangChain Comparison — Week 3, Day 2

Three pipelines side by side:
  A — Dense only (raw Python, Week 2 baseline)
  B — Hybrid + cross-encoder reranking (raw Python)
  C — LangChain RAG (same result, less code)

Requirements:
    pip install qdrant-client rank-bm25 sentence-transformers groq
    pip install langchain langchain-groq langchain-huggingface langchain-qdrant

Run Qdrant first:
    docker run -p 6333:6333 qdrant/qdrant
"""

import os
import sys
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain imports — using LCEL (LangChain Expression Language)
# This is the modern LangChain approach, works with all recent versions
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# ── Configuration ──────────────────────────────────────────────────────────────

COLLECTION_NAME = "novamind_day2"
LC_COLLECTION = "novamind_langchain"
DENSE_SIZE = 384
GROQ_MODEL = "llama-3.3-70b-versatile"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60

# ── NovaMind documents ─────────────────────────────────────────────────────────

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

# ── Step 1: Load models ────────────────────────────────────────────────────────

print("Loading dense embedding model...")
dense_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Dense model loaded.")

print("Loading cross-encoder re-ranker (downloads ~80MB on first run)...")
reranker = CrossEncoder(RERANK_MODEL)
print("Re-ranker loaded.")

# ── Step 2: Qdrant connection ──────────────────────────────────────────────────

print("Connecting to Qdrant...")
try:
    client = QdrantClient("localhost", port=6333)
    client.get_collections()
    print("Connected to Qdrant.")
except Exception as e:
    print(f"ERROR: Cannot connect to Qdrant: {e}")
    print("Run: docker run -p 6333:6333 qdrant/qdrant")
    sys.exit(1)

# ── Step 3: Collection setup ───────────────────────────────────────────────────

existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=DENSE_SIZE, distance=Distance.COSINE)
)
print(f"Created collection: {COLLECTION_NAME}")

# ── Step 4: Chunk all documents ────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
)

all_chunks = []  # list of (text, source)
for source, text in DOCUMENTS.items():
    for chunk in splitter.split_text(text.strip()):
        if len(chunk) > 80:
            all_chunks.append((chunk, source))

print(f"Total chunks: {len(all_chunks)}")

# ── Step 5: Build BM25 index ───────────────────────────────────────────────────

texts = [chunk for chunk, _ in all_chunks]
tokenised = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenised)
print("BM25 index built.")

# ── Step 6: Embed + insert into Qdrant ────────────────────────────────────────

print("Embedding and inserting into Qdrant...")
dense_vectors = dense_model.encode(texts, show_progress_bar=False)

points = []
for i, ((text, source), vec) in enumerate(zip(all_chunks, dense_vectors)):
    points.append(PointStruct(
        id=i,
        vector=vec.tolist(),
        payload={"text": text, "source": source}
    ))

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Inserted {len(points)} points.\n")

# ══════════════════════════════════════════════════════════════════════════════
# RAW PYTHON HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def dense_search(query: str, top_k: int = 10):
    qvec = dense_model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=top_k,
    )
    return [(p.id, p.score, p.payload) for p in results.points]


def bm25_search(query: str, top_k: int = 10):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        text, source = all_chunks[idx]
        results.append((idx, score, {"text": text, "source": source}))
    return results


def rrf_fusion(dense_results, sparse_results, top_k: int = 10):
    rrf_scores = {}
    chunk_data = {}
    for rank, (idx, score, payload) in enumerate(dense_results, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + RRF_K)
        chunk_data[idx] = payload
    for rank, (idx, score, payload) in enumerate(sparse_results, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + RRF_K)
        chunk_data[idx] = payload
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(idx, score, chunk_data[idx]) for idx, score in ranked]


def rerank(query: str, candidates: list, top_k: int = 3):
    """
    Cross-encoder reranking.
    Feeds [query, chunk] pairs jointly into cross-encoder.
    Returns top_k by cross-encoder score.
    """
    pairs = [[query, item[2]["text"]] for item in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:top_k]]


def build_context(results) -> str:
    parts = []
    for i, (idx, score, payload) in enumerate(results, 1):
        parts.append(f"[Chunk {i} | Source: {payload.get('source')}]\n{payload.get('text')}")
    return "\n\n---\n\n".join(parts)


def ask_groq_raw(query: str, context: str) -> str:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    system_prompt = """You are NovaMind's internal knowledge assistant.
Answer ONLY using the information in the <context> tags.
If context is insufficient say: "I don't have enough information to answer this."
Cite the chunk your answer comes from e.g. [Chunk 1].
Maximum 3 sentences."""

    user_msg = f"<context>\n{context}\n</context>\n\n<question>\n{query}\n</question>"

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Groq error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# LANGCHAIN PIPELINE — LCEL (LangChain Expression Language)
# Modern approach: chain components with | operator
# ══════════════════════════════════════════════════════════════════════════════

print("Setting up LangChain pipeline...")

# Same embedding model, wrapped in LangChain interface
lc_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Convert chunks to LangChain Document objects
lc_docs = [
    Document(page_content=text, metadata={"source": source})
    for text, source in all_chunks
]

# Build vector store — embeds + inserts in one call
# Uses a separate collection so it doesn't collide with raw Python data
lc_vectorstore = QdrantVectorStore.from_documents(
    documents=lc_docs,
    embedding=lc_embeddings,
    url="http://localhost:6333",
    collection_name=LC_COLLECTION,
    force_recreate=True,
)
print("LangChain vector store ready.")

# Retriever — standard interface over Qdrant
lc_retriever = lc_vectorstore.as_retriever(search_kwargs={"k": 5})

# LLM
lc_llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Prompt template
lc_prompt = ChatPromptTemplate.from_template("""
You are NovaMind's internal knowledge assistant.
Answer ONLY using the context below.
If context is insufficient say: "I don't have enough information to answer this."
Keep answers to 3 sentences maximum.

Context:
{context}

Question: {question}
""")


def format_docs(docs):
    """Convert retrieved LangChain Documents into a context string."""
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


# LCEL chain: retriever | format | prompt | llm | parse
# The | operator chains components — output of each feeds into the next
lc_chain = (
    {"context": lc_retriever | format_docs, "question": RunnablePassthrough()}
    | lc_prompt
    | lc_llm
    | StrOutputParser()
)

print("LangChain chain ready.\n")


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def compare(query: str):
    """
    Run query through three pipelines and print results side by side.
    A: dense only
    B: hybrid + cross-encoder reranking
    C: LangChain
    """
    print("\n" + "═" * 70)
    print(f"QUERY: {query}")
    print("═" * 70)

    # ── A: Dense only ─────────────────────────────────────────────────────────
    dense_results = dense_search(query, top_k=3)
    print("\n── A: DENSE ONLY (no reranking) ────────────────────────────")
    for i, (idx, score, payload) in enumerate(dense_results, 1):
        preview = payload["text"][:110].replace("\n", " ")
        print(f"  [{i}] source={payload['source']} score={score:.3f} | {preview}...")
    context_a = build_context(dense_results)
    answer_a = ask_groq_raw(query, context_a)
    print(f"  ANSWER: {answer_a}")

    # ── B: Hybrid + reranking ──────────────────────────────────────────────────
    # Stage 1: hybrid retrieval — top 10 candidates
    d_res = dense_search(query, top_k=10)
    s_res = bm25_search(query, top_k=10)
    candidates = rrf_fusion(d_res, s_res, top_k=10)

    # Stage 2: cross-encoder reranks candidates → top 3
    reranked = rerank(query, candidates, top_k=3)

    print("\n── B: HYBRID + CROSS-ENCODER RERANKING ─────────────────────")
    for i, (idx, score, payload) in enumerate(reranked, 1):
        preview = payload["text"][:110].replace("\n", " ")
        print(f"  [{i}] source={payload['source']} | {preview}...")
    context_b = build_context(reranked)
    answer_b = ask_groq_raw(query, context_b)
    print(f"  ANSWER: {answer_b}")

    # ── C: LangChain ──────────────────────────────────────────────────────────
    print("\n── C: LANGCHAIN RAG ────────────────────────────────────────")
    try:
        # Retrieve docs separately so we can print them
        lc_retrieved = lc_retriever.invoke(query)
        for i, doc in enumerate(lc_retrieved[:3], 1):
            preview = doc.page_content[:110].replace("\n", " ")
            source = doc.metadata.get("source", "unknown")
            print(f"  [{i}] source={source} | {preview}...")
        # Run the full chain for the answer
        lc_answer = lc_chain.invoke(query)
        print(f"  ANSWER: {lc_answer}")
    except Exception as e:
        print(f"  LangChain error: {e}")


# ── Demo queries ───────────────────────────────────────────────────────────────

print("█" * 70)
print("  NOVAMIND — DENSE vs RERANKING vs LANGCHAIN")
print("█" * 70)

compare("What is the parental leave policy for primary caregivers?")
compare("What does the MX-4400 connector support?")
compare("How does the LLM-7B fine-tuning module work?")

# ── Interactive mode ───────────────────────────────────────────────────────────

print("\n" + "═" * 70)
print("INTERACTIVE MODE — try your own queries (or 'quit' to exit)")
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

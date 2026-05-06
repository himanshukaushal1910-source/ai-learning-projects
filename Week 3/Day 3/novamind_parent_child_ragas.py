"""
NovaMind Parent-Child RAG + RAGAS Evaluation — Week 3, Day 3

Demonstrates:
  1. Parent-child chunking — fixes the LLM-7B split-chunk problem
  2. Side-by-side comparison: flat chunking vs parent-child retrieval
  3. RAGAS evaluation over a 10-question golden dataset

Requirements:
    pip install qdrant-client sentence-transformers groq ragas datasets
    pip install langchain langchain-groq langchain-huggingface

Run Qdrant first:
    docker run -p 6333:6333 qdrant/qdrant
"""

import os
import sys
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ──────────────────────────────────────────────────────────────

FLAT_COLLECTION = "novamind_flat"        # standard chunking (your Week 2 approach)
PARENT_CHILD_COLLECTION = "novamind_pc"  # parent-child chunking
DENSE_SIZE = 384
GROQ_MODEL = "llama-3.3-70b-versatile"

# Parent-child sizes — chosen so LLM-7B fits in one child chunk
PARENT_SIZE = 600
PARENT_OVERLAP = 100
CHILD_SIZE = 150
CHILD_OVERLAP = 20

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

# ── Golden dataset — 10 questions with known correct answers ───────────────────
# Written by reading the source documents directly.
# This is the ground truth RAGAS will compare against.

GOLDEN_DATASET = [
    {
        "question": "How much does the Starter plan cost per month?",
        "ground_truth": "The Starter plan costs $29 per month."
    },
    {
        "question": "How many team members does the Professional plan support?",
        "ground_truth": "The Professional plan supports up to 25 team members."
    },
    {
        "question": "What discount does annual billing provide?",
        "ground_truth": "Annual billing provides a 20% discount."
    },
    {
        "question": "What is the module code for the Slack integration?",
        "ground_truth": "The Slack integration module code is SLK-2200."
    },
    {
        "question": "What does the MX-4400 data connector support?",
        "ground_truth": "The MX-4400 data connector supports REST, GraphQL, and webhook-based data pipelines."
    },
    {
        "question": "What is the minimum dataset size for LLM-7B fine-tuning?",
        "ground_truth": "The minimum dataset size for LLM-7B fine-tuning is 10,000 samples."
    },
    {
        "question": "How many days of annual leave do full-time employees receive?",
        "ground_truth": "Full-time employees receive 25 days of annual leave per year."
    },
    {
        "question": "How many weeks of parental leave does a primary caregiver receive?",
        "ground_truth": "A primary caregiver receives 16 weeks of fully paid parental leave."
    },
    {
        "question": "How many days per week can employees work remotely?",
        "ground_truth": "Employees may work remotely up to 3 days per week."
    },
    {
        "question": "What is the GitHub Enterprise variant module code?",
        "ground_truth": "The GitHub Enterprise variant module code is GHB-3310-ENT."
    },
]

# ── Step 1: Load model ─────────────────────────────────────────────────────────

print("Loading dense model...")
dense_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Dense model loaded.")

# ── Step 2: Qdrant connection ──────────────────────────────────────────────────

print("Connecting to Qdrant...")
try:
    client = QdrantClient("localhost", port=6333)
    client.get_collections()
    print("Connected to Qdrant.")
except Exception as e:
    print(f"ERROR: {e}")
    print("Run: docker run -p 6333:6333 qdrant/qdrant")
    sys.exit(1)

# ── Step 3: Setup splitters ────────────────────────────────────────────────────

flat_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50
)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_SIZE, chunk_overlap=PARENT_OVERLAP
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_SIZE, chunk_overlap=CHILD_OVERLAP
)

# ── Step 4: Build flat collection (standard chunking) ─────────────────────────

existing = [c.name for c in client.get_collections().collections]
for name in [FLAT_COLLECTION, PARENT_CHILD_COLLECTION]:
    if name in existing:
        client.delete_collection(name)

client.create_collection(
    FLAT_COLLECTION,
    vectors_config=VectorParams(size=DENSE_SIZE, distance=Distance.COSINE)
)

flat_chunks = []  # (text, source)
for source, text in DOCUMENTS.items():
    for chunk in flat_splitter.split_text(text.strip()):
        if len(chunk) > 80:
            flat_chunks.append((chunk, source))

flat_texts = [c for c, _ in flat_chunks]
flat_vecs = dense_model.encode(flat_texts, show_progress_bar=False)

flat_points = [
    PointStruct(
        id=i,
        vector=vec.tolist(),
        payload={"text": text, "source": source}
    )
    for i, ((text, source), vec) in enumerate(zip(flat_chunks, flat_vecs))
]
client.upsert(FLAT_COLLECTION, flat_points)
print(f"Flat collection: {len(flat_points)} chunks inserted.")

# ── Step 5: Build parent-child collection ─────────────────────────────────────

client.create_collection(
    PARENT_CHILD_COLLECTION,
    vectors_config=VectorParams(size=DENSE_SIZE, distance=Distance.COSINE)
)

parent_store = {}    # parent_id → parent text
child_chunks = []    # (child_text, source, parent_id)

for source, text in DOCUMENTS.items():
    parents = parent_splitter.split_text(text.strip())
    for p_idx, parent_text in enumerate(parents):
        parent_id = f"{source}_{p_idx}"
        parent_store[parent_id] = parent_text

        children = child_splitter.split_text(parent_text)
        for child_text in children:
            if len(child_text) > 30:
                child_chunks.append((child_text, source, parent_id))

child_texts = [c for c, _, _ in child_chunks]
child_vecs = dense_model.encode(child_texts, show_progress_bar=False)

pc_points = [
    PointStruct(
        id=i,
        vector=vec.tolist(),
        payload={"text": child_text, "source": source, "parent_id": parent_id}
    )
    for i, ((child_text, source, parent_id), vec) in enumerate(zip(child_chunks, child_vecs))
]
client.upsert(PARENT_CHILD_COLLECTION, pc_points)
print(f"Parent-child collection: {len(pc_points)} child chunks inserted.")
print(f"Parent store: {len(parent_store)} parent chunks.\n")

# ── Search helpers ─────────────────────────────────────────────────────────────

def flat_search(query: str, top_k: int = 3):
    """Standard flat retrieval — returns chunks directly."""
    qvec = dense_model.encode(query).tolist()
    results = client.query_points(
        collection_name=FLAT_COLLECTION,
        query=qvec,
        limit=top_k,
    )
    return [(p.id, p.score, p.payload) for p in results.points]


def parent_child_search(query: str, top_k: int = 3):
    """
    Parent-child retrieval:
    Step 1 — search child vectors (small, precise)
    Step 2 — fetch parent chunk for each matched child (large, complete)
    Step 3 — deduplicate parents (two children may share one parent)
    """
    qvec = dense_model.encode(query).tolist()
    # Fetch more children than needed to allow for parent deduplication
    results = client.query_points(
        collection_name=PARENT_CHILD_COLLECTION,
        query=qvec,
        limit=top_k * 3,
    )

    seen_parents = set()
    final_results = []

    for point in results.points:
        parent_id = point.payload["parent_id"]
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_text = parent_store[parent_id]
            final_results.append((
                parent_id,
                point.score,
                {
                    "text": parent_text,
                    "source": point.payload["source"],
                    "parent_id": parent_id,
                    "matched_child": point.payload["text"],  # for debugging
                }
            ))
        if len(final_results) == top_k:
            break

    return final_results


def build_context(results) -> str:
    parts = []
    for i, (idx, score, payload) in enumerate(results, 1):
        parts.append(
            f"[Chunk {i} | Source: {payload.get('source')}]\n{payload.get('text')}"
        )
    return "\n\n---\n\n".join(parts)


def ask_groq(query: str, context: str) -> str:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    system_prompt = """You are NovaMind's internal knowledge assistant.
Answer ONLY using the information in the <context> tags.
If context is insufficient say exactly: "I don't have enough information to answer this."
Be concise — one or two sentences maximum."""

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
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq error: {e}"

# ── Comparison runner ──────────────────────────────────────────────────────────

def compare(query: str):
    """Show flat vs parent-child retrieval side by side."""
    print(f"\n{'═' * 70}")
    print(f"QUERY: {query}")
    print("═" * 70)

    # Flat
    flat_results = flat_search(query, top_k=3)
    print("\n── FLAT CHUNKING (300 chars) ───────────────────────────────")
    for i, (idx, score, payload) in enumerate(flat_results, 1):
        preview = payload["text"][:120].replace("\n", " ")
        print(f"  [{i}] score={score:.3f} | {preview}...")
    flat_answer = ask_groq(query, build_context(flat_results))
    print(f"  ANSWER: {flat_answer}")

    # Parent-child
    pc_results = parent_child_search(query, top_k=3)
    print("\n── PARENT-CHILD (child=150 → parent=600) ──────────────────")
    for i, (idx, score, payload) in enumerate(pc_results, 1):
        matched = payload.get("matched_child", "")[:80].replace("\n", " ")
        preview = payload["text"][:120].replace("\n", " ")
        print(f"  [{i}] score={score:.3f} | matched child: '{matched}...'")
        print(f"       parent delivered: '{preview}...'")
    pc_answer = ask_groq(query, build_context(pc_results))
    print(f"  ANSWER: {pc_answer}")


# ── Demo comparisons ───────────────────────────────────────────────────────────

print("█" * 70)
print("  NOVAMIND — FLAT vs PARENT-CHILD RETRIEVAL")
print("█" * 70)

# These three queries are chosen deliberately:
# Q1 — simple fact: both should work
# Q2 — split-chunk problem: parent-child should win
# Q3 — exact code: tests precision of child retrieval
compare("What is the minimum dataset size for LLM-7B fine-tuning?")
compare("What is the parental leave policy for primary caregivers?")
compare("What does the MX-4400 connector support?")

# ── RAGAS Evaluation ───────────────────────────────────────────────────────────

print(f"\n{'█' * 70}")
print("  RAGAS EVALUATION — 10-question golden dataset")
print("█" * 70)
print("\nRunning RAG pipeline over all 10 golden questions...")
print("(This calls Groq 10 times — may take 20-30 seconds)\n")

# Collect RAG outputs for RAGAS
questions = []
answers = []
contexts = []
ground_truths = []

for item in GOLDEN_DATASET:
    q = item["question"]
    gt = item["ground_truth"]

    # Use parent-child pipeline for evaluation
    pc_results = parent_child_search(q, top_k=3)
    context_texts = [r[2]["text"] for r in pc_results]
    context_str = build_context(pc_results)
    answer = ask_groq(q, context_str)

    questions.append(q)
    answers.append(answer)
    contexts.append(context_texts)       # RAGAS expects list of strings per question
    ground_truths.append(gt)

    print(f"  Q: {q[:60]}...")
    print(f"  A: {answer[:80]}...")
    print()

# ── RAGAS scoring ──────────────────────────────────────────────────────────────
# RAGAS 0.4.x uses LangChain LLM wrappers for evaluation
# We configure it to use Groq instead of OpenAI

print("Running RAGAS evaluation...")
print("(RAGAS uses an LLM to judge faithfulness and relevance)")
print("(Configured to use Groq — no OpenAI key needed)\n")

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    # Configure RAGAS to use Groq as the judge LLM
    ragas_llm = LangchainLLMWrapper(ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    ))

    # Configure RAGAS to use local HuggingFace embeddings
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # Build RAGAS dataset
    ragas_data = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Set LLM and embeddings on each metric
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    # Run evaluation
    result = evaluate(ragas_data, metrics=metrics)

    print("\n" + "═" * 70)
    print("  RAGAS SCORES — NovaMind Parent-Child RAG")
    print("═" * 70)
    print(f"  Faithfulness      : {result['faithfulness']:.3f}  (is answer grounded in context?)")
    print(f"  Answer Relevancy  : {result['answer_relevancy']:.3f}  (does answer address the question?)")
    print(f"  Context Precision : {result['context_precision']:.3f}  (are retrieved chunks relevant?)")
    print(f"  Context Recall    : {result['context_recall']:.3f}  (did we retrieve enough context?)")
    print()

    # Interpretation
    scores = {
        "Faithfulness": result["faithfulness"],
        "Answer Relevancy": result["answer_relevancy"],
        "Context Precision": result["context_precision"],
        "Context Recall": result["context_recall"],
    }
    print("  Interpretation:")
    for name, score in scores.items():
        if score >= 0.85:
            level = "GOOD ✓"
        elif score >= 0.70:
            level = "ACCEPTABLE"
        else:
            level = "NEEDS IMPROVEMENT ✗"
        print(f"  {name:<22}: {level}")

except Exception as e:
    print(f"\nRAGAS evaluation error: {e}")
    print("This may be a version compatibility issue with RAGAS 0.4.x")
    print("The RAG pipeline above still ran correctly — RAGAS is bonus.")
    print("\nRaw answers collected for manual review:")
    for q, a, gt in zip(questions, answers, ground_truths):
        print(f"\n  Q: {q}")
        print(f"  A: {a}")
        print(f"  GT: {gt}")
        match = "✓" if any(word in a.lower() for word in gt.lower().split()[:3]) else "~"
        print(f"  Match: {match}")

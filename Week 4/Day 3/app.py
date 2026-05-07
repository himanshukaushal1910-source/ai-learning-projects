"""
NovaMind Knowledge Base — Hugging Face Spaces
=============================================
This file combines the RAG pipeline from Weeks 2-3 with a Gradio frontend
for deployment on Hugging Face Spaces.

Key differences from the local versions:
1. Qdrant is in-memory (Spaces runs one process, no docker-compose)
2. GROQ_API_KEY comes from Hugging Face Secrets (not Windows env vars)
3. Document is loaded from the same folder as this file
4. Entry point is app.py — Spaces requires this exact filename
5. demo.launch() at the bottom starts the Gradio server

How Hugging Face Spaces works:
- You push this file + requirements.txt + novamind_sample.txt to your Space
- Spaces installs requirements, runs app.py, gives you a public URL
- Anyone with the URL can use the UI — no API knowledge needed
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DOCUMENT_PATH = "novamind_sample.txt"   # must be in same folder as app.py
COLLECTION_NAME = "novamind_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384
TOP_K = 3
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are NovaMind's internal knowledge assistant.
Your job is to answer questions using ONLY the context provided below.
The context contains excerpts from NovaMind's internal documents.

Rules you must follow without exception:
1. Answer ONLY from information explicitly stated word-for-word in the context.
2. Do not infer, extrapolate, or fill gaps with general knowledge.
3. If the answer is not explicitly in the context, say exactly:
   "I don't have enough information to answer this question."
4. Always cite which chunk your answer came from using [Chunk N] notation.
5. Keep answers concise — 2-3 sentences maximum unless the question requires more.
"""


# ==============================================================================
# STARTUP — load everything once when the Space starts
# In Gradio we don't have FastAPI's lifespan pattern.
# Instead we load at module level — runs once when Python imports this file.
# All subsequent requests reuse these already-loaded objects.
# ==============================================================================

print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

# In-memory Qdrant — no persistence, but fine for a demo Space.
# Vectors reload from the document every time the Space restarts.
print("Setting up Qdrant in-memory...")
qdrant = QdrantClient(":memory:")
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBEDDING_DIMS, distance=Distance.COSINE)
)

# Load and chunk the NovaMind document
print(f"Loading document: {DOCUMENT_PATH}")
with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(raw_text)
chunks = [c for c in chunks if len(c) > 100]
print(f"Created {len(chunks)} chunks")

# Embed all chunks and store in Qdrant
print("Embedding and indexing chunks...")
embeddings = embedder.encode(chunks)
points = [
    PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={
            "text": chunks[i],
            "chunk_index": i,
            "preview": chunks[i][:80]
        }
    )
    for i in range(len(chunks))
]
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Indexed {len(chunks)} chunks. Ready.")

# Connect to Groq
# On Hugging Face Spaces, GROQ_API_KEY is injected from Secrets at runtime.
# os.environ reads it automatically — same as how it worked on your Windows machine.
groq_client = Groq()


# ==============================================================================
# RAG PIPELINE — same logic as all previous versions
# ==============================================================================

def run_rag(question: str) -> tuple[str, str]:
    """
    Run the full RAG pipeline for a given question.

    Returns:
        answer (str): grounded answer from Groq
        sources (str): formatted source citations for display

    This function is called by Gradio every time the user clicks Submit.
    """
    if not question.strip():
        return "Please enter a question.", ""

    # Embed the question
    query_vector = embedder.encode(question).tolist()

    # Retrieve top-k chunks from Qdrant
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K
    ).points

    # Assemble context block with labelled chunks
    context_parts = []
    for i, result in enumerate(results):
        context_parts.append(f"[Chunk {i+1}]\n{result.payload['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    # Build prompt with XML delimiters
    user_prompt = f"""<context>
{context_block}
</context>

<question>
{question}
</question>

Answer the question using ONLY the information in the context above.
If the answer is not in the context, say exactly: "I don't have enough information to answer this question."
"""

    # Call Groq
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    # Format sources for display in the UI
    # Each source shows chunk number, similarity score, and a preview
    source_lines = []
    for i, result in enumerate(results):
        score = round(result.score, 4)
        preview = result.payload['preview']
        source_lines.append(f"[Chunk {i+1}] Score: {score}\n{preview}...")

    sources_text = "\n\n".join(source_lines)

    return answer, sources_text


# ==============================================================================
# GRADIO UI
# gr.Interface reads your function signature and builds the UI automatically.
# inputs and outputs define what the user sees.
# title, description appear at the top of the page.
# examples give users starter questions to try immediately.
# ==============================================================================

demo = gr.Interface(
    fn=run_rag,

    inputs=gr.Textbox(
        label="Ask a question about NovaMind",
        placeholder="e.g. What is NovaMind's hiring plan?",
        lines=2
    ),

    outputs=[
        gr.Textbox(
            label="Answer",
            lines=5
        ),
        gr.Textbox(
            label="Sources used",
            lines=6
        )
    ],

    title="NovaMind Knowledge Base",
    description="""
    Ask questions about NovaMind's internal documents.
    This system uses RAG (Retrieval Augmented Generation) to find relevant
    information and generate grounded answers with source citations.

    Built with: Qdrant · sentence-transformers · Groq (LLaMA 3.3 70B) · FastAPI · Gradio
    """,

    # Example questions shown at the bottom — users can click to auto-fill
    examples=[
        ["What is NovaMind's hiring plan?"],
        ["What happened in Q2 review?"],
        ["What is NovaMind's approach to pricing?"],
        ["What are the Q3 engineering goals?"],
    ],

    # Show the submit button clearly
    submit_btn="Ask NovaMind",
    clear_btn="Clear",
)


# ==============================================================================
# LAUNCH
# demo.launch() starts the Gradio server.
# On Hugging Face Spaces, Spaces calls this automatically and handles the URL.
# Locally, this opens http://localhost:7860 in your browser.
# ==============================================================================

if __name__ == "__main__":
    demo.launch()

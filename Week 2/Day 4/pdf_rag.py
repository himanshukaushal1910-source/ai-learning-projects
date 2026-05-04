# pdf_rag.py
# Week 2, Day 4 — Slot 5: Deploy/Extra (updated with error handling)
# PDF RAG pipeline — drop in any PDF, ask questions, get cited answers
# v3: Added error handling for empty PDFs and Groq API failures

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys
import re
import fitz
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. Resolve PDF path ───────────────────────────────────────────────────────
if len(sys.argv) > 1:
    PDF_PATH = sys.argv[1]
else:
    PDF_PATH = os.path.join(os.path.dirname(__file__), "novamind_sample.pdf")

print(f"Loading PDF: {PDF_PATH}")

# ERROR HANDLING: file not found
if not os.path.exists(PDF_PATH):
    print(f"ERROR: File not found — {PDF_PATH}")
    print("Please check the path and try again.")
    sys.exit(1)

# ── 2. Extract text page by page using PyMuPDF ───────────────────────────────
try:
    doc = fitz.open(PDF_PATH)
except Exception as e:
    print(f"ERROR: Could not open PDF — {e}")
    sys.exit(1)

print(f"PDF loaded — {doc.page_count} pages.\n")

page_texts = []
for page in doc:
    raw_page_text = page.get_text()
    page_texts.append((page.number + 1, raw_page_text))

page_count = doc.page_count
doc.close()

# ── 3. Clean extracted text ───────────────────────────────────────────────────
def clean_page_text(text: str) -> str:
    """Remove line-wrap artifacts while preserving paragraph breaks."""
    text = text.replace("\n\n", "<<PARA>>")
    text = text.replace("\n", " ")
    text = text.replace("<<PARA>>", "\n\n")
    text = re.sub(r' +', ' ', text)
    return text.strip()

cleaned_pages = [(page_num, clean_page_text(text)) for page_num, text in page_texts]
full_text = "\n\n".join(text for _, text in cleaned_pages)

# ERROR HANDLING: empty PDF (scanned/image-only)
if not full_text.strip():
    print("ERROR: No text could be extracted from this PDF.")
    print("This may be a scanned PDF (image-only). OCR is not supported.")
    print("Please use a digital native PDF created from Word, Google Docs, etc.")
    sys.exit(1)

print(f"Extracted {len(full_text)} characters from {page_count} pages.\n")

# ── 4. Chunk the document ─────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)
raw_chunks = splitter.split_text(full_text)
chunks = [c for c in raw_chunks if len(c) >= 100]

# ERROR HANDLING: no usable chunks
if not chunks:
    print("ERROR: Document too short or all chunks below minimum length.")
    sys.exit(1)

print(f"Chunking complete — {len(raw_chunks)} raw → {len(chunks)} after filter.\n")

# ── 5. Map each chunk to its page number ──────────────────────────────────────
def find_page_for_chunk(chunk: str, pages: list) -> int:
    """Find which page a chunk most likely came from using fingerprint matching."""
    fingerprint = chunk[:80].strip()
    for page_num, page_text in pages:
        if fingerprint in page_text:
            return page_num
    return 1

chunk_pages = [find_page_for_chunk(chunk, cleaned_pages) for chunk in chunks]

# ── 6. Load embedding model ───────────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── 7. Embed all chunks ───────────────────────────────────────────────────────
print("Embedding chunks...")
embeddings = model.encode(chunks).tolist()
print(f"Embedded {len(embeddings)} chunks.\n")

# ── 8. Set up Qdrant in-memory collection ─────────────────────────────────────
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="pdf_rag",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
print("Qdrant collection 'pdf_rag' created.\n")

# ── 9. Insert chunks with page numbers in payload ─────────────────────────────
points = [
    PointStruct(
        id=i,
        vector=embeddings[i],
        payload={
            "chunk_text":   chunks[i],
            "chunk_index":  i,
            "chunk_length": len(chunks[i]),
            "page_number":  chunk_pages[i],
        }
    )
    for i in range(len(chunks))
]
client.upsert(collection_name="pdf_rag", points=points)
print(f"Inserted {len(points)} points into Qdrant.\n")

# ── 10. Set up Groq client ────────────────────────────────────────────────────
# ERROR HANDLING: missing API key caught at startup
try:
    groq_client = Groq()
except Exception as e:
    print(f"ERROR: Could not initialise Groq client — {e}")
    print("Make sure GROQ_API_KEY is set in your environment variables.")
    sys.exit(1)

# ── 11. System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a document assistant that helps users find information in PDF documents.
You are concise, professional, and never speculate beyond what is stated.

STRICT RULES:
1. Answer ONLY using the information provided in the <context> block.
2. Do NOT use any knowledge from your training data or outside sources.
3. If the answer is not present in the context, respond with exactly:
   "I don't have enough information to answer this based on the available documents."
4. Be concise and factual. Do not speculate or infer beyond what is stated.
5. Cite the source chunk inline using its label, e.g. "According to [Chunk 1]..."
6. Maximum 3 sentences unless the answer requires a list.
7. If the answer is a list of items, use bullet points."""

# ── 12. Helper: retrieve top-k chunks ─────────────────────────────────────────
def retrieve(query: str, top_k: int = 3) -> list:
    """Embed query and retrieve top_k most similar chunks from Qdrant."""
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name="pdf_rag",
        query=query_vector,
        limit=top_k,
        using=None,
    ).points
    return results

# ── 13. Helper: assemble context with page numbers ────────────────────────────
def assemble_context(results: list) -> str:
    """Build labelled context block with page numbers."""
    parts = []
    for i, result in enumerate(results, start=1):
        chunk_text = result.payload["chunk_text"]
        score = result.score
        page = result.payload["page_number"]
        parts.append(
            f"[Chunk {i} | Page {page}] (similarity: {score:.4f})\n{chunk_text}"
        )
    return "\n\n---\n\n".join(parts)

# ── 14. Helper: build user message ────────────────────────────────────────────
def build_user_message(context: str, question: str) -> str:
    """Wrap context and question in XML-style delimiters."""
    return f"""Answer the question below using ONLY the context provided.
Cite the relevant chunk labels inline in your answer.

<context>
{context}
</context>

<question>
{question}
</question>"""

# ── 15. Helper: generate answer via Groq ──────────────────────────────────────
def generate_answer(context: str, question: str) -> str:
    """Send context + question to Groq LLaMA and return answer.
    
    ERROR HANDLING: API failures caught gracefully — loop continues running.
    User sees a safe error message instead of a crash.
    """
    user_message = build_user_message(context, question)
    try:
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
    except Exception as e:
        # Log the error but don't crash the program
        # The live loop continues and asks for the next question
        print(f"\n[API ERROR] {e}")
        return "Sorry, I couldn't generate an answer right now. Please try again."

# ── 16. Helper: print citations with page numbers ─────────────────────────────
def print_citations(results: list):
    """Print sources section with page numbers after every answer."""
    print("\n── SOURCES ───────────────────────────────────────────────")
    for i, result in enumerate(results, start=1):
        chunk_preview = result.payload["chunk_text"][:100].replace("\n", " ")
        page = result.payload["page_number"]
        score = result.score
        print(f"  [Chunk {i}]  Page: {page}  Score: {score:.4f}")
        print(f"            \"{chunk_preview}...\"")
    print("─" * 65)

# ── 17. Full RAG query function ───────────────────────────────────────────────
def rag_query(question: str):
    """Run full PDF RAG pipeline: retrieve → assemble → generate → cite."""
    print("\n" + "=" * 65)
    print(f"QUESTION: {question}")
    print("=" * 65)

    results = retrieve(question, top_k=3)
    context = assemble_context(results)

    print("\n── ANSWER ────────────────────────────────────────────────\n")
    answer = generate_answer(context, question)
    print(answer)

    print_citations(results)
    print()

# ── 18. Live query loop ───────────────────────────────────────────────────────
pdf_name = os.path.basename(PDF_PATH)
print("=" * 65)
print(f"PDF RAG READY — {pdf_name} ({page_count} pages, {len(chunks)} chunks)")
print("Type your question. Type 'exit' to quit.")
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

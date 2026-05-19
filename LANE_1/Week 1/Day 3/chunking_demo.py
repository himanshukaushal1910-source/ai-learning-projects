import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# Load the NovaMind sample document
# strip() removes leading/trailing whitespace
# ─────────────────────────────────────────────

with open("novamind_sample.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

print("")
print("=" * 60)
print("NovaMind Chunking Demo")
print("=" * 60)
print("Document length: {} characters".format(len(text)))
print("")

# ─────────────────────────────────────────────
# STRATEGY 1 — Fixed size chunking
# Pure Python, no libraries
# Split every 200 characters with 40 char overlap
# Overlap means next chunk starts 160 chars after
# previous chunk started (200 - 40 = 160)
# ─────────────────────────────────────────────

def fixed_size_chunks(text, chunk_size=200, overlap=40):
    chunks = []
    start = 0
    step = chunk_size - overlap  # how far to move forward each time

    while start < len(text):
        # take a slice of chunk_size characters
        end = start + chunk_size
        chunk = text[start:end].strip()

        # only add non-empty chunks
        if chunk:
            chunks.append(chunk)

        # move forward by step (not full chunk_size)
        # this creates the overlap with the next chunk
        start += step

    return chunks

fixed_chunks = fixed_size_chunks(text, chunk_size=200, overlap=40)

# ─────────────────────────────────────────────
# STRATEGY 2 — Sentence chunking
# Split on sentence boundaries using regex
# Merge sentences that are too short (<50 chars)
# into the next sentence for more context
# ─────────────────────────────────────────────

def sentence_chunks(text, min_length=50):
    # split on period, question mark or exclamation
    # followed by whitespace or end of string
    # keep the punctuation with the sentence using lookahead
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # accumulate sentences until we hit min_length
        if current:
            current = current + " " + sentence
        else:
            current = sentence

        # once current chunk is long enough, save it
        if len(current) >= min_length:
            chunks.append(current)
            current = ""

    # add any remaining text as final chunk
    if current:
        chunks.append(current)

    return chunks

sent_chunks = sentence_chunks(text, min_length=50)

# ─────────────────────────────────────────────
# STRATEGY 3 — Recursive chunking via LangChain
# RecursiveCharacterTextSplitter tries to split
# on paragraphs first, then sentences, then words
# falling back to characters only as last resort
# This preserves natural text boundaries
# ─────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    length_function=len,
)

recursive_chunks = splitter.split_text(text)

# ─────────────────────────────────────────────
# Helper — print results for one strategy
# Shows total chunks, average length, first 3
# ─────────────────────────────────────────────

def print_strategy_results(name, chunks):
    avg_length = sum(len(c) for c in chunks) / len(chunks)

    print("-" * 60)
    print("Strategy: {}".format(name))
    print("  Total chunks   : {}".format(len(chunks)))
    print("  Avg chunk length: {:.0f} characters".format(avg_length))
    print("")
    print("  First 3 chunks:")
    for i, chunk in enumerate(chunks[:3]):
        # truncate display to 120 chars for readability
        display = chunk[:120] + "..." if len(chunk) > 120 else chunk
        print("  [{}] {}".format(i + 1, display))
        print("")

# ─────────────────────────────────────────────
# Print results for all three strategies
# ─────────────────────────────────────────────

print_strategy_results("1 -- Fixed size (200 chars, 40 overlap)", fixed_chunks)
print_strategy_results("2 -- Sentence chunking (min 50 chars)",   sent_chunks)
print_strategy_results("3 -- Recursive / LangChain (200, 40)",    recursive_chunks)

# ─────────────────────────────────────────────
# Comparison table — all three side by side
# ─────────────────────────────────────────────

print("=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
print("")
print("{:<12} {:>8} {:>12} {:<30}".format(
    "Strategy", "Chunks", "Avg Length", "Best for"))
print("-" * 60)

strategies = [
    ("Fixed size",  fixed_chunks,     "Quick prototyping"),
    ("Sentence",    sent_chunks,       "Meeting notes, short docs"),
    ("Recursive",   recursive_chunks,  "Long documents, mixed content"),
]

for name, chunks, best_for in strategies:
    avg = sum(len(c) for c in chunks) / len(chunks)
    print("{:<12} {:>8} {:>12.0f} {:<30}".format(
        name, len(chunks), avg, best_for))

print("")
print("=" * 60)
print("KEY OBSERVATION:")
print("  Fixed size  -- fast but cuts mid-sentence")
print("  Sentence    -- clean boundaries, variable length")
print("  Recursive   -- respects paragraphs first, most intelligent")
print("=" * 60)

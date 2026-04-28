import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Load both models once at startup
# MiniLM  — 384 dimensions, fast, lightweight
# mpnet   — 768 dimensions, slower, higher quality
# ─────────────────────────────────────────────

print("")
print("Loading models... (this may take a moment)")
minilm = SentenceTransformer("all-MiniLM-L6-v2")
mpnet  = SentenceTransformer("all-mpnet-base-v2")
print("Both models ready.")
print("")

# ─────────────────────────────────────────────
# Define the 4 sentence pairs to compare
# Pairs 1 and 2 are the interesting ones —
# paraphrases with no shared keywords
# Pairs 3 and 4 are control pairs —
# both models should agree on these
# ─────────────────────────────────────────────

pairs = [
    (
        "We need to reduce server costs",
        "The infrastructure budget is too high"
    ),
    (
        "What did we decide about pricing in Q2?",
        "The team agreed on per-seat billing in the second quarter"
    ),
    (
        "The dog is playing in the park",
        "A dog is running and having fun outside"
    ),
    (
        "I love machine learning",
        "I enjoy eating pasta"
    ),
]

labels = [
    "Pair 1 -- Server costs vs infrastructure budget",
    "Pair 2 -- Pricing Q2 vs per-seat billing",
    "Pair 3 -- Dog playing vs dog running (control)",
    "Pair 4 -- Machine learning vs pasta (control)",
]

# ─────────────────────────────────────────────
# Helper — cosine similarity using raw numpy
# ─────────────────────────────────────────────

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ─────────────────────────────────────────────
# Run comparison across all 4 pairs
# For each pair: encode with both models,
# compute cosine similarity, compare scores
# ─────────────────────────────────────────────

print("=" * 65)
print("MODEL COMPARISON -- all-MiniLM-L6-v2 vs all-mpnet-base-v2")
print("=" * 65)

results = []

for label, (sent_a, sent_b) in zip(labels, pairs):
    # Encode with MiniLM (384 dims)
    minilm_a = minilm.encode(sent_a)
    minilm_b = minilm.encode(sent_b)
    minilm_score = cosine_similarity(minilm_a, minilm_b)

    # Encode with mpnet (768 dims)
    mpnet_a = mpnet.encode(sent_a)
    mpnet_b = mpnet.encode(sent_b)
    mpnet_score = cosine_similarity(mpnet_a, mpnet_b)

    # Determine which model scored higher
    diff = mpnet_score - minilm_score
    if abs(diff) < 0.01:
        winner = "Tied"
    elif diff > 0:
        winner = "mpnet higher"
    else:
        winner = "MiniLM higher"

    results.append((label, sent_a, sent_b, minilm_score, mpnet_score, diff, winner))

# ─────────────────────────────────────────────
# Print results in a clear readable table
# ─────────────────────────────────────────────

for label, sent_a, sent_b, minilm_score, mpnet_score, diff, winner in results:
    print("")
    print(label)
    print("  A: {}".format(sent_a))
    print("  B: {}".format(sent_b))
    print("  " + "-" * 55)
    print("  MiniLM score  (384 dims) : {:.4f}".format(minilm_score))
    print("  mpnet score   (768 dims) : {:.4f}".format(mpnet_score))
    print("  Difference               : {:+.4f}".format(diff))
    print("  Winner                   : {}".format(winner))

# ─────────────────────────────────────────────
# Summary — what the results tell us
# ─────────────────────────────────────────────

print("")
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print("")
print("Pairs 1 and 2 -- paraphrases with no shared keywords")
print("  These reveal quality differences between models")
print("  Higher score = better at understanding paraphrases")
print("")
print("Pairs 3 and 4 -- control pairs")
print("  Pair 3 should score HIGH on both  (similar sentences)")
print("  Pair 4 should score LOW on both   (unrelated sentences)")
print("  If both models agree here -- they are working correctly")
print("")
print("Key insight:")
print("  mpnet has 768 dims vs MiniLM's 384 -- 2x more room")
print("  to encode subtle meaning relationships")
print("  But mpnet is also ~3x slower to run")
print("  Quality vs speed -- always a tradeoff")

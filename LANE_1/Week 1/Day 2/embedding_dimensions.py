import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5 sentences — a mix of related and unrelated topics
sentences = [
    "NovaMind is building a knowledge base for their team.",
    "The pricing model was decided in Q2 — per seat billing.",
    "The engineering roadmap focuses on API performance.",
    "I enjoy eating pasta with tomato sauce.",
    "Machine learning models require large amounts of training data."
]

# Encode all 5 sentences at once
# Output is a numpy array of shape (5, 384)
embeddings = model.encode(sentences)

# ─────────────────────────────────────────────
# 1. Shape of the output
# ─────────────────────────────────────────────

print("=" * 55)
print("1. SHAPE OF THE OUTPUT")
print("=" * 55)
print("Shape: {}".format(embeddings.shape))
print("")
print("What this tells us:")
print("  {} sentences were encoded".format(embeddings.shape[0]))
print("  Each sentence is represented by {} numbers".format(embeddings.shape[1]))
print("  Each row = one sentence vector")
print("  Each column = one dimension of meaning")
print("")

# ─────────────────────────────────────────────
# 2. First 10 numbers of the first embedding
# ─────────────────────────────────────────────

print("=" * 55)
print("2. FIRST 10 NUMBERS OF THE FIRST EMBEDDING")
print("=" * 55)
print("Sentence: '{}'".format(sentences[0]))
print("")
print("First 10 values:")
for i, val in enumerate(embeddings[0][:10]):
    print("  Dimension {:>3} : {:.6f}".format(i+1, val))
print("")
print("What this tells us:")
print("  These are small decimals, positive and negative")
print("  No single number means anything on its own")
print("  Together all 384 numbers locate this sentence")
print("  in meaning-space")
print("")

# ─────────────────────────────────────────────
# 3. Min / max / mean of each embedding
# ─────────────────────────────────────────────

print("=" * 55)
print("3. MIN / MAX / MEAN FOR EACH EMBEDDING")
print("=" * 55)
print("")

for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
    min_val  = np.min(embedding)
    max_val  = np.max(embedding)
    mean_val = np.mean(embedding)
    print("Sentence {}: '{}'".format(i+1, sentence[:50]))
    print("  Min  : {:.4f}".format(min_val))
    print("  Max  : {:.4f}".format(max_val))
    print("  Mean : {:.4f}".format(mean_val))
    print("")

print("=" * 55)
print("WHAT THE STATS TELL US:")
print("=" * 55)
print("")
print("  Min/Max range:")
print("  Values stay in a small range (roughly -1 to 1)")
print("  The model is trained to keep values bounded")
print("  This makes similarity scores stable and comparable")
print("")
print("  Mean close to 0:")
print("  Embeddings are roughly centred around zero")
print("  This is by design -- prevents any one dimension")
print("  from dominating the similarity calculation")
print("")
print("  All 5 sentences have similar ranges:")
print("  Consistent range means embeddings are comparable")
print("  You can safely compute similarity across any pair")

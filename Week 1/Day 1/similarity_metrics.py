import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model — this runs locally, no API key needed
model = SentenceTransformer("all-MiniLM-L6-v2")

pair_1 = (
    "The dog is playing in the park.",
    "A dog is running and having fun outside."
)

pair_2 = (
    "The stock market crashed yesterday.",
    "I enjoy eating pasta with tomato sauce."
)

pair_3 = (
    "Great battery life.",
    "Great battery life. The keyboard is comfortable. The screen is bright. Build quality is excellent. Highly recommended for anyone looking for a reliable laptop."
)

pairs = [pair_1, pair_2, pair_3]
labels = [
    "Pair 1 -- Two very similar sentences",
    "Pair 2 -- Two completely unrelated sentences",
    "Pair 3 -- Short vs long, same meaning",
]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product(a, b):
    return np.dot(a, b)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

for label, (sent_a, sent_b) in zip(labels, pairs):
    vec_a = model.encode(sent_a)
    vec_b = model.encode(sent_b)

    cos  = cosine_similarity(vec_a, vec_b)
    dot  = dot_product(vec_a, vec_b)
    euc  = euclidean_distance(vec_a, vec_b)

    print("")
    print("=" * 55)
    print(label)
    print("  A: " + sent_a[:60])
    print("  B: " + sent_b[:60])
    print("-" * 55)
    print("  Cosine similarity : {:.4f}  (1=identical, 0=unrelated)".format(cos))
    print("  Dot product       : {:.4f}  (higher=more similar)".format(dot))
    print("  Euclidean distance: {:.4f}  (lower=more similar)".format(euc))

print("")
print("=" * 55)
print("WHAT TO NOTICE:")
print("  Pair 1 -- all three metrics agree")
print("  Pair 2 -- all three metrics agree: clearly unrelated")
print("  Pair 3 -- cosine stays high (ignores length)")
print("            euclidean is larger than pair 1 (confused by length)")
print("            THIS is why cosine is the default for text")
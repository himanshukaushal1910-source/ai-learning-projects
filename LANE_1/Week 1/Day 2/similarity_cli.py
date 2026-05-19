import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Load the model ONCE at startup — outside the
# loop. Loading takes ~2 seconds. If this were
# inside the loop it would reload every single
# time the user enters a new pair.
# ─────────────────────────────────────────────

print("")
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.")
print("")
print("=" * 50)
print("  NovaMind Sentence Similarity Tool")
print("=" * 50)

# ─────────────────────────────────────────────
# Helper function — get a non-empty sentence
# from the user. Keeps asking until they type
# something real. Prevents empty input crashes.
# ─────────────────────────────────────────────

def get_sentence(label):
    while True:
        text = input("Enter sentence {}: ".format(label)).strip()
        if text:
            return text
        print("  Please enter a sentence — input cannot be empty.")

# ─────────────────────────────────────────────
# Helper function — compute cosine similarity
# between two numpy vectors using raw numpy.
# Formula: dot(a,b) / (norm(a) * norm(b))
# Returns a float between -1 and 1.
# ─────────────────────────────────────────────

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ─────────────────────────────────────────────
# Helper function — turn a numeric score into
# a human-readable verdict so the user doesn't
# have to interpret raw numbers.
# ─────────────────────────────────────────────

def get_verdict(score):
    if score > 0.85:
        return "Highly similar"
    elif score >= 0.60:
        return "Related"
    elif score >= 0.30:
        return "Loosely related"
    else:
        return "Unrelated"

# ─────────────────────────────────────────────
# Main loop — runs until the user types 'n'.
# Each iteration: get two sentences, encode
# them, compute similarity, print results.
# ─────────────────────────────────────────────

while True:
    print("")

    # Get sentence A — re-prompts if empty
    sentence_a = get_sentence("A")

    # Get sentence B — re-prompts if empty
    sentence_b = get_sentence("B")

    # Encode both sentences into embedding vectors
    # model.encode() returns a numpy array of shape (384,)
    embedding_a = model.encode(sentence_a)
    embedding_b = model.encode(sentence_b)

    # Compute cosine similarity between the two vectors
    score = cosine_similarity(embedding_a, embedding_b)

    # Get the human-readable verdict for this score
    verdict = get_verdict(score)

    # Print the results clearly
    print("")
    print("Results:")
    print("  Cosine similarity : {:.4f}".format(score))
    print("  Verdict           : {}".format(verdict))
    print("")
    print("-" * 50)

    # Ask if the user wants to try another pair
    # Accept 'y' or 'n' only — re-prompt for anything else
    while True:
        again = input("Try another pair? (y/n): ").strip().lower()
        if again in ("y", "n"):
            break
        print("  Please enter y or n.")

    # Exit the main loop if user said no
    if again == "n":
        print("")
        print("Goodbye.")
        break

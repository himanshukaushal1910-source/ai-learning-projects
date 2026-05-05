"""
Cross-encoder diagnostic — Week 3 Day 2
Shows raw cross-encoder scores for query + chunk pairs.
Verifies our prediction: Pair 2 > Pair 3 > Pair 1
"""

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "How does the LLM-7B fine-tuning module work?"

pairs = [
    [query, "The MX-4400 data connector handles custom data source ingestion for enterprise clients. MX-4400 supports REST, GraphQL, and webhook-based data pipelines."],
    [query, "The LLM-7B fine-tuning module allows enterprise customers to fine-tune NovaMind's"],
    [query, "internal language model on their proprietary data. Minimum dataset size: 10,000 samples."],
]

labels = ["Pair 1 — MX-4400 chunk", "Pair 2 — LLM-7B part 1", "Pair 3 — LLM-7B part 2"]

scores = reranker.predict(pairs)

print("\nCross-encoder scores for query:")
print(f'"{query}"\n')
print(f"{'Pair':<30} {'Score':>10}  {'Rank'}")
print("─" * 50)

ranked = sorted(zip(scores, labels), reverse=True)
rank_map = {label: i+1 for i, (score, label) in enumerate(ranked)}

for label, score in zip(labels, scores):
    print(f"{label:<30} {score:>10.4f}  rank {rank_map[label]}")

print("\nRanked order (highest to lowest):")
for i, (score, label) in enumerate(ranked, 1):
    print(f"  {i}. {label} → {score:.4f}")

print("\nPrediction was: Pair 2 > Pair 3 > Pair 1")
actual = [label for score, label in ranked]
prediction = ["Pair 2 — LLM-7B part 1", "Pair 3 — LLM-7B part 2", "Pair 1 — MX-4400 chunk"]
print(f"Actual order:   {' > '.join(l.split('—')[0].strip() for l in actual)}")
print(f"Prediction {'CORRECT ✓' if actual == prediction else 'WRONG ✗ — interesting, see why above'}")

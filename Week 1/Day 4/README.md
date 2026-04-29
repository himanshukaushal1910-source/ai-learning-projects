# NovaMind Semantic Search

> **Project 1 of 5** — Week 1, AI Application Developer Program  
> A complete semantic search system over internal company documents using vector embeddings and Chroma vector database.

---

## What This Does

NovaMind Semantic Search lets you search 20 internal company documents by **meaning** — not just keywords.

Ask *"What was decided about pricing?"* and it finds the pricing decision chunk even if your exact words don't appear in the document. Ask *"What are the engineering priorities?"* and it returns the Q3 roadmap — ranked by semantic similarity.

Built as a foundation for a full RAG (Retrieval-Augmented Generation) system.

---

## How It Works

```
Query text
    ↓
Embed with sentence-transformers (384-dimensional vector)
    ↓
HNSW approximate nearest neighbour search in Chroma
    ↓
Optional metadata filter (by source type)
    ↓
Ranked results with similarity scores
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `sentence-transformers` | Converts text to 384-dim embedding vectors |
| `ChromaDB` | Vector database with HNSW indexing + persistent storage |
| `all-MiniLM-L6-v2` | Embedding model — fast, accurate, 384 dimensions |

---

## Dataset

20 NovaMind internal document chunks across 4 sources:

| Source | Topics covered |
|--------|---------------|
| `meeting_notes` | Pricing decisions, budget allocation, financials, sales pipeline |
| `engineering_spec` | Q3 roadmap, SSO integration, infrastructure migration, API rate limiting |
| `hr_document` | Open roles, hiring process, remote work policy, compensation |
| `product_roadmap` | v2.0 launch, mobile app, AI features, enterprise tier |

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the search system
python novamind_search.py
```

---

## Usage

```
Query > What was decided about pricing?
Query > What are the Q3 engineering priorities?
Query > filter:hr_document Who is NovaMind hiring?
Query > filter:engineering_spec How is SSO being implemented?
Query > quit
```

**Filter syntax:** prefix any query with `filter:<source>` to restrict results to one source type.

**Available sources:** `meeting_notes` · `engineering_spec` · `hr_document` · `product_roadmap`

---

## Example Output

```
Results (filtered to: hr_document)
------------------------------------------------------------
  Rank 1  |  81.3% match  |  distance: 0.1867
  Source : hr_document
  Topic  : open_roles
  Date   : 2024-06-25
  Text   : NovaMind is hiring three roles starting September: a senior backend engineer...
```

---

## Key Concepts Demonstrated

- **Vector embeddings** — text converted to numerical vectors that encode semantic meaning
- **Cosine similarity** — measuring direction-based closeness between vectors
- **HNSW indexing** — approximate nearest neighbour search for fast retrieval at scale
- **Metadata filtering** — combining exact attribute filters with similarity search
- **Persistent vector storage** — embeddings saved to disk, no re-computation on restart

---

## What I Learned Building This

- Why regular databases cannot do similarity search efficiently
- How HNSW navigates a multi-layer graph from coarse to fine
- Why approximate nearest neighbour search is accurate enough in practice
- How metadata filtering restricts the search space before HNSW runs
- The failure modes of semantic search: short queries, domain vocabulary gaps, stale embeddings

---

*Part of a 4-week AI Application Developer program. Next project: PDF Q&A Chatbot using RAG.*

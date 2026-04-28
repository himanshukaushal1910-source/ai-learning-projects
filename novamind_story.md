# The NovaMind Story — Lane 1 Running Narrative
## Your 4-Week AI Engineering Journey told as a Story

---

## The Setup

You just got hired as the first AI engineer at **NovaMind** — a 40-person product startup that builds project management software. They're smart people but they're drowning in their own information.

They have:
- 300+ internal documents — product specs, meeting notes, research papers, decision logs
- A Notion workspace nobody can search properly
- Engineers who spend 2 hours a day just *looking* for things they know exist somewhere

The CEO walks up to you on Day 1 and says:

> *"I need to ask a question like 'what did we decide about the pricing model in Q2?' and get the right answer in seconds. Can you build that?"*

You say yes. This is your 4 weeks.

---

## Week 1 — Learning to Speak in Numbers

---

### Day 1 — Turning NovaMind's Documents into Something a Computer Can Think With

You sit down on your first day. Someone hands you a USB drive with a CSV file — `novamind_docs.csv`. It has 300 rows. Each row is a document excerpt. Some rows are empty. Some have messy formatting. Some are in ALL CAPS.

Before you can do anything intelligent with these documents, you need to clean them. This is your first task.

**Why cleaning matters:**

Imagine you're trying to find two documents that mean the same thing. One says `"  PRICING MODEL DECISION  "` and another says `"pricing model decision"`. To a computer comparing raw text — these look completely different. Extra spaces, different cases — all noise that gets in the way.

Your pandas cleaning script solves exactly this. It:
- Loads the CSV into a DataFrame — a spreadsheet that lives in Python
- Removes rows with no content — empty documents are useless to search over
- Catches the sneaky ones — rows that look non-empty but are just spaces
- Lowercases and strips everything — so meaning can be compared fairly

After cleaning, your 300 documents become a clean Python list — one string per document. This list is what every subsequent step in the NovaMind system will build on.

```python
# This single line is the bridge between raw data and AI
documents = df["description"].tolist()
# ["pricing model decision from q2...", "engineering roadmap for...", ...]
```

---

**Now the real question: how do you make a computer understand what these documents mean?**

You need to convert each document into a vector.

**What is a vector?**

Think of every NovaMind document as a location on a map. Documents about pricing sit in one neighbourhood. Documents about engineering sit in another. Documents about hiring sit somewhere else entirely.

A vector is just the coordinates of that location — a list of numbers that encodes where in meaning-space this document lives.

```python
# A document about pricing might look like this as a vector:
[0.21, -0.45, 0.88, 0.12, ...]  # 384 numbers total
```

No single number means anything on its own. But together, all 384 numbers precisely locate that document in meaning-space.

**Why does this help NovaMind?**

When the CEO asks *"what did we decide about pricing?"* — you convert that question into a vector too. Then you find which document vectors are closest to the question vector. Those are your answers.

This is the entire foundation of NovaMind's knowledge base. Everything else builds on this one idea.

---

**How do you measure which documents are closest?**

You use **cosine similarity** — it measures the angle between two vectors.

Think of each document as an arrow pointing out from the centre of a globe. Two documents about pricing point in nearly the same direction — small angle between them, high similarity score. A pricing document and an engineering document point in very different directions — large angle, low similarity score.

Cosine similarity gives you a number between 0 and 1:
- `0.95` — these documents are about almost the same thing
- `0.4` — loosely related
- `0.05` — completely unrelated

**Why not just use straight-line distance between the points?**

Because NovaMind's documents vary wildly in length. A 2-line meeting note and a 20-page spec document might say the same thing about pricing — but straight-line distance (euclidean) would see them as far apart just because one vector is much longer than the other.

Cosine similarity only cares about direction, not length. The 2-line note and the 20-page spec about the same topic will correctly score as similar.

---

**What you built on Day 1:**

By end of Day 1 you have two working scripts in your NovaMind project folder:

`clean_documents.py` — takes raw messy CSV, produces a clean list of documents ready for the next step

`similarity_check.py` — takes any two pieces of text, converts them to vectors using a local embedding model, and scores how similar they are using cosine similarity, dot product and euclidean distance

You run a quick test. You take two NovaMind document excerpts:
- *"Q2 pricing decision: we will go with a per-seat model"*
- *"The team agreed on per-seat pricing in the second quarter"*

Cosine similarity: `0.91` — the system correctly identifies these as meaning the same thing, even though they use different words.

The CEO walks past. You tell her the first piece is working.

---

**What's still missing:**

You can compare two documents. But NovaMind has 300. You can't compare the CEO's question against every single document one by one — that would be too slow at scale.

You need a way to store all 300 document vectors and search them instantly.

That's Day 2 and beyond.

---

### Day 2 — How Meaning Becomes Numbers

You spent Day 1 treating the embedding model as a black box — text goes in, vector comes out. Today you opened the box. And what's inside changes how you think about everything.

**The three steps that turn a sentence into a vector:**

Every time you call `model.encode("some sentence")` three things happen in sequence:

**Step 1 — Tokenisation**

The sentence gets broken into subword pieces called tokens. Not words — subwords. "NovaMind" becomes `["Nova", "##Mind"]`. "Q2" might become `["Q", "##2"]`.

Two special tokens get added automatically — `[CLS]` at the start and `[SEP]` at the end. The model was trained to expect these markers. They tell it where a sentence begins and ends.

Why subwords and not whole words? Two reasons. First — vocabulary size. There are millions of possible words but only ~30,000 subword pieces. Second — unknown words. The model has never seen "NovaMind" before, but it knows "Nova" and "Mind." Subwords let it handle any word it's never seen.

**Step 2 — Attention**

Each token starts with a generic vector — just its ID looked up in a table. "Bank" has the same starting vector whether you mean a river bank or a financial institution.

Then attention runs. Every token looks at every other token in the sentence and updates its own meaning based on context. After attention, "bank" near "river" has a completely different vector than "bank" near "money." The meaning is now context-aware.

This happens across multiple layers — each layer refining meaning further. By the end, each token holds not just its own meaning but the meaning of the entire sentence's context.

**Step 3 — Pooling**

After attention you have one vector per token. A 9-token sentence gives you 9 vectors — shape `(1, 9, 384)`. But you need one vector for the whole sentence.

Mean pooling takes the average of all token vectors. The result is shape `(384,)` — one vector, 384 numbers, representing the meaning of the entire sentence.

Then normalisation scales that vector to length 1 — same direction, magnitude of exactly 1. This makes dot product and cosine similarity mathematically identical, and makes all vectors comparable regardless of sentence length.

---

**What you discovered about models:**

You compared `all-MiniLM-L6-v2` (384 dims) against `all-mpnet-base-v2` (768 dims) on the same sentence pairs. The results were surprising:

- mpnet scored *higher* on Pair 2 (pricing/billing) — more dimensions captured the paraphrase better
- mpnet scored *lower* on Pair 1 (server costs/infrastructure) — stricter domain separation hurt it here
- Both models agreed on obvious pairs

The lesson: bigger model does not always mean better on every pair. Always benchmark on your actual data before committing to a model.

---

**What you built on Day 2:**

`tokenisation_demo.py` — saw tokenisation happen with real tokens and IDs. Confirmed that "bank" has the same token ID in both sentences — meaning only changes after attention, not during tokenisation.

`embedding_dimensions.py` — saw shape `(5, 384)` with real numbers. Understood that each of the 384 values is small, positive or negative, and only meaningful together.

`mean_pooling_demo.py` — manually reproduced what `model.encode()` does internally. Confirmed the pipeline: 9 token vectors → mean pool → 1 sentence vector → normalise.

`similarity_cli.py` — a real usable tool. Type any two sentences, get a similarity score and verdict instantly. Model loads once at startup.

`model_comparison.py` — compared two models side by side with real data. Learned that model choice depends on your specific domain.

`chroma_preview.py` — first look at vector database search. Three documents stored, one query, correct result returned first with metadata — zero search logic written.

---

**The gap that Day 2 closed:**

On Day 1 you knew *what* a vector was.
On Day 2 you know *how* a sentence becomes one.

The black box is now transparent. Every time you call `model.encode()` for the rest of this program — you'll know exactly what's happening inside.

---

**What's still missing:**

Chroma preview showed you that storing and searching 3 documents is easy. But NovaMind has 300 documents — and real documents aren't single sentences. They're paragraphs, pages, entire PDFs.

You can't embed a 20-page document as one vector — the context window won't allow it, and the meaning would be too diluted. You need to split documents into chunks first.

How you split them determines everything. That's Day 3.

---

## Story Progress Tracker

| Day | Chapter | What NovaMind got |
|-----|---------|-------------------|
| W1D1 | Learning to speak in numbers | Clean documents + similarity scoring |
| W1D2 | How meaning becomes numbers | Full embedding pipeline understood + Chroma preview |
| W1D3 | *How to slice a document* | *Coming tomorrow* |
| W1D4 | | |
| W1D5 | | |
| W2D1 | | |
| W2D2 | | |
| W2D3 | | |
| W2D4 | | |
| W2D5 | | |
| W3D1 | | |
| W3D2 | | |
| W3D3 | | |
| W3D4 | | |
| W3D5 | | |
| W4D1 | | |
| W4D2 | | |
| W4D3 | | |
| W4D4 | | |
| W4D5 | | |

---

## Scripts Built So Far

| Script | What it does |
|--------|-------------|
| `clean_products.py` | Loads CSV, removes empty rows, cleans text, exports clean CSV |
| `similarity_metrics.py` | Computes cosine, dot product and euclidean between sentence pairs |
| `tokenisation_demo.py` | Shows tokenisation with real token IDs |
| `embedding_dimensions.py` | Shows embedding shape and value ranges |
| `mean_pooling_demo.py` | Manually reproduces model.encode() step by step |
| `similarity_cli.py` | CLI tool — enter any two sentences, get similarity score |
| `model_comparison.py` | Compares MiniLM vs mpnet side by side |
| `chroma_preview.py` | Stores 3 docs in Chroma, runs similarity search |

---

*This file grows every day. After each day's review slot a new chapter is added.*
*By end of Week 4 this is the complete story of building NovaMind's knowledge base.*

# novamind_search.py
# NovaMind — Complete Semantic Search System
# Week 1, Day 4 — Portfolio Project 1
#
# Pipeline:
#   20 NovaMind chunks → embed → store in Chroma (persistent)
#   → live search loop with optional source filtering

import os
import chromadb
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# STEP 1 — Load embedding model
# all-MiniLM-L6-v2 produces 384-dimensional vectors
# Same model used in Days 2 and 3 — vectors are compatible
# --------------------------------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# STEP 2 — Create a PERSISTENT Chroma client
# Saves to disk in novamind_chroma_db/ folder
# Unlike Client(), this survives between runs
# Second run: loads existing data, no re-embedding needed
# --------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "novamind_chroma_db")
client = chromadb.PersistentClient(path=DB_PATH)

# --------------------------------------------------
# STEP 3 — Get or create collection with cosine distance
# get_or_create: if collection exists on disk, load it
# if not, create it fresh — handles both first and later runs
# --------------------------------------------------
collection = client.get_or_create_collection(
    name="novamind_docs",
    metadata={"hnsw:space": "cosine"}
)

# --------------------------------------------------
# STEP 4 — Define 20 NovaMind chunks across 4 sources
# 5 chunks per source: meeting_notes, engineering_spec,
# hr_document, product_roadmap
# Each chunk has: source, topic, chunk_index, date
# --------------------------------------------------
chunks = [

    # --- MEETING NOTES (5 chunks) ---
    {
        "id": "mn_0",
        "text": "The leadership team agreed on per-seat pricing after reviewing competitor models. Flat rate was rejected due to poor scalability for enterprise clients.",
        "source": "meeting_notes", "topic": "pricing_decision",
        "chunk_index": 0, "date": "2024-04-10"
    },
    {
        "id": "mn_1",
        "text": "Q2 revenue exceeded targets by 12%. The board approved additional budget for engineering headcount in Q3 to accelerate product delivery.",
        "source": "meeting_notes", "topic": "financials",
        "chunk_index": 1, "date": "2024-04-22"
    },
    {
        "id": "mn_2",
        "text": "The sales team reported 3 enterprise deals closing in May. All three clients requested SSO integration as a hard requirement before signing.",
        "source": "meeting_notes", "topic": "sales_pipeline",
        "chunk_index": 2, "date": "2024-05-06"
    },
    {
        "id": "mn_3",
        "text": "Marketing proposed a referral program offering 20% discount for successful referrals. Decision deferred pending legal review of discount structures.",
        "source": "meeting_notes", "topic": "marketing_strategy",
        "chunk_index": 3, "date": "2024-05-20"
    },
    {
        "id": "mn_4",
        "text": "The Q3 budget was finalised at 2.4M. Engineering receives 60%, marketing 25%, and operations 15%. No further reallocation until Q4 review.",
        "source": "meeting_notes", "topic": "budget_allocation",
        "chunk_index": 4, "date": "2024-06-03"
    },

    # --- ENGINEERING SPEC (5 chunks) ---
    {
        "id": "es_0",
        "text": "Q3 engineering priorities include reducing API latency below 200ms, implementing Redis caching, and completing the Slack integration for NovaMind alerts.",
        "source": "engineering_spec", "topic": "q3_roadmap",
        "chunk_index": 5, "date": "2024-06-10"
    },
    {
        "id": "es_1",
        "text": "The infrastructure team will migrate from AWS EC2 to containerised workloads using Docker and Kubernetes to improve deployment reliability and reduce costs.",
        "source": "engineering_spec", "topic": "infrastructure_migration",
        "chunk_index": 6, "date": "2024-06-18"
    },
    {
        "id": "es_2",
        "text": "SSO integration will use SAML 2.0 and OAuth 2.0. Implementation is scoped for 6 weeks. Auth0 selected as the identity provider after evaluating Okta and Cognito.",
        "source": "engineering_spec", "topic": "sso_integration",
        "chunk_index": 7, "date": "2024-07-01"
    },
    {
        "id": "es_3",
        "text": "Database sharding strategy approved for Q3. User data will be partitioned by region to comply with GDPR and reduce cross-region latency by an estimated 40%.",
        "source": "engineering_spec", "topic": "database_architecture",
        "chunk_index": 8, "date": "2024-07-15"
    },
    {
        "id": "es_4",
        "text": "API rate limiting will be enforced at 1000 requests per minute per tenant. Burst allowance of 1500 for 30 seconds. Limits configurable per enterprise tier.",
        "source": "engineering_spec", "topic": "api_rate_limiting",
        "chunk_index": 9, "date": "2024-07-22"
    },

    # --- HR DOCUMENT (5 chunks) ---
    {
        "id": "hr_0",
        "text": "NovaMind is hiring three roles starting September: a senior backend engineer, a machine learning engineer, and a product designer. 47 applications received so far.",
        "source": "hr_document", "topic": "open_roles",
        "chunk_index": 10, "date": "2024-06-25"
    },
    {
        "id": "hr_1",
        "text": "All engineering candidates must complete a 3-stage interview: technical screen, system design round, and culture fit. Target time-to-hire is 4 weeks from first contact.",
        "source": "hr_document", "topic": "hiring_process",
        "chunk_index": 11, "date": "2024-07-02"
    },
    {
        "id": "hr_2",
        "text": "Remote work policy updated for Q3. Employees may work fully remote with optional office access. Monthly team meetups in Bangalore office are mandatory.",
        "source": "hr_document", "topic": "remote_work_policy",
        "chunk_index": 12, "date": "2024-07-08"
    },
    {
        "id": "hr_3",
        "text": "Performance review cycle moves from annual to bi-annual starting Q3 2024. Managers must submit structured feedback using the new GROW framework template.",
        "source": "hr_document", "topic": "performance_reviews",
        "chunk_index": 13, "date": "2024-07-14"
    },
    {
        "id": "hr_4",
        "text": "Compensation bands revised upward by 8% across all engineering levels to stay competitive with market. Changes effective from August 1st payroll cycle.",
        "source": "hr_document", "topic": "compensation",
        "chunk_index": 14, "date": "2024-07-20"
    },

    # --- PRODUCT ROADMAP (5 chunks) ---
    {
        "id": "pr_0",
        "text": "NovaMind v2.0 launches in September. Key features: real-time collaboration, AI-powered document summarisation, and a redesigned dashboard with customisable widgets.",
        "source": "product_roadmap", "topic": "v2_launch",
        "chunk_index": 15, "date": "2024-06-12"
    },
    {
        "id": "pr_1",
        "text": "Mobile app development begins Q3. iOS first, Android in Q4. Core features: search, notifications, and document preview. Full feature parity targeted for Q1 2025.",
        "source": "product_roadmap", "topic": "mobile_app",
        "chunk_index": 16, "date": "2024-06-20"
    },
    {
        "id": "pr_2",
        "text": "AI summarisation feature will use GPT-4 API with a fallback to Claude. Summaries capped at 150 words. User feedback loop included for quality improvement.",
        "source": "product_roadmap", "topic": "ai_features",
        "chunk_index": 17, "date": "2024-07-05"
    },
    {
        "id": "pr_3",
        "text": "Analytics dashboard v2 will include team productivity metrics, document engagement heatmaps, and exportable reports in CSV and PDF format.",
        "source": "product_roadmap", "topic": "analytics",
        "chunk_index": 18, "date": "2024-07-18"
    },
    {
        "id": "pr_4",
        "text": "Enterprise tier will include dedicated account managers, custom SLA guarantees of 99.99% uptime, and priority support with 2-hour response time.",
        "source": "product_roadmap", "topic": "enterprise_tier",
        "chunk_index": 19, "date": "2024-07-25"
    },
]

# --------------------------------------------------
# STEP 5 — Check if collection already has data
# If chunks already exist from a previous run, skip ingestion
# This is what makes persistence useful — no re-embedding
# --------------------------------------------------
existing_count = collection.count()

if existing_count == 0:
    print(f"Ingesting {len(chunks)} NovaMind chunks into Chroma...")

    # Embed all chunk texts in one batch — faster than one at a time
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Add everything to Chroma in one call
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "source": c["source"],
            "topic": c["topic"],
            "chunk_index": c["chunk_index"],
            "date": c["date"]
        } for c in chunks],
        ids=[c["id"] for c in chunks]
    )
    print(f"Done. {len(chunks)} chunks stored in {DB_PATH}\n")

else:
    # Data already on disk — load instantly, no re-embedding
    print(f"Loaded {existing_count} existing chunks from {DB_PATH}\n")

# --------------------------------------------------
# STEP 6 — Live search loop
# User types a query — sees top 3 ranked results
# Optional: prefix query with 'filter:source_name'
# to restrict results to one source only
# Type 'quit' to exit
#
# Filter syntax examples:
#   filter:meeting_notes what was the budget decision?
#   filter:engineering_spec how is SSO being implemented?
#   filter:hr_document what are the open roles?
#   filter:product_roadmap when does the mobile app launch?
# --------------------------------------------------

print("=" * 60)
print("  NovaMind Semantic Search")
print("=" * 60)
print("  Type a query to search NovaMind documents.")
print("  Optional: prefix with filter:<source> to narrow results.")
print("  Sources: meeting_notes | engineering_spec | hr_document | product_roadmap")
print("  Type 'quit' to exit.")
print("=" * 60)

VALID_SOURCES = ["meeting_notes", "engineering_spec", "hr_document", "product_roadmap"]

while True:

    # Get user input
    user_input = input("\nQuery > ").strip()

    # Exit condition
    if user_input.lower() == "quit":
        print("Exiting NovaMind Search. Goodbye.")
        break

    # Skip empty input
    if not user_input:
        print("Please enter a query.")
        continue

    # --------------------------------------------------
    # Parse optional filter prefix
    # Format: filter:source_name actual query text
    # Example: filter:meeting_notes Q3 budget decision
    # --------------------------------------------------
    active_filter = None
    query_text = user_input

    if user_input.lower().startswith("filter:"):
        parts = user_input.split(" ", 1)
        source_tag = parts[0].replace("filter:", "").strip()

        if source_tag not in VALID_SOURCES:
            print(f"Unknown source '{source_tag}'. Valid: {', '.join(VALID_SOURCES)}")
            continue

        if len(parts) < 2 or not parts[1].strip():
            print("Please add a query after the filter. Example: filter:hr_document open roles")
            continue

        active_filter = source_tag
        query_text = parts[1].strip()

    # --------------------------------------------------
    # Embed the query using the same model as the documents
    # Critical: must use identical model — same vector space
    # --------------------------------------------------
    query_embedding = model.encode([query_text]).tolist()

    # --------------------------------------------------
    # Run the Chroma query
    # where= applies metadata filter if user specified one
    # n_results=3 returns top 3 closest chunks
    # --------------------------------------------------
    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": 3,
        "include": ["documents", "metadatas", "distances"]
    }

    if active_filter:
        query_kwargs["where"] = {"source": active_filter}

    results = collection.query(**query_kwargs)

    # --------------------------------------------------
    # Display results cleanly
    # Distance: lower = more similar (cosine distance)
    # 0.0 = identical, 1.0 = completely unrelated
    # --------------------------------------------------
    print()
    if active_filter:
        print(f"Results (filtered to: {active_filter})")
    else:
        print("Results (all sources)")
    print("-" * 60)

    ids = results["ids"][0]

    if not ids:
        print("No results found.")
        continue

    for i in range(len(ids)):
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]
        dist = results["distances"][0][i]

        # Similarity score — converts cosine distance to
        # a 0-100% scale that's easier to read
        similarity = round((1 - dist) * 100, 1)

        print(f"  Rank {i + 1}  |  {similarity}% match  |  distance: {dist:.4f}")
        print(f"  Source : {meta['source']}")
        print(f"  Topic  : {meta['topic']}")
        print(f"  Date   : {meta['date']}")
        print(f"  Text   : {text[:100]}...")
        print()

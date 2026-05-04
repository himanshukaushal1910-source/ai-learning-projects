# openai_first_call.py
# Week 2, Day 2 — Slot 2: Tool
# Your first OpenAI API call — raw, no frameworks, every line commented

# ── Imports ───────────────────────────────────────────────────────────────────
import os                          # to read environment variables safely
from openai import OpenAI          # the official OpenAI Python client

# ── 1. Create the client ──────────────────────────────────────────────────────
# OpenAI() reads your API key from the environment variable OPENAI_API_KEY.
# Never hardcode your API key in code — it would get committed to GitHub.
# Set it in your terminal first:
#   Windows: set OPENAI_API_KEY=sk-your-key-here
#   Then run this script in the same terminal session.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 2. Build the messages list ────────────────────────────────────────────────
# The API expects a conversation — not a single string.
# Each message has a "role" and "content".
#
# role: "system"    → instructions to the model — sets behaviour and rules
# role: "user"      → the human's input — the actual question
# role: "assistant" → the model's previous replies (used for multi-turn chat)
#
# For a single Q&A you only need system + user.
messages = [
    {
        "role": "system",
        # The system prompt is your instruction layer.
        # This is where you control tone, constraints, and behaviour.
        "content": "You are a helpful assistant for NovaMind, an AI productivity company. "
                   "Answer questions clearly and concisely. "
                   "If you don't know the answer, say so — do not make things up."
    },
    {
        "role": "user",
        # The user message is the actual question being asked.
        "content": "What is the difference between a vector database and a traditional database?"
    }
]

# ── 3. Call the API ───────────────────────────────────────────────────────────
# client.chat.completions.create() sends your messages to the model.
#
# model="gpt-4o-mini"  → cheapest capable OpenAI model, perfect for learning
# messages=messages    → the full conversation you built above
# max_tokens=500       → hard cap on response length (prevents runaway costs)
# temperature=0        → deterministic output — same question = same answer
#                         use 0 for factual tasks, higher for creative tasks
print("Calling OpenAI API...\n")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_tokens=500,
    temperature=0,
)

# ── 4. Print the FULL response object ─────────────────────────────────────────
# Read this carefully before looking at just the content.
# The response contains far more than just the text:
#   - id             → unique ID for this API call
#   - model          → exact model version used
#   - choices        → list of responses (usually 1)
#   - usage          → token counts — this is what you're billed for
#   - finish_reason  → why the model stopped ("stop" = natural end, "length" = hit max_tokens)
print("=" * 65)
print("FULL RESPONSE OBJECT:")
print("=" * 65)
print(response)
print()

# ── 5. Extract just the text content ─────────────────────────────────────────
# response.choices    → list of response options (n=1 by default, so index 0)
# .message            → the assistant's message object
# .content            → the actual text string you care about
answer = response.choices[0].message.content

print("=" * 65)
print("EXTRACTED ANSWER:")
print("=" * 65)
print(answer)
print()

# ── 6. Print token usage ──────────────────────────────────────────────────────
# This tells you exactly what you were billed for.
# prompt_tokens     → tokens in your system + user messages
# completion_tokens → tokens in the model's response
# total_tokens      → sum — this is your cost unit
print("=" * 65)
print("TOKEN USAGE:")
print("=" * 65)
print(f"  Prompt tokens     : {response.usage.prompt_tokens}")
print(f"  Completion tokens : {response.usage.completion_tokens}")
print(f"  Total tokens      : {response.usage.total_tokens}")
print(f"  Approx cost       : ${response.usage.total_tokens * 0.00000015:.6f} USD")
print()

# ── 7. Print finish reason ────────────────────────────────────────────────────
# "stop"   → model finished naturally — good
# "length" → model hit max_tokens before finishing — increase max_tokens if needed
print(f"Finish reason: {response.choices[0].finish_reason}")

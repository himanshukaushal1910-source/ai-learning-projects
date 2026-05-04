# groq_first_call.py
# Week 2, Day 3 — Slot 2: Tool
# Complete Groq API call — every line commented

# ── Imports ───────────────────────────────────────────────────────────────────
import os                  # to read environment variables
from groq import Groq      # official Groq Python client

# ── 1. Create the client ──────────────────────────────────────────────────────
# Groq() with no arguments automatically reads GROQ_API_KEY from environment.
# You set this in Windows Environment Variables — no hardcoding needed.
# OpenAI equivalent: OpenAI() — identical pattern, just different class name.
client = Groq()

# ── 2. Build the messages list ────────────────────────────────────────────────
# Groq uses the exact same message structure as OpenAI.
# A conversation is a list of dicts, each with "role" and "content".
#
# role: "system"    → your instructions — sets model behaviour and constraints
# role: "user"      → the human's question or input
# role: "assistant" → model's previous replies (used for multi-turn chat)
#
# For single Q&A: system + user is all you need.
messages = [
    {
        "role": "system",
        # System prompt — tells the model who it is and how to behave.
        # This is your instruction layer. Be specific — vague instructions
        # produce vague behaviour.
        "content": (
            "You are a helpful assistant for NovaMind, an AI productivity company. "
            "Answer questions clearly and concisely. "
            "If you don't know the answer, say so — do not make things up."
        )
    },
    {
        "role": "user",
        # User message — the actual question being asked.
        "content": "What is the difference between a vector database and a traditional database?"
    }
]

# ── 3. Call the API ───────────────────────────────────────────────────────────
# client.chat.completions.create() sends messages to the model.
# This is identical to OpenAI's client.chat.completions.create().
#
# model="llama-3.3-70b-versatile" → Meta's LLaMA 3.3 70B, free on Groq
# temperature=0                   → deterministic output, no creativity
#                                   same question always = same answer
# max_tokens=500                  → hard cap on response length
print("Calling Groq API...\n")
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    temperature=0,
    max_tokens=500,
)

# ── 4. Print the FULL response object ─────────────────────────────────────────
# Read this carefully — the full object contains much more than just the text.
# Key fields to notice:
#   id            → unique ID for this API call
#   model         → exact model version used
#   choices       → list of responses (usually just 1)
#   usage         → token counts — this is what you're billed for
#   finish_reason → why the model stopped generating
print("=" * 65)
print("FULL RESPONSE OBJECT:")
print("=" * 65)
print(response)
print()

# ── 5. Extract just the answer text ──────────────────────────────────────────
# response.choices       → list of response options (n=1 by default)
# [0]                    → first (and only) response
# .message               → the assistant message object
# .content               → the actual text string — this is your answer
answer = response.choices[0].message.content

print("=" * 65)
print("EXTRACTED ANSWER:")
print("=" * 65)
print(answer)
print()

# ── 6. Print token usage ──────────────────────────────────────────────────────
# response.usage.prompt_tokens     → tokens in your system + user messages
# response.usage.completion_tokens → tokens in the model's response
# response.usage.total_tokens      → sum — this is your cost unit
# On Groq free tier: cost = $0 — but tracking tokens is good habit
print("=" * 65)
print("TOKEN USAGE:")
print("=" * 65)
print(f"  Prompt tokens     : {response.usage.prompt_tokens}")
print(f"  Completion tokens : {response.usage.completion_tokens}")
print(f"  Total tokens      : {response.usage.total_tokens}")
print()

# ── 7. Print finish reason ────────────────────────────────────────────────────
# "stop"   → model finished naturally — this is what you want
# "length" → model hit max_tokens before finishing
#             if you see this, increase max_tokens
finish_reason = response.choices[0].finish_reason
print(f"Finish reason: {finish_reason}")

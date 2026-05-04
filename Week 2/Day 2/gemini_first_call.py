# gemini_first_call.py
# Week 2, Day 2 — Slot 2: Tool
# Your first Gemini API call — updated to use new google-genai library

# ── Imports ───────────────────────────────────────────────────────────────────
import os
from google import genai
from google.genai import types

# ── 1. Create the client ──────────────────────────────────────────────────────
# Hardcoded key for now — we'll move to env variables once everything works.
# Never commit API keys to GitHub — keep this file in .gitignore.
API_KEY = "YOUR_KEY_HERE" # never commit your actual key to GitHub!
client = genai.Client(api_key=API_KEY)

# ── 2. Define system instruction and user message ─────────────────────────────
# system_instruction → your rules and behaviour for the model
# user_message       → the actual question being asked
system_instruction = (
    "You are a helpful assistant for NovaMind, an AI productivity company. "
    "Answer questions clearly and concisely. "
    "If you don't know the answer, say so — do not make things up."
)

user_message = "What is the difference between a vector database and a traditional database?"

# ── 3. Call the API ───────────────────────────────────────────────────────────
# gemini-2.0-flash → current free tier model, fast and capable
# temperature=0    → deterministic, factual output
# max_output_tokens=500 → hard cap on response length
print("Calling Gemini API...\n")
response = client.models.generate_content(
    model="gemini-1.5-flash-8b",
    contents=user_message,
    config=types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=500,
        system_instruction=system_instruction,
    ),
)

# ── 4. Print the FULL response object ─────────────────────────────────────────
# Read this before extracting just the text.
# Notice: candidates, usage_metadata, finish_reason inside the object.
print("=" * 65)
print("FULL RESPONSE OBJECT:")
print("=" * 65)
print(response)
print()

# ── 5. Extract just the text content ─────────────────────────────────────────
# response.text → shortcut to the answer string
# OpenAI equivalent: response.choices[0].message.content
answer = response.text

print("=" * 65)
print("EXTRACTED ANSWER:")
print("=" * 65)
print(answer)
print()

# ── 6. Print token usage ──────────────────────────────────────────────────────
# prompt_token_count     → tokens you sent (system + user)
# candidates_token_count → tokens in the model's response
# total_token_count      → sum of both
print("=" * 65)
print("TOKEN USAGE:")
print("=" * 65)
print(f"  Prompt tokens     : {response.usage_metadata.prompt_token_count}")
print(f"  Completion tokens : {response.usage_metadata.candidates_token_count}")
print(f"  Total tokens      : {response.usage_metadata.total_token_count}")
print()

# ── 7. Print finish reason ────────────────────────────────────────────────────
# STOP      → model finished naturally — good
# MAX_TOKENS → hit the token limit — increase max_output_tokens if needed
print(f"Finish reason: {response.candidates[0].finish_reason}")

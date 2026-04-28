import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Setup — load both the raw model and the
# sentence-transformers wrapper
# ─────────────────────────────────────────────

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Raw transformer model — gives us token-level embeddings
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
raw_model  = AutoModel.from_pretrained(MODEL_NAME)

# Sentence-transformers wrapper — hides pooling internally
st_model = SentenceTransformer("all-MiniLM-L6-v2")

SENTENCE = "NovaMind is building a knowledge base"

print("=" * 55)
print("Sentence: '{}'".format(SENTENCE))
print("=" * 55)
print("")

# ─────────────────────────────────────────────
# STEP 1 — Tokenise the sentence
# ─────────────────────────────────────────────

print("STEP 1 -- Tokenise")
print("-" * 55)

# encoded_input is a dict with input_ids, attention_mask etc
encoded_input = tokenizer(
    SENTENCE,
    return_tensors="pt",      # return PyTorch tensors
    padding=True,
    truncation=True
)

tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
print("Tokens : {}".format(tokens))
print("Count  : {} tokens (including [CLS] and [SEP])".format(len(tokens)))
print("")

# ─────────────────────────────────────────────
# STEP 2 — Get raw token embeddings (before pooling)
# ─────────────────────────────────────────────

print("STEP 2 -- Raw token embeddings (before pooling)")
print("-" * 55)

with torch.no_grad():   # no gradients needed — we are not training
    model_output = raw_model(**encoded_input)

# last_hidden_state contains one vector per token
# shape: (batch_size, num_tokens, hidden_size)
# = (1 sentence, N tokens, 384 dimensions)
token_embeddings = model_output.last_hidden_state

print("Shape of raw token embeddings : {}".format(tuple(token_embeddings.shape)))
print("  Dimension 0 = batch size     : {} sentence".format(token_embeddings.shape[0]))
print("  Dimension 1 = num tokens     : {} tokens".format(token_embeddings.shape[1]))
print("  Dimension 2 = hidden size    : {} dimensions".format(token_embeddings.shape[2]))
print("")
print("Each of the {} tokens has its own 384-number vector".format(token_embeddings.shape[1]))
print("We need to collapse these into ONE vector")
print("")

# ─────────────────────────────────────────────
# STEP 3 — Manual mean pooling
# ─────────────────────────────────────────────

print("STEP 3 -- Mean pooling manually")
print("-" * 55)

# attention_mask tells us which tokens are real vs padding
# real tokens = 1, padding tokens = 0
# we only want to average over real tokens
attention_mask = encoded_input["attention_mask"]

# expand mask to same shape as token embeddings for multiplication
mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

# sum only the real token vectors (padding gets zeroed out by mask)
sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)

# count real tokens per sentence
sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

# divide to get the mean
manual_pooled = sum_embeddings / sum_mask

# convert to numpy for comparison
manual_embedding = manual_pooled[0].numpy()

print("Shape after mean pooling: {}".format(manual_embedding.shape))
print("First 5 values: {}".format(
    ["  {:.6f}".format(v) for v in manual_embedding[:5]]
))
print("")

# ─────────────────────────────────────────────
# STEP 4 — Compare with model.encode()
# ─────────────────────────────────────────────

print("STEP 4 -- Verify against model.encode()")
print("-" * 55)

# this is what you normally call — it does all steps internally
st_embedding = st_model.encode(SENTENCE)

print("Shape from model.encode() : {}".format(st_embedding.shape))
print("First 5 values: {}".format(
    ["  {:.6f}".format(v) for v in st_embedding[:5]]
))
print("")

# check how close the two results are
# note: sentence-transformers also normalises the vector
# so we normalise our manual result before comparing
manual_normalised = manual_embedding / np.linalg.norm(manual_embedding)
cosine_match = np.dot(manual_normalised, st_embedding) / (
    np.linalg.norm(manual_normalised) * np.linalg.norm(st_embedding)
)

print("Cosine similarity between manual and model.encode():")
print("  {:.8f}".format(cosine_match))
print("")
if cosine_match > 0.9999:
    print("Match: TRUE -- manual pooling produces the same result")
else:
    print("Match: CLOSE -- small difference due to normalisation")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

print("")
print("=" * 55)
print("THE FULL PIPELINE IN NUMBERS:")
print("=" * 55)
print("  Input    : 1 sentence")
print("  Step 1   : {} tokens after tokenisation".format(len(tokens)))
print("  Step 2   : shape {} -- one vector per token".format(
    tuple(token_embeddings.shape)))
print("  Step 3   : shape {} -- one vector for the sentence".format(
    manual_embedding.shape))
print("  Step 4   : model.encode() gives the same result")
print("")
print("THIS is what happens every time you call model.encode()")
print("Tokenise -> transformer -> {} token vectors -> mean pool -> 1 vector".format(
    len(tokens)))

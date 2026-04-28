from transformers import AutoTokenizer

# Load the tokenizer for all-MiniLM-L6-v2
# This is the same tokenizer used inside sentence-transformers
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def show_tokens(sentence):
    # Tokenise the sentence
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)

    print("Sentence : " + sentence)
    print("Tokens   : " + str(tokens))
    print("Full sequence with special tokens:")
    for token, token_id in zip(token_strings, token_ids):
        print("  {:<15} -> {}".format(token, token_id))
    print("Total tokens: {}".format(len(token_ids)))
    print("")

# ─────────────────────────────────────────────
# Three main sentences
# ─────────────────────────────────────────────

print("=" * 55)
print("PART 1 -- THREE SENTENCES")
print("=" * 55)
print("")

show_tokens("NovaMind is building a knowledge base")
show_tokens("The pricing model was decided in Q2")
show_tokens("bank")

# ─────────────────────────────────────────────
# The word 'bank' in two different contexts
# ─────────────────────────────────────────────

print("=" * 55)
print("PART 2 -- THE WORD 'BANK' IN TWO CONTEXTS")
print("=" * 55)
print("")
print("NOTE: The TOKEN for 'bank' will be identical in both")
print("sentences. Tokenisation happens before the model reads")
print("context. The MEANING of bank only changes after the")
print("transformer runs attention across all tokens.")
print("")

show_tokens("I went to the river bank")
show_tokens("I deposited money at the bank")

# ─────────────────────────────────────────────
# Key observations
# ─────────────────────────────────────────────

print("=" * 55)
print("WHAT TO NOTICE:")
print("  [CLS] = special start token added automatically")
print("  [SEP] = special end token added automatically")
print("  ## prefix = this subword connects to the previous token")
print("  'bank' has the SAME token ID in both sentences")
print("  meaning only changes AFTER attention -- not here")
print("=" * 55)

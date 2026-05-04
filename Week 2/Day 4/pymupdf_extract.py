# pymupdf_extract.py
# Week 2, Day 4 — Slot 2: Tool
# Extract text from a PDF page by page using PyMuPDF
# PyMuPDF imports as 'fitz' — historical name from the MuPDF library it wraps

# ── Imports ───────────────────────────────────────────────────────────────────
import fitz   # PyMuPDF — imports as fitz, not pymupdf
import os     # for path handling

# ── 1. Set the PDF path ───────────────────────────────────────────────────────
# Update this to point to your PDF file.
# Keep it in the same folder as this script for simplicity.
PDF_PATH = os.path.join(os.path.dirname(__file__), "novamind_sample.pdf")

# ── 2. Open the PDF ───────────────────────────────────────────────────────────
# fitz.open() reads the PDF and returns a Document object.
# The Document object gives you access to all pages and metadata.
# It works like a list — you can iterate over it or index into it.
print(f"Opening PDF: {PDF_PATH}\n")
doc = fitz.open(PDF_PATH)

# ── 3. Print basic document info ──────────────────────────────────────────────
# doc.page_count → total number of pages in the PDF
# doc.metadata   → dict with title, author, creation date, etc.
print(f"Total pages    : {doc.page_count}")
print(f"PDF title      : {doc.metadata.get('title', 'No title')}")
print(f"PDF author     : {doc.metadata.get('author', 'No author')}")
print()

# ── 4. Extract text page by page ──────────────────────────────────────────────
# Iterating over doc gives you one Page object per page.
# page.number    → 0-indexed page number (page 1 = index 0)
# page.get_text() → extracts all text from this page as a plain string
#
# What get_text() does:
#   - Reads character coordinates from the PDF
#   - Reconstructs words and lines based on position
#   - Returns a single string with newlines between lines
#
# What can go wrong:
#   - Extra newlines between every word (common in some PDFs)
#   - Headers/footers mixed into body text
#   - Columns merged incorrectly
#   - Tables turned into unstructured text
#
# Reading the raw output below tells you what your chunker will receive.

print("=" * 65)
print("PAGE-BY-PAGE EXTRACTION")
print("=" * 65)

all_text = ""          # accumulate full document text
total_chars = 0        # track total characters extracted

for page in doc:
    # Extract raw text from this page
    page_text = page.get_text()

    # Accumulate into full document string
    all_text += page_text

    # Count characters on this page
    page_chars = len(page_text)
    total_chars += page_chars

    # Print page summary
    # page.number is 0-indexed — add 1 for human-readable page number
    print(f"\n── Page {page.number + 1} ({'─' * 50})")
    print(f"   Characters on this page : {page_chars}")
    print(f"   First 200 characters    :")
    print()

    # Print first 200 characters — strip leading whitespace for clean display
    # This is your preview of what the chunker will receive from this page
    preview = page_text.strip()[:200]
    print(f"   {repr(preview)}")   # repr() shows \n as literal \n so you can see newlines
    print()
    print(f"   Raw preview (formatted):")
    print(f"   {page_text.strip()[:200]}")   # formatted version

# ── 5. Close the document ─────────────────────────────────────────────────────
# Always close the document when done — releases file handle.


# ── 6. Print extraction summary ───────────────────────────────────────────────
print()
print("=" * 65)
print("EXTRACTION SUMMARY")
print("=" * 65)
print(f"  Total pages     : {doc.page_count}")
print(f"  Total characters: {total_chars}")
print(f"  Average per page: {total_chars // doc.page_count if doc.page_count > 0 else 0}")
print()

# ── 7. Show full extracted text length ───────────────────────────────────────
# This is what gets passed to your chunker in the RAG pipeline.
# Compare this to the original .txt file character count from Day 3 (3528 chars).
# They should be similar for a clean digital native PDF.
print(f"  Full text length: {len(all_text)} characters")
print()
print("  First 500 characters of full extracted text:")
print("  " + "-" * 60)
print(all_text[:500])

doc.close()
# NovaMind RAG API — Dockerfile
# ================================
# This file tells Docker how to build an image for our FastAPI app.
# Docker reads this top to bottom, executing each instruction as a layer.
# Layers are cached — unchanged layers are reused on rebuild, saving time.

# Step 1: Start from official Python 3.11 slim image.
# "slim" = minimal OS, no unnecessary tools, smaller final image.
# We never build from a blank OS — always start from an official base.
FROM python:3.11-slim

# Step 2: Set working directory inside the container.
# All subsequent commands run from /app.
# This is where our code will live inside the container.
WORKDIR /app

# Step 3: Copy requirements FIRST, then install.
# Critical ordering for layer caching:
# - requirements.txt changes rarely
# - Your code changes constantly
# By copying requirements before code, Docker caches the pip install layer.
# Change your code → Docker skips reinstalling packages → fast rebuild.
COPY requirements.txt .

# Step 4: Install dependencies.
# --no-cache-dir keeps the image smaller by not storing pip's download cache.
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the application code.
# This layer changes every time you change code — that's fine, it's fast.
# The dot means: copy everything in the build context (current directory)
# into /app inside the container.
COPY . .

# Step 6: Tell Docker what command to run when a container starts.
# This is NOT executed during build — only when a container launches.
# We use the array form (not string) — more reliable, no shell interpretation.
# reload=False because file watching doesn't work properly inside containers.
CMD ["uvicorn", "novamind_fastapi_docker:app", "--host", "0.0.0.0", "--port", "8000"]

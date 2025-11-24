# Use Python 3.11 slim as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies for FAISS, torch, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Step 1: Install langchain-core first to avoid dependency conflicts
RUN pip install langchain-core==1.1.0

# Step 2: Install rest of the dependencies
RUN pip install -r requirements.txt

# Railway port
ENV PORT 8080

# Start the Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--config", "gunicorn.conf.py"]

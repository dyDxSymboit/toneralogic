# Use Python 3.11 slim as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for FAISS, torch, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy everything in the current folder to /app
COPY . /app

# Upgrade pip and install setuptools/wheel
RUN pip install --upgrade pip setuptools wheel

# Step 1: Install langchain-core first to avoid conflicts
RUN pip install langchain-core==1.0.2

# Step 2: Install the rest of the requirements
RUN pip install -r requirements.txt

# Expose Railway port
ENV PORT 8080

# Start the Flask app with Gunicorn using your config
CMD ["gunicorn", "app:app", "--config", "gunicorn.conf.py"]

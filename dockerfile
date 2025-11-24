# Use Python 3.11 slim as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files first (to speed up caching)
COPY . /app

# Upgrade pip and build tools
RUN pip install --upgrade pip setuptools wheel

# ------------------------------------------
# 🔥 Install langchain-core FIRST to avoid version conflicts
# ------------------------------------------
RUN pip install langchain-core==1.1.0

# ------------------------------------------
# 🔥 Install all other dependencies
# ------------------------------------------
RUN pip install -r requirements.txt

# Expose Railway port
ENV PORT=8080

# Start the Flask app using Gunicorn (PRODUCTION)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

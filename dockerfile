FROM python:3.11-slim

WORKDIR /app

# System dependencies for FAISS and Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install exact versions used in Colab
RUN pip install faiss-cpu==1.13.0 \
    Flask==3.1.2 \
    huggingface-hub==0.36.0 \
    langchain==0.3.27 \
    langchain-classic==1.0.0 \
    langchain-community==0.4.1 \
    langchain-core==1.1.0 \
    langchain-huggingface==0.3.0 \
    langchain-text-splitters==1.0.0 \
    numpy==2.0.2 \
    python-dotenv==1.2.1 \
    requests==2.32.5 \
    requests-oauthlib==2.0.0 \
    requests-toolbelt==1.0.0 \
    scikit-learn==1.6.1 \
    scipy==1.16.3 \
    sentence-transformers==5.1.2 \
    torch==2.9.0\
    torchao==0.10.0 \
    torchaudio==2.9.0\
    torchdata==0.11.0 \
    torchsummary==1.5.1 \
    torchtune==0.6.1 \
    torchvision==0.24.0 \
    transformers==4.57.1 \
    Werkzeug==3.1.3

# Railway port
ENV PORT 8080

# Start Flask with Gunicorn
CMD ["gunicorn", "app:app", "--config", "gunicorn.conf.py"]

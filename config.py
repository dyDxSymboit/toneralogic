import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_URL = os.getenv("COHERE_URL")
VECTORSTORE_DIR = os.getenv(r"VECTORSTORE_DIR")
EMB_MODEL = "BAAI/bge-base-en-v1.5"

# Input Limits
MAX_WORDS = int(os.getenv("MAX_WORDS", 250))
MAX_CHARS = int(os.getenv("MAX_CHARS", 2000))
MIN_CHARS = int(os.getenv("MIN_CHARS", 3))

# Rate Limiting
PROMPTS_PER_WINDOW = 15
WINDOW_SECONDS = 43200

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Cache settings (6 hours by default)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 24 * 60 * 60))
CACHE_PREFIX = os.getenv("CACHE_PREFIX", "cache:answer")
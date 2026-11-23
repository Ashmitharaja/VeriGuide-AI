"""All tunable settings in one place. Override any of them with environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.environ.get("VG_DOCS_DIR", ROOT / "data" / "docs"))
STORE_DIR = Path(os.environ.get("VG_STORE_DIR", ROOT / "data" / "store"))
INDEX_PATH = STORE_DIR / "index.faiss"
DB_PATH = STORE_DIR / "veriguide.db"

# 2. Document processing
EMBED_MODEL = os.environ.get("VG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_WORDS = int(os.environ.get("VG_CHUNK_WORDS", 120))
CHUNK_OVERLAP = int(os.environ.get("VG_CHUNK_OVERLAP", 30))

# 4. Retrieval
TOP_K = int(os.environ.get("VG_TOP_K", 4))
# Below this cosine similarity, the best passage is treated as "not about the question".
MIN_RETRIEVAL_SCORE = float(os.environ.get("VG_MIN_RETRIEVAL_SCORE", 0.35))

# 5. Verification: a claim is "supported" if some retrieved sentence is this similar.
CLAIM_SUPPORT_THRESHOLD = float(os.environ.get("VG_CLAIM_THRESHOLD", 0.55))

# LLM (any OpenAI-compatible endpoint). Default: Google Gemini.
# With no API key, VeriGuide falls back to an offline extractive answerer.
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash")

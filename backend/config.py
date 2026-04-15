"""
Centralized configuration. Every tunable value lives here.
Loaded from .env at import time — no hardcoded paths or keys anywhere else.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
QA_CSV_DIR = Path(os.getenv("QA_CSV_DIR", BASE_DIR / "data" / "csvs"))
CHUNK_PKL_DIR = Path(os.getenv("CHUNK_PKL_DIR", BASE_DIR / "data" / "chunks"))
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", BASE_DIR / "data" / "indices"))

# ── DeepSeek LLM ─────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── Embedding ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── RAG tuning ───────────────────────────────────────────────────────────
QA_MATCH_THRESHOLD = float(os.getenv("QA_MATCH_THRESHOLD", "0.82"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))
CHUNK_TOP_K = int(os.getenv("CHUNK_TOP_K", "4"))

# ── Stock / Yahoo Finance ────────────────────────────────────────────────
YF_CACHE_TTL_SECONDS = int(os.getenv("YF_CACHE_TTL_SECONDS", "300"))
YF_MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "3"))

# ── Server ───────────────────────────────────────────────────────────────
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
]
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
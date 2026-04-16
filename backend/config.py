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
YF_MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "3"))
YF_COOLDOWN_SECONDS = float(os.getenv("YF_COOLDOWN_SECONDS", "2.0"))
YF_RETRY_BASE_SECONDS = int(os.getenv("YF_RETRY_BASE_SECONDS", "15"))
YF_RATE_LIMIT_WAIT = int(os.getenv("YF_RATE_LIMIT_WAIT", "75"))
YF_QUOTE_DELAY_MS = int(os.getenv("YF_QUOTE_DELAY_MS", "250"))

# TTLs (seconds)
YF_TTL_QUOTES = int(os.getenv("YF_TTL_QUOTES", "600"))           # 10 min
YF_TTL_PRICE = int(os.getenv("YF_TTL_PRICE", "21600"))           # 6 hours
YF_TTL_FUNDAMENTALS = int(os.getenv("YF_TTL_FUNDAMENTALS", "86400"))  # 24 hours
YF_TTL_INFO = int(os.getenv("YF_TTL_INFO", "21600"))             # 6 hours

# ── Twelve Data fallback (replaces dead FMP) ─────────────────────────────
# Free tier: 800 req/day, 8 req/min. Get key: https://twelvedata.com/register
TD_API_KEY = os.getenv("TD_API_KEY", "")
TD_DAILY_BUDGET = int(os.getenv("TD_DAILY_BUDGET", "800"))
TD_RESERVE_BUDGET = int(os.getenv("TD_RESERVE_BUDGET", "50"))  # keep for quotes

# User agent rotation pool
YF_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]

# ── Server ───────────────────────────────────────────────────────────────
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
]
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
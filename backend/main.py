"""
Buffett Bureaucracy — FastAPI application.

Startup:
  1. Load config from .env
  2. Initialize BuffettAgent (loads/builds FAISS indices)
  3. Mount stock and chat routes
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import CORS_ORIGINS, HOST, PORT, LOG_LEVEL
from rag import BuffettAgent
from stock import stock_router

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Global agent (initialized at startup) ────────────────────────────────

agent: Optional[BuffettAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG agent once at startup, release at shutdown."""
    global agent
    logger.info("Starting BuffettAgent initialization …")
    agent = BuffettAgent()
    logger.info("Agent ready: %s", agent.index_stats)
    yield
    logger.info("Shutting down.")


# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Buffett Bureaucracy API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ───────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    strategy: str
    reason: str
    confidence: str
    tools_called: list[dict]
    sources: list[dict]
    duration_ms: int


# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    stats = agent.index_stats if agent else {}
    return {
        "service": "Buffett Bureaucracy API",
        "version": "2.0.0",
        "indices": stats,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    history = [{"role": m.role, "content": m.content} for m in request.history]
    result = agent.answer(request.query, history=history)
    return result.to_dict()


@app.post("/admin/rebuild-indices")
def rebuild_indices():
    """Force rebuild of FAISS indices from source data."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    stats = agent.rebuild_indices()
    return {"status": "rebuilt", "stats": stats}


@app.get("/admin/index-stats")
def index_stats():
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent.index_stats


# ── Mount stock routes ───────────────────────────────────────────────────

app.include_router(stock_router, prefix="/stock", tags=["Stock Analysis"])

# Also expose quotes at /api/quotes for backward compat with existing frontend
app.include_router(stock_router, prefix="/api", tags=["Compat"], include_in_schema=False)
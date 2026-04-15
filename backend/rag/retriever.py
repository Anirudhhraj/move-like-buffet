"""
Dual-store retriever — searches QA index and chunk index in parallel.
Returns ranked results with metadata and similarity scores.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import RETRIEVAL_TOP_K, CHUNK_TOP_K, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    question: str
    answer: str
    label: str
    sublabel: str
    source: str
    quality: Optional[float]
    similarity: float
    prompt_type: str = ""


@dataclass
class ChunkResult:
    text: str
    label: str
    source_file: str
    source_section: str
    similarity: float
    chunk_id: str = ""


@dataclass
class RetrievalResult:
    qa_results: list[QAResult] = field(default_factory=list)
    chunk_results: list[ChunkResult] = field(default_factory=list)
    best_qa_similarity: float = 0.0


class DualRetriever:
    """Searches both QA and chunk FAISS indices for a query.
    Over-fetches by 3x when label filtering, then trims to top_k."""

    def __init__(
        self,
        qa_index: faiss.Index,
        qa_meta: list[dict],
        chunk_index: faiss.Index,
        chunk_meta: list[dict],
        encoder: Optional[SentenceTransformer] = None,
    ):
        self.qa_index = qa_index
        self.qa_meta = qa_meta
        self.chunk_index = chunk_index
        self.chunk_meta = chunk_meta
        self.encoder = encoder or SentenceTransformer(EMBEDDING_MODEL)

    def search(
        self,
        query: str,
        qa_top_k: int = RETRIEVAL_TOP_K,
        chunk_top_k: int = CHUNK_TOP_K,
        label_filter: Optional[list[str]] = None,
    ) -> RetrievalResult:
        embedding = self.encoder.encode(
            [query], normalize_embeddings=True,
        ).astype(np.float32)

        qa_results = self._search_qa(embedding, qa_top_k, label_filter)
        chunk_results = self._search_chunks(embedding, chunk_top_k, label_filter)
        best_sim = qa_results[0].similarity if qa_results else 0.0

        return RetrievalResult(
            qa_results=qa_results,
            chunk_results=chunk_results,
            best_qa_similarity=best_sim,
        )

    def _search_qa(
        self,
        embedding: np.ndarray,
        top_k: int,
        label_filter: Optional[list[str]],
    ) -> list[QAResult]:
        if self.qa_index is None or self.qa_index.ntotal == 0:
            return []

        fetch_k = top_k * 3 if label_filter else top_k
        fetch_k = min(fetch_k, self.qa_index.ntotal)
        scores, indices = self.qa_index.search(embedding, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.qa_meta):
                continue
            meta = self.qa_meta[idx]
            if label_filter and meta.get("label", "") not in label_filter:
                continue
            results.append(QAResult(
                question=meta["question"],
                answer=meta["answer"],
                label=meta.get("label", ""),
                sublabel=meta.get("sublabel", ""),
                source=meta.get("source", ""),
                quality=meta.get("quality"),
                similarity=round(float(score), 4),
                prompt_type=meta.get("prompt_type", ""),
            ))
            if len(results) >= top_k:
                break
        return results

    def _search_chunks(
        self,
        embedding: np.ndarray,
        top_k: int,
        label_filter: Optional[list[str]],
    ) -> list[ChunkResult]:
        if self.chunk_index is None or self.chunk_index.ntotal == 0:
            return []

        fetch_k = top_k * 3 if label_filter else top_k
        fetch_k = min(fetch_k, self.chunk_index.ntotal)
        scores, indices = self.chunk_index.search(embedding, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_meta):
                continue
            meta = self.chunk_meta[idx]
            if label_filter and meta.get("label", "") not in label_filter:
                continue
            results.append(ChunkResult(
                text=meta["text"],
                label=meta.get("label", ""),
                source_file=meta.get("source_file", ""),
                source_section=meta.get("source_section", ""),
                similarity=round(float(score), 4),
                chunk_id=meta.get("chunk_id", ""),
            ))
            if len(results) >= top_k:
                break
        return results
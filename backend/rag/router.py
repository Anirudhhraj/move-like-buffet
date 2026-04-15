"""
Query router — selects RAG strategy based on retrieval quality and label spread.

Strategies:
  qa_match    — strong curated match, use directly
  synthesis   — combine multiple sources for a focused answer
  multi_label — broad question spanning 3+ topic labels
  reject      — used by agent pre-retrieval gate (not produced here)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retriever import RetrievalResult

from config import QA_MATCH_THRESHOLD

logger = logging.getLogger(__name__)

# Strategy constants (agent.py imports these for prompt selection + metadata)
STRATEGY_QA_MATCH = "qa_match"
STRATEGY_SYNTHESIS = "synthesis"
STRATEGY_MULTI_LABEL = "multi_label"
STRATEGY_REJECT = "reject"


def is_coherent(query: str) -> bool:
    """Fast heuristic to reject gibberish before any LLM call."""
    text = query.strip()
    if len(text) < 2:
        return False
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    if alpha_ratio < 0.5:
        return False
    # Need at least one real word
    words = [w for w in text.split() if len(w) >= 2]
    return len(words) >= 1


def classify_query(query: str, retrieval: RetrievalResult) -> dict:
    """Route to qa_match, synthesis, or multi_label based on retrieval results."""
    best_sim = retrieval.best_qa_similarity
    qa_results = retrieval.qa_results
    chunk_results = retrieval.chunk_results

    # 1. Strong direct QA match
    if best_sim >= QA_MATCH_THRESHOLD and qa_results:
        return {
            "strategy": STRATEGY_QA_MATCH,
            "reason": f"QA match sim={best_sim:.3f} (label: {qa_results[0].label})",
        }

    # 2. Multi-label: only if TOP results (not all results) span 3+ labels
    #    AND similarity is reasonable. Prevents focused questions from
    #    triggering multi-label just because low-ranked noise spans labels.
    if best_sim >= 0.4:
        top_labels = set()
        for r in qa_results[:3]:
            if r.label:
                top_labels.add(r.label)
        for r in chunk_results[:2]:
            if r.label:
                top_labels.add(r.label)
        if len(top_labels) >= 3:
            return {
                "strategy": STRATEGY_MULTI_LABEL,
                "reason": f"Top results span {len(top_labels)} labels ({', '.join(sorted(top_labels))})",
            }

    # 3. Default: synthesis
    return {
        "strategy": STRATEGY_SYNTHESIS,
        "reason": f"Synthesis — best_sim={best_sim:.3f}, {len(qa_results)} QA + {len(chunk_results)} chunks",
    }
"""
BuffettAgent — orchestrates query gating, enrichment, dual-store RAG,
strategy routing, and citation-aware LLM generation.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    EMBEDDING_MODEL, FAISS_INDEX_DIR, QA_CSV_DIR, CHUNK_PKL_DIR,
    RETRIEVAL_TOP_K, CHUNK_TOP_K, QA_MATCH_THRESHOLD,
)
from .indexer import (
    load_qa_index, load_chunk_index,
    load_qa_dataframe, load_chunks,
    build_qa_index, build_chunk_index,
    _get_encoder,
)
from .retriever import DualRetriever, RetrievalResult
from .router import (
    classify_query, is_coherent,
    STRATEGY_QA_MATCH, STRATEGY_SYNTHESIS, STRATEGY_MULTI_LABEL, STRATEGY_REJECT,
)

logger = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a focused expert on Warren Buffett — his investing philosophy, \
strategies, decisions, psychology, and life history.

RESPONSE RULES:
- Answer ONLY the specific question asked. Do not give broad overviews.
- For focused questions: 2-4 precise sentences.
- For "how/why" questions: 3-6 sentences or short bullet points.
- Use specific names, numbers, dates from the sources provided.
- CITE every factual claim using the source number, e.g. [1], [3]. \
Do not make claims without a citation.
- If the sources do not contain relevant information, say: \
"I don't have specific information about that in my knowledge base."
- Never fabricate facts beyond what the numbered sources provide.
- Never say "the passage says" or "according to the document." \
Speak as if from deeply studied knowledge, but still cite [1], [2], etc.
- If conversation history is present, build on it. Don't repeat prior answers.
- Match response length to question complexity. Short question = short answer."""

GATE_PROMPT = """\
You are a query preprocessor for a Warren Buffett investment knowledge base.

{history_block}Current question: "{query}"

Tasks:
1. Is this question (considering any conversation above) about Warren Buffett, \
Berkshire Hathaway, or investing concepts he discusses (value investing, moats, \
margin of safety, capital allocation, risk management, etc.)?
2. If it references prior conversation, rewrite it as a complete standalone question \
that includes the key terms and context needed for a search.

Respond in EXACTLY this format (two lines, nothing else):
RELEVANT: YES or NO
QUERY: <complete standalone question>"""

QA_MATCH_PROMPT = """\
USER QUESTION: {query}

Source [1] closely matches this question. Use it as primary, cite as [1].
Other sources provide supplementary detail — cite them if you use them.

SOURCES:
{numbered_context}"""

SYNTHESIS_PROMPT = """\
USER QUESTION: {query}

Answer using ONLY these numbered sources. Cite every factual claim inline [1], [2], etc.
Stay focused on the specific question. Do not give a general overview.

SOURCES:
{numbered_context}"""

MULTI_LABEL_PROMPT = """\
USER QUESTION: {query}

This question spans multiple aspects. Answer with short bullet points by theme.
Cite every claim inline [1], [2], etc.

SOURCES:
{numbered_context}"""

REJECT_INCOHERENT = (
    "I couldn't understand that. Could you rephrase your question "
    "about Warren Buffett's investing philosophy?"
)
REJECT_IRRELEVANT = (
    "I'm designed to answer questions about Warren Buffett — his investing "
    "philosophy, strategies, risk management, psychology, and life history. "
    "Could you ask something in that area?"
)


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    tool: str
    query: str
    results_count: int
    best_score: Optional[float] = None
    labels_hit: list[str] = field(default_factory=list)


@dataclass
class SourceRef:
    source_type: str  # "qa_pair" | "chunk"
    label: str
    ref_idx: int = 0  # citation number [1], [2], etc.
    sublabel: str = ""
    source_file: str = ""
    source_section: str = ""
    quality: Optional[float] = None
    similarity: float = 0.0


@dataclass
class AgentResponse:
    answer: str
    strategy: str
    reason: str
    confidence: str  # "high" | "medium" | "low" | "none"
    enriched_query: str = ""  # rewritten query if follow-up was enriched
    tools_called: list[ToolCall] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "strategy": self.strategy,
            "reason": self.reason,
            "confidence": self.confidence,
            "enriched_query": self.enriched_query,
            "tools_called": [vars(t) for t in self.tools_called],
            "sources": [vars(s) for s in self.sources],
            "duration_ms": self.duration_ms,
        }


# ── Agent ────────────────────────────────────────────────────────────────

class BuffettAgent:
    """Stateful RAG agent. Create once at startup, call .answer() per request."""

    def __init__(self):
        logger.info("Initializing BuffettAgent...")
        self.encoder = _get_encoder()
        self.llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

        qa_index, qa_meta = self._load_or_build_qa()
        chunk_index, chunk_meta = self._load_or_build_chunks()

        self.retriever = DualRetriever(
            qa_index=qa_index, qa_meta=qa_meta,
            chunk_index=chunk_index, chunk_meta=chunk_meta,
            encoder=self.encoder,
        )
        logger.info(
            "Agent ready — %d QA pairs, %d chunks indexed",
            qa_index.ntotal if qa_index else 0,
            chunk_index.ntotal if chunk_index else 0,
        )

    # ── Index management ─────────────────────────────────────────────

    def _load_or_build_qa(self):
        idx, meta = load_qa_index(FAISS_INDEX_DIR)
        if idx is not None:
            return idx, meta
        logger.info("Building QA index from CSVs...")
        df = load_qa_dataframe(QA_CSV_DIR)
        return build_qa_index(df, self.encoder, FAISS_INDEX_DIR)

    def _load_or_build_chunks(self):
        idx, meta = load_chunk_index(FAISS_INDEX_DIR)
        if idx is not None:
            return idx, meta
        chunks = load_chunks(CHUNK_PKL_DIR)
        if not chunks:
            logger.warning("No chunk data — chunk index empty")
            import faiss
            dim = self.encoder.get_sentence_embedding_dimension()
            return faiss.IndexFlatIP(dim), []
        logger.info("Building chunk index from pickles...")
        return build_chunk_index(chunks, self.encoder, FAISS_INDEX_DIR)

    # ── Pre-retrieval: gate + enrich ─────────────────────────────────

    def _gate_and_enrich(
        self, query: str, history: Optional[list[dict]],
    ) -> tuple[bool, str]:
        """Single LLM call: checks relevance AND rewrites follow-ups.
        Returns (is_relevant, search_query)."""
        # Build history block for context
        history_block = ""
        if history and len(history) >= 2:
            recent = history[-4:]  # last 2 exchanges max
            lines = []
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content'][:300]}")
            history_block = "Previous conversation:\n" + "\n".join(lines) + "\n\n"

        try:
            resp = self.llm.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{
                    "role": "user",
                    "content": GATE_PROMPT.format(
                        history_block=history_block,
                        query=query,
                    ),
                }],
                temperature=0.0,
                max_tokens=150,
            )
            text = resp.choices[0].message.content.strip()

            # Parse response
            relevant = True
            enriched = query
            for line in text.split("\n"):
                line = line.strip()
                if line.upper().startswith("RELEVANT:"):
                    val = line.split(":", 1)[1].strip().upper()
                    relevant = val.startswith("YES")
                elif line.upper().startswith("QUERY:"):
                    val = line.split(":", 1)[1].strip().strip('"')
                    if 5 < len(val) < 500:
                        enriched = val

            return relevant, enriched

        except Exception as exc:
            logger.warning("Gate check failed (%s) — allowing query", exc)
            return True, query  # fail open

    # ── Context filtering ────────────────────────────────────────────

    @staticmethod
    def _filter_context(retrieval: RetrievalResult) -> RetrievalResult:
        """Remove low-similarity noise before building prompts."""
        qa = [r for r in retrieval.qa_results if r.similarity >= 0.25]
        chunks = [r for r in retrieval.chunk_results if r.similarity >= 0.20]
        return RetrievalResult(
            qa_results=qa,
            chunk_results=chunks,
            best_qa_similarity=qa[0].similarity if qa else 0.0,
        )

    # ── Build numbered context with citation refs ────────────────────

    def _build_numbered_context(
        self, strategy: str, retrieval: RetrievalResult,
    ) -> tuple[str, list[SourceRef]]:
        """Number all sources [1]..[N] for inline citations.
        Returns (formatted_context_string, source_refs_with_indices)."""
        # Limit per strategy
        if strategy == STRATEGY_QA_MATCH:
            qa_limit, chunk_limit = 3, 2
        else:
            qa_limit, chunk_limit = 5, 4

        sources = []
        lines = []
        idx = 1

        for r in retrieval.qa_results[:qa_limit]:
            tag = f"[{idx}]"
            loc = r.label
            if r.sublabel:
                loc += f"/{r.sublabel}"
            lines.append(
                f"{tag} ({loc}, {r.source})\n"
                f"Q: {r.question}\n"
                f"A: {r.answer}"
            )
            sources.append(SourceRef(
                ref_idx=idx,
                source_type="qa_pair",
                label=r.label,
                sublabel=r.sublabel,
                source_file=r.source,
                quality=r.quality,
                similarity=r.similarity,
            ))
            idx += 1

        for r in retrieval.chunk_results[:chunk_limit]:
            tag = f"[{idx}]"
            section = r.source_section[:80] if r.source_section else ""
            loc = r.label
            if r.source_file:
                loc += f", {r.source_file}"
            if section:
                loc += f" > {section}"
            lines.append(f"{tag} ({loc})\n{r.text[:600]}")
            sources.append(SourceRef(
                ref_idx=idx,
                source_type="chunk",
                label=r.label,
                source_file=r.source_file,
                source_section=r.source_section,
                similarity=r.similarity,
            ))
            idx += 1

        return "\n\n".join(lines), sources

    # ── Main entry point ─────────────────────────────────────────────

    def answer(
        self,
        query: str,
        history: Optional[list[dict]] = None,
    ) -> AgentResponse:
        t0 = time.perf_counter()

        # Gate 1: coherence (heuristic, instant)
        if not is_coherent(query):
            return self._reject(REJECT_INCOHERENT, t0)

        # Gate 2: relevance + query enrichment (single LLM call, ~400ms)
        relevant, search_query = self._gate_and_enrich(query, history)
        if not relevant:
            return self._reject(REJECT_IRRELEVANT, t0)

        # Retrieve using enriched query
        retrieval = self.retriever.search(
            search_query,
            qa_top_k=RETRIEVAL_TOP_K,
            chunk_top_k=CHUNK_TOP_K,
        )
        tools_called = self._log_tools(search_query, retrieval)

        # Strip noise
        retrieval = self._filter_context(retrieval)

        # Route
        route = classify_query(search_query, retrieval)
        strategy = route["strategy"]

        # Build numbered context + source refs
        numbered_ctx, sources = self._build_numbered_context(
            strategy, retrieval,
        )

        # Generate
        user_prompt = self._build_prompt(strategy, query, numbered_ctx)
        messages = self._build_messages(user_prompt, history)
        answer_text = self._generate(messages)

        # Package
        confidence = self._assess_confidence(strategy, retrieval)
        duration = int((time.perf_counter() - t0) * 1000)

        return AgentResponse(
            answer=answer_text,
            strategy=strategy,
            reason=route["reason"],
            confidence=confidence,
            enriched_query=search_query if search_query != query else "",
            tools_called=tools_called,
            sources=sources,
            duration_ms=duration,
        )

    def _reject(self, message: str, t0: float) -> AgentResponse:
        return AgentResponse(
            answer=message,
            strategy=STRATEGY_REJECT,
            reason=message,
            confidence="none",
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )

    # ── Tool logging ─────────────────────────────────────────────────

    @staticmethod
    def _log_tools(query: str, retrieval: RetrievalResult) -> list[ToolCall]:
        return [
            ToolCall(
                tool="qa_search",
                query=query,
                results_count=len(retrieval.qa_results),
                best_score=retrieval.best_qa_similarity,
                labels_hit=_unique_labels(
                    [r.label for r in retrieval.qa_results]
                ),
            ),
            ToolCall(
                tool="chunk_search",
                query=query,
                results_count=len(retrieval.chunk_results),
                best_score=(
                    retrieval.chunk_results[0].similarity
                    if retrieval.chunk_results else None
                ),
                labels_hit=_unique_labels(
                    [r.label for r in retrieval.chunk_results]
                ),
            ),
        ]

    # ── Prompt construction ──────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        strategy: str, query: str, numbered_context: str,
    ) -> str:
        if strategy == STRATEGY_QA_MATCH:
            return QA_MATCH_PROMPT.format(
                query=query,
                numbered_context=numbered_context,
            )
        if strategy == STRATEGY_SYNTHESIS:
            return SYNTHESIS_PROMPT.format(
                query=query,
                numbered_context=numbered_context,
            )
        # MULTI_LABEL
        return MULTI_LABEL_PROMPT.format(
            query=query,
            numbered_context=numbered_context,
        )

    @staticmethod
    def _build_messages(
        user_prompt: str, history: Optional[list[dict]],
    ) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for msg in history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _generate(self, messages: list[dict]) -> str:
        try:
            resp = self.llm.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            return "I'm having trouble generating a response. Please try again."

    # ── Confidence ───────────────────────────────────────────────────

    @staticmethod
    def _assess_confidence(strategy: str, retrieval: RetrievalResult) -> str:
        if strategy == STRATEGY_QA_MATCH:
            return "high"
        best = retrieval.best_qa_similarity
        if best >= 0.65:
            return "medium"
        if best >= 0.45:
            return "low"
        return "low"

    # ── Admin ────────────────────────────────────────────────────────

    def rebuild_indices(self) -> dict:
        from .indexer import rebuild_all
        stats = rebuild_all()
        qa_index, qa_meta = load_qa_index(FAISS_INDEX_DIR)
        chunk_index, chunk_meta = load_chunk_index(FAISS_INDEX_DIR)
        self.retriever = DualRetriever(
            qa_index=qa_index, qa_meta=qa_meta,
            chunk_index=chunk_index, chunk_meta=chunk_meta,
            encoder=self.encoder,
        )
        return stats

    @property
    def index_stats(self) -> dict:
        return {
            "qa_vectors": (
                self.retriever.qa_index.ntotal
                if self.retriever.qa_index else 0
            ),
            "chunk_vectors": (
                self.retriever.chunk_index.ntotal
                if self.retriever.chunk_index else 0
            ),
        }


# ── Helpers ──────────────────────────────────────────────────────────────

def _unique_labels(labels: list[str]) -> list[str]:
    seen = []
    for lb in labels:
        if lb and lb not in seen:
            seen.append(lb)
    return seen
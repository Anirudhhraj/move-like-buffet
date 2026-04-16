"""
ChunkResearcher — multi-round agentic retrieval with HyDE.

Called by BuffettAgent for synthesis/multi-label queries.
Explores the chunk index iteratively:
  1. HyDE: generate hypothetical passage, embed, search
  2. Analyze collected evidence, identify gaps
  3. Generate targeted queries, search again
  4. Repeat until sufficient or max rounds

Streams ResearchEvents via callback for live terminal display.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
from sentence_transformers import SentenceTransformer

from config import DEEPSEEK_MODEL

logger = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────────

HYDE_PROMPT = """\
Write a short factual paragraph (3-4 sentences) answering this question \
about Warren Buffett as if you were an expert with deep knowledge of his \
letters, speeches, and investment decisions. Use specific names, numbers, \
and examples.

Question: {query}"""

ANALYSIS_PROMPT = """\
You are evaluating evidence for a question about Warren Buffett.

QUESTION: {query}

CURATED Q&A ALREADY AVAILABLE:
{qa_context}

RAW SOURCE PASSAGES COLLECTED:
{evidence}

Tasks:
1. Considering both the Q&A pairs and raw passages, is there enough evidence \
to answer the question thoroughly with specific facts and examples?
2. If not, what specific aspect is missing? Generate 1-2 short search queries \
to find the missing evidence.

Respond in EXACTLY this format (three lines):
SUFFICIENT: YES or NO
REASONING: <one sentence>
QUERIES: <1-2 queries separated by | , or NONE if sufficient>"""


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class ResearchEvent:
    """Single event emitted during research for live streaming."""
    step: str     # hyde, search, analyze, gap, query, found, sufficient, done
    detail: str

@dataclass
class ResearchResult:
    """Output of a research run."""
    chunks: list[dict]
    events: list[ResearchEvent]
    rounds_used: int
    total_searches: int


# ── Researcher ───────────────────────────────────────────────────────────

class ChunkResearcher:
    """Multi-round chunk retrieval with HyDE and gap analysis.

    Usage:
        researcher = ChunkResearcher(encoder, index, meta, llm, on_event=print_fn)
        result = researcher.research(query, initial_chunks, qa_context)
    """

    MAX_ROUNDS = 3       # max analysis rounds after HyDE
    MAX_CHUNKS = 15      # stop accumulating beyond this
    HYDE_TOP_K = 6       # chunks retrieved via HyDE
    QUERY_TOP_K = 4      # chunks per follow-up query
    MIN_SIM = 0.15       # ignore chunks below this

    def __init__(
        self,
        encoder: SentenceTransformer,
        chunk_index,       # faiss.Index
        chunk_meta: list[dict],
        llm,               # OpenAI-compatible client
        on_event: Optional[Callable[[ResearchEvent], None]] = None,
    ):
        self.encoder = encoder
        self.chunk_index = chunk_index
        self.chunk_meta = chunk_meta
        self.llm = llm
        self.on_event = on_event
        self.events: list[ResearchEvent] = []
        self._search_count = 0

    def _emit(self, step: str, detail: str):
        event = ResearchEvent(step=step, detail=detail)
        self.events.append(event)
        if self.on_event:
            self.on_event(event)

    # ── Main loop ────────────────────────────────────────────────────

    def research(
        self,
        query: str,
        initial_chunks: list[dict],
        qa_context: str = "",
    ) -> ResearchResult:
        """Run HyDE + iterative retrieval. Returns enriched chunk set."""
        # Early exit if no chunk index
        if self.chunk_index is None or self.chunk_index.ntotal == 0:
            self._emit("skip", "No chunk index available")
            return ResearchResult(
                chunks=initial_chunks, events=self.events,
                rounds_used=0, total_searches=0,
            )

        all_chunks = list(initial_chunks)
        seen_ids = {
            c.get("chunk_id", "") for c in all_chunks
            if c.get("chunk_id")
        }
        rounds_used = 0

        # ── HyDE pass ───────────────────────────────────────────────
        self._emit("hyde", "Generating hypothetical passage...")
        hypothesis = self._generate_hypothesis(query)
        if hypothesis:
            self._emit("hyde_done", hypothesis + "...")
            hyde_results = self._search_chunks(hypothesis, self.HYDE_TOP_K)
            new = self._merge_new(hyde_results, seen_ids)
            all_chunks.extend(new)
            self._emit("search", f"HyDE retrieved {len(new)} new chunks")
        else:
            self._emit("hyde_fail", "HyDE generation failed, continuing without")

        # ── Iterative gap-fill rounds ────────────────────────────────
        for round_num in range(1, self.MAX_ROUNDS + 1):
            rounds_used = round_num

            if len(all_chunks) >= self.MAX_CHUNKS:
                self._emit("cap", f"Chunk limit reached ({self.MAX_CHUNKS})")
                break

            self._emit("analyze", f"Round {round_num}: Evaluating evidence...")
            analysis = self._analyze_evidence(query, all_chunks, qa_context)

            if analysis["sufficient"]:
                self._emit("sufficient", analysis["reasoning"])
                break

            self._emit("gap", analysis["reasoning"])

            for fq in analysis["queries"][:2]:
                if len(all_chunks) >= self.MAX_CHUNKS:
                    break
                self._emit("query", fq)
                results = self._search_chunks(fq, self.QUERY_TOP_K)
                new = self._merge_new(results, seen_ids)
                all_chunks.extend(new)
                self._emit("found", f"+{len(new)} chunks")

        # Sort by similarity (best first)
        all_chunks.sort(key=lambda c: c.get("similarity", 0), reverse=True)

        self._emit(
            "done",
            f"Research complete: {len(all_chunks)} chunks from {self._search_count} searches",
        )
        return ResearchResult(
            chunks=all_chunks,
            events=self.events,
            rounds_used=rounds_used,
            total_searches=self._search_count,
        )

    # ── HyDE hypothesis generation ───────────────────────────────────

    def _generate_hypothesis(self, query: str) -> str:
        try:
            resp = self.llm.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{
                    "role": "user",
                    "content": HYDE_PROMPT.format(query=query),
                }],
                temperature=0.5,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)
            return ""

    # ── Chunk search (embed + FAISS) ─────────────────────────────────

    def _search_chunks(self, text: str, top_k: int) -> list[dict]:
        self._search_count += 1

        embedding = self.encoder.encode(
            [text], normalize_embeddings=True,
        ).astype(np.float32)

        k = min(top_k, self.chunk_index.ntotal)
        scores, indices = self.chunk_index.search(embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_meta):
                continue
            sim = round(float(score), 4)
            if sim < self.MIN_SIM:
                continue
            meta = self.chunk_meta[idx]
            results.append({
                "text": meta.get("text", ""),
                "label": meta.get("label", ""),
                "source_file": meta.get("source_file", ""),
                "source_section": meta.get("source_section", ""),
                "chunk_id": meta.get("chunk_id", ""),
                "similarity": sim,
            })
        return results

    # ── Evidence gap analysis ────────────────────────────────────────

    def _analyze_evidence(
        self, query: str, chunks: list[dict], qa_context: str,
    ) -> dict:
        evidence = "\n\n".join(
            f"[{c.get('label', '?')} — {c.get('source_file', '?')}]\n"
            f"{c['text'][:400]}"
            for c in chunks[:10]
        )
        try:
            resp = self.llm.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{
                    "role": "user",
                    "content": ANALYSIS_PROMPT.format(
                        query=query,
                        evidence=evidence or "None yet.",
                        qa_context=qa_context if qa_context else "None.",
                    ),
                }],
                temperature=0.0,
                max_tokens=200,
            )
            return self._parse_analysis(resp.choices[0].message.content.strip())
        except Exception as exc:
            logger.warning("Evidence analysis failed: %s", exc)
            return {
                "sufficient": True,
                "reasoning": f"Analysis failed ({exc}), using current evidence",
                "queries": [],
            }

    @staticmethod
    def _parse_analysis(text: str) -> dict:
        sufficient = True
        reasoning = ""
        queries = []
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("SUFFICIENT:"):
                sufficient = line.split(":", 1)[1].strip().upper().startswith("YES")
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.upper().startswith("QUERIES:"):
                val = line.split(":", 1)[1].strip()
                if val.upper() != "NONE" and val:
                    queries = [q.strip() for q in val.split("|") if q.strip()]
        return {"sufficient": sufficient, "reasoning": reasoning, "queries": queries}

    # ── Deduplication ────────────────────────────────────────────────

    @staticmethod
    def _merge_new(chunks: list[dict], seen_ids: set) -> list[dict]:
        new = []
        for c in chunks:
            cid = c.get("chunk_id", "")
            if cid and cid in seen_ids:
                continue
            if cid:
                seen_ids.add(cid)
            new.append(c)
        return new
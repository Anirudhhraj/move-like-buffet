"""
Shared infrastructure for the Buffett Q&A synthesis pipeline.
All notebooks import from this module.
Chunking logic lives in each document's notebook — it's document-specific.
"""

import json
import re
import csv
import hashlib
import asyncio
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from collections import Counter

import fitz  # PyMuPDF
import nest_asyncio
from litellm import acompletion
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()


# ============================================================
# UTILITIES
# ============================================================

def _parse_llm_json(raw: str):
    """Strip markdown fences, preamble text, and parse JSON."""
    cleaned = re.sub(r'```json\s*|```\s*', '', raw).strip()
    for i, char in enumerate(cleaned):
        if char in ('{', '['):
            cleaned = cleaned[i:]
            break
    for i in range(len(cleaned) - 1, -1, -1):
        if cleaned[i] in ('}', ']'):
            cleaned = cleaned[:i + 1]
            break
    return json.loads(cleaned)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Chunk:
    text: str
    source_file: str
    source_section: Optional[str] = None
    chunk_strategy: str = ""
    pre_label: Optional[str] = None
    llm_label: Optional[str] = None
    confidence: Optional[float] = None

    @property
    def label(self) -> Optional[str]:
        return self.pre_label or self.llm_label

    @property
    def chunk_id(self) -> str:
        return hashlib.md5(f"{self.source_file}:{self.text[:100]}".encode()).hexdigest()[:12]

    def __repr__(self):
        return f"Chunk({self.source_file[:25]}.. | {self.label} | {len(self.text)} chars)"


@dataclass
class QAPair:
    question: str
    answer: str
    label: str
    sublabel: Optional[str] = None
    source_chunk_id: str = ""
    source_file: str = ""
    generation_model: str = ""
    prompt_type: str = ""  # "reference", "conceptual", or "analytical"
    groundedness_score: Optional[float] = None
    label_fit_score: Optional[float] = None
    richness_score: Optional[float] = None
    novelty_score: Optional[float] = None

    @property
    def composite_score(self) -> Optional[float]:
        scores = [self.groundedness_score, self.label_fit_score,
                  self.richness_score, self.novelty_score]
        if any(s is None for s in scores):
            return None
        weights = [0.35, 0.25, 0.20, 0.20]
        return sum(s * w for s, w in zip(scores, weights))


# ============================================================
# LABEL DEFINITIONS
# ============================================================

LABELS = {
    "Personal Life": {
        "description": "How Buffett's personal upbringing, life circumstances, mindsets, or habits influence his investing style.",
        "sublabels": ["early_life", "education", "mentors", "habits", "family_influence", "personal_values", "lifestyle"],
        "min_pairs": 100,
    },
    "Strategy Development": {
        "description": "Core investing strategy and how it was developed. Underlying principles and logic guiding decision-making.",
        "sublabels": ["value_investing_framework", "margin_of_safety", "competitive_moat", "business_quality", "circle_of_competence", "capital_allocation", "graham_influence"],
        "min_pairs": 100,
    },
    "Timing": {
        "description": "How Buffett decides when to enter and exit a position — fundamental signals, market valuation, or sentiment.",
        "sublabels": ["entry_criteria", "exit_criteria", "market_valuation", "opportunity_cost", "patience", "price_vs_value", "market_cycles"],
        "min_pairs": 100,
    },
    "Risk Management": {
        "description": "How Buffett manages risk and limits losses. Position sizing, leverage avoidance, margin of safety.",
        "sublabels": ["position_sizing", "diversification", "leverage_avoidance", "permanent_loss", "insurance_float", "margin_of_safety_risk", "concentration"],
        "min_pairs": 100,
    },
    "Adaptability": {
        "description": "How Buffett adjusts strategy in different or adverse market conditions.",
        "sublabels": ["bear_market_behavior", "crisis_response", "strategy_evolution", "mistake_correction", "market_regime_shifts", "new_opportunities"],
        "min_pairs": 100,
    },
    "Psychology": {
        "description": "Role of psychology in Buffett's investing. Temperament, discipline, emotional management, contrarian thinking.",
        "sublabels": ["temperament", "emotional_discipline", "contrarian_thinking", "patience_psychology", "fear_greed", "independence", "rationality"],
        "min_pairs": 100,
    },
}


# ============================================================
# LLM CONFIGURATION
# ============================================================

PRIMARY_MODEL = "deepseek/deepseek-chat"
FALLBACK_MODEL = "gpt-4o"
SCORING_MODEL = "deepseek/deepseek-chat"


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


# ============================================================
# CLASSIFICATION
# ============================================================

CLASSIFICATION_PROMPT = """You are classifying text about Warren Buffett into exactly one label.

LABELS:
1. Personal Life - upbringing, habits, personal values that shaped his investing
2. Strategy Development - core framework, principles, decision-making logic, value investing
3. Timing - when to enter/exit positions, market valuation signals, patience
4. Risk Management - limiting losses, position sizing, leverage avoidance, margin of safety
5. Adaptability - adjusting strategy in different markets, handling crises, evolving approach
6. Psychology - emotional discipline, temperament, contrarian thinking, rationality

PASSAGE:
{chunk_text}

Respond with ONLY a JSON object:
{{"label": "<label name>", "confidence": <0.0-1.0>}}"""


async def classify_chunk(chunk: Chunk, model: str = PRIMARY_MODEL) -> Chunk:
    if chunk.pre_label is not None:
        return chunk
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": CLASSIFICATION_PROMPT.format(
                chunk_text=chunk.text[:2000]
            )}],
            temperature=0.1,
            max_tokens=100,
        )
        result = _parse_llm_json(response.choices[0].message.content)
        chunk.llm_label = result["label"]
        chunk.confidence = result.get("confidence", 0.5)
    except Exception as e:
        print(f"  [WARN] Classification failed for {chunk.chunk_id}: {e}")
        chunk.llm_label = None
        chunk.confidence = 0.0
    return chunk


async def classify_chunks(chunks: List[Chunk], batch_size: int = 10) -> List[Chunk]:
    needs_work = [c for c in chunks if c.pre_label is None]
    print(f"Classifying {len(needs_work)} chunks (skipping {len(chunks) - len(needs_work)} pre-labeled)")
    for i in range(0, len(needs_work), batch_size):
        batch = needs_work[i:i + batch_size]
        await asyncio.gather(*[classify_chunk(c) for c in batch])
        print(f"  {min(i + batch_size, len(needs_work))}/{len(needs_work)}")

    failed = sum(1 for c in chunks if c.label is None)
    if failed:
        print(f"\n  WARNING: {failed} chunks remain unclassified")

    dist = Counter(c.label for c in chunks if c.label)
    print(f"\nLabel distribution:")
    for label, count in dist.most_common():
        print(f"  {label}: {count}")
    return chunks


# ============================================================
# Q&A GENERATION — THREE SEPARATE PROMPTS
# ============================================================

REFERENCE_PROMPT = """You are generating training data for a Warren Buffett chatbot. Generate a SHORT factual Q&A pair.

LABEL: {label}
FOCUS: {label_description}
SUBLABELS: {sublabels}

KNOWLEDGE SOURCE (use for facts, never reference directly):
---
{chunk_text}
---

RULES:
1. Generate exactly 1 quick factual Q&A pair.
2. The QUESTION must sound like something a real person would type into a chatbot. Short, direct, natural language. Examples: "When did Buffett buy See's Candy?", "What is Buffett's margin of safety?", "How much did Berkshire pay for GEICO?"
3. The ANSWER must be 1-2 sentences. Crisp and precise.
4. Both question and answer must be completely self-contained. A reader should understand them with zero external context. NEVER use phrases like "the passage says", "according to the text", "in this excerpt", "the source mentions", or any reference to an underlying document.
5. Ground every claim in the knowledge source above, but write as if speaking from memory — not quoting a document.
6. Include at least one specific name, number, or date.

Respond with ONLY a JSON array:
[{{"question": "...", "answer": "...", "sublabel": "..."}}]"""


CONCEPTUAL_PROMPT = """You are generating training data for a Warren Buffett chatbot. Generate a REASONING Q&A pair that explains WHY or HOW Buffett thinks about something.

LABEL: {label}
FOCUS: {label_description}
SUBLABELS: {sublabels}

KNOWLEDGE SOURCE (use for facts, never reference directly):
---
{chunk_text}
---

RULES:
1. Generate exactly 1 "why" or "how" Q&A pair.
2. The QUESTION must sound like a real person asking a chatbot for an explanation. One sentence, natural phrasing. Examples: "Why does Buffett avoid leverage?", "How does Buffett evaluate whether a business has a moat?", "Why did Buffett sell his textile operations?"
3. The ANSWER should explain Buffett's reasoning in his own voice — preserve his analogies, logic, and specific examples rather than abstracting them into generic investment language. No length constraint — answer as thoroughly as the explanation requires, typically 3-5 sentences.
4. Both question and answer must be completely self-contained. NEVER reference "the passage", "the text", "this source", "the letter", or any underlying document. Write as if the chatbot is speaking from deep internalized knowledge of Buffett's philosophy.
5. Ground every claim in the knowledge source above, but present it as direct knowledge, not citation.

Respond with ONLY a JSON array:
[{{"question": "...", "answer": "...", "sublabel": "..."}}]"""


ANALYTICAL_PROMPT = """You are generating training data for a Warren Buffett chatbot. Generate a DEEP analytical Q&A pair that connects ideas or examines a specific investment decision in detail.

LABEL: {label}
FOCUS: {label_description}
SUBLABELS: {sublabels}

KNOWLEDGE SOURCE (use for facts, never reference directly):
---
{chunk_text}
---

RULES:
1. Generate exactly 1 analytical Q&A pair.
2. The QUESTION should ask about a specific decision, tradeoff, comparison, or evolution in Buffett's thinking. Keep it to 1-2 sentences max. Natural language, not academic essay prompts. Examples: "How did Buffett's approach to See's Candy change his thinking about what makes a great business?", "What went wrong with Buffett's US Air investment and what did he learn from it?", "How does Buffett's experience with the textile business illustrate the difference between a good manager and a good business?"
3. The ANSWER should be comprehensive — 5-8 sentences with specific names, numbers, and examples. Walk through Buffett's reasoning process, not just his conclusions. Preserve his analogies and concrete details.
4. Both question and answer must be completely self-contained. NEVER reference "the passage", "the text", "the source", "the letter", "this excerpt", or any underlying document. The chatbot speaks from internalized knowledge, not from a document it is reading.
5. Ground every claim in the knowledge source above. If the source doesn't contain enough for a deep analytical answer, return an empty array: []

Respond with ONLY a JSON array:
[{{"question": "...", "answer": "...", "sublabel": "..."}}]"""

PROMPTS = {
    "reference": REFERENCE_PROMPT,
    "conceptual": CONCEPTUAL_PROMPT,
    "analytical": ANALYTICAL_PROMPT,
}


async def generate_pairs(chunk: Chunk, prompt_type: str = "conceptual",
                         model: str = PRIMARY_MODEL) -> List[QAPair]:
    label = chunk.label
    if not label or label not in LABELS:
        return []
    lcfg = LABELS[label]
    prompt_template = PROMPTS[prompt_type]
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt_template.format(
                label=label,
                label_description=lcfg["description"],
                sublabels=", ".join(lcfg["sublabels"]),
                chunk_text=chunk.text[:3000],
            )}],
            temperature=0.7,
            max_tokens=2000,
        )
        pairs_data = _parse_llm_json(response.choices[0].message.content)
        return [
            QAPair(
                question=p["question"], answer=p["answer"], label=label,
                sublabel=p.get("sublabel"), source_chunk_id=chunk.chunk_id,
                source_file=chunk.source_file, generation_model=model,
                prompt_type=prompt_type,
            )
            for p in pairs_data if "question" in p and "answer" in p
        ]
    except Exception as e:
        print(f"  [ERROR] {prompt_type} generation failed for {chunk.chunk_id}: {e}")
        return []


async def generate_all(chunks: List[Chunk], batch_size: int = 5) -> List[QAPair]:
    """Run all three generation passes (reference, conceptual, analytical) for each chunk."""
    all_pairs = []
    total_calls = len(chunks) * 3

    for prompt_type in ["reference", "conceptual", "analytical"]:
        print(f"\n  Pass: {prompt_type}")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            results = await asyncio.gather(*[generate_pairs(c, prompt_type) for c in batch])
            for pairs in results:
                all_pairs.extend(pairs)
            done = min(i + batch_size, len(chunks))
            print(f"    {done}/{len(chunks)} chunks | {len(all_pairs)} total pairs")

    return all_pairs


# ============================================================
# QUALITY SCORING
# ============================================================

SCORING_PROMPT = """You are a STRICT quality auditor for a Q&A training dataset about Warren Buffett.
Score this pair on each dimension from 0.0 to 1.0. Be critical — a score of 1.0 means flawless, 
0.8 means good with minor issues, 0.6 means mediocre. Most pairs should score between 0.6 and 0.85.

LABEL: {label}
QUESTION: {question}
ANSWER: {answer}
SOURCE: {source_text}

Score on:
1. groundedness - Is EVERY claim in the answer directly supported by the source text? Penalize anything added beyond the source, even if true. (0.5=partly grounded, 0.7=mostly grounded, 0.9=fully grounded)
2. label_fit - Does this pair clearly belong under '{label}' and not a different label? (0.5=ambiguous, 0.7=reasonable fit, 0.9=perfect fit)
3. richness - Is the answer appropriately detailed FOR THE QUESTION ASKED? A simple factual question answered in 1-2 precise sentences scores just as high as a complex question answered in 5-7 sentences. Penalize vagueness, not brevity. (0.5=vague regardless of length, 0.7=adequate detail, 0.9=precisely right amount of detail)
4. novelty - Would this teach something non-obvious to someone studying Buffett? Not a generic fact anyone would know? (0.5=common knowledge, 0.7=useful insight, 0.9=genuinely illuminating)

Respond ONLY with JSON:
{{"groundedness": 0.0, "label_fit": 0.0, "richness": 0.0, "novelty": 0.0}}"""


async def score_pair(pair: QAPair, source_chunk: Chunk,
                     model: str = SCORING_MODEL) -> QAPair:
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": SCORING_PROMPT.format(
                label=pair.label, question=pair.question,
                answer=pair.answer, source_text=source_chunk.text[:1500],
            )}],
            temperature=0.0,
            max_tokens=100,
        )
        scores = _parse_llm_json(response.choices[0].message.content)
        pair.groundedness_score = scores.get("groundedness", 0)
        pair.label_fit_score = scores.get("label_fit", 0)
        pair.richness_score = scores.get("richness", 0)
        pair.novelty_score = scores.get("novelty", 0)
    except Exception as e:
        print(f"  [WARN] Scoring failed: {e}")
        pair.groundedness_score = 0.0
    return pair


async def score_all(pairs: List[QAPair], chunk_map: dict,
                    batch_size: int = 10) -> List[QAPair]:
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        await asyncio.gather(*[
            score_pair(p, chunk_map.get(p.source_chunk_id, Chunk(text="", source_file="")))
            for p in batch
        ])
        print(f"  Scored {min(i + batch_size, len(pairs))}/{len(pairs)}")
    return pairs


def filter_by_quality(pairs: List[QAPair], threshold: float = 0.7) -> List[QAPair]:
    passed = [p for p in pairs if p.composite_score and p.composite_score >= threshold]
    print(f"Quality filter: {len(passed)}/{len(pairs)} passed (threshold={threshold})")
    return passed


# ============================================================
# COVERAGE AUDIT
# ============================================================

def coverage_audit(pairs: List[QAPair]) -> dict:
    report = {}
    total = len(pairs)
    for label, lcfg in LABELS.items():
        label_pairs = [p for p in pairs if p.label == label]
        sub_counts = Counter(p.sublabel for p in label_pairs if p.sublabel)
        covered = [s for s in lcfg["sublabels"] if sub_counts.get(s, 0) > 0]
        count = len(label_pairs)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {label:25s} {count:3d} pairs ({pct:4.1f}%)")
        if covered:
            print(f"    Sublabels hit: {', '.join(covered)}")
        report[label] = {"count": count, "sublabels_covered": covered}
    print(f"\n  Total: {total} pairs")
    return report


# ============================================================
# EXPORT
# ============================================================

def export_csv(pairs: List[QAPair], path: Path):
    sorted_pairs = sorted(pairs, key=lambda p: p.label)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Questions", "Answers", "Label"])
        for p in sorted_pairs:
            writer.writerow([p.question, p.answer, p.label])
    dist = Counter(p.label for p in sorted_pairs)
    print(f"Exported {len(sorted_pairs)} pairs to {path}")
    for label, count in dist.most_common():
        print(f"  {label}: {count}")


def export_detailed(pairs: List[QAPair], path: Path):
    sorted_pairs = sorted(pairs, key=lambda p: (p.label, -(p.composite_score or 0)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Questions", "Answers", "Label", "Sublabel", "Source",
                         "Quality", "Groundedness", "Label_Fit", "Richness", "Novelty", "Prompt_Type"])
        for p in sorted_pairs:
            writer.writerow([
                p.question, p.answer, p.label, p.sublabel or "",
                p.source_file, f"{p.composite_score:.2f}" if p.composite_score else "",
                p.groundedness_score, p.label_fit_score,
                p.richness_score, p.novelty_score, p.prompt_type,
            ])
    print(f"Detailed export: {path}")


# ============================================================
# PERSISTENCE
# ============================================================

def save_checkpoint(data, name: str, directory: Path = None):
    if directory is None:
        directory = Path(__file__).resolve().parent.parent / "intermediate"
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved: {path}")


def load_checkpoint(name: str, directory: Path = None):
    if directory is None:
        directory = Path(__file__).resolve().parent.parent / "intermediate"
    path = directory / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
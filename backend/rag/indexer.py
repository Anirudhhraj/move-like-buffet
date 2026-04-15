"""
Build and persist dual FAISS indices from the pipeline outputs.

Index A  — "qa": embeds *questions*, stores answers + full metadata.
           Used for direct-match retrieval when the user's query closely
           matches a curated question.

Index B  — "chunks": embeds raw classified source chunks.
           Used for synthesis when no strong QA match exists.

Both are saved to FAISS_INDEX_DIR and loaded once at startup.
Rebuild only when underlying data changes.
"""

import glob
import hashlib
import io
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, QA_CSV_DIR, CHUNK_PKL_DIR, FAISS_INDEX_DIR

logger = logging.getLogger(__name__)

# ── Persistent artefact filenames ────────────────────────────────────────
QA_INDEX_FILE = "qa.index"
QA_META_FILE = "qa_meta.json"
CHUNK_INDEX_FILE = "chunks.index"
CHUNK_META_FILE = "chunks_meta.json"


# ── Local Chunk class for safe unpickling ────────────────────────────────

class _Chunk:
    """
    Backend-local stand-in for the pipeline's Chunk dataclass.
    Pickle restores instance attributes (__dict__) directly.
    """
    text: str = ""
    source_file: str = ""
    source_section: str = ""
    chunk_strategy: str = ""
    pre_label: str = None
    llm_label: str = None
    confidence: float = None

    @property
    def label(self) -> Optional[str]:
        return self.pre_label or self.llm_label

    @property
    def chunk_id(self) -> str:
        raw = f"{self.source_file}:{(self.text or '')[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def __repr__(self):
        src = (self.source_file or "?")[:25]
        return f"_Chunk({src}.. | {self.label} | {len(self.text or '')} chars)"


class _ChunkUnpickler(pickle.Unpickler):
    """Maps ANY class named 'Chunk' to our local _Chunk."""
    def find_class(self, module: str, name: str):
        if name == "Chunk":
            return _Chunk
        if name == "QAPair":
            return type("QAPair", (), {})
        return super().find_class(module, name)


def _safe_unpickle(filepath: str) -> list:
    with open(filepath, "rb") as fh:
        return _ChunkUnpickler(fh).load()


# ── Encoder ──────────────────────────────────────────────────────────────

def _get_encoder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


# ── Data loading ─────────────────────────────────────────────────────────

def load_qa_dataframe(csv_dir: Path) -> pd.DataFrame:
    files = glob.glob(str(csv_dir / "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]
            rename = {}
            if "questions" in df.columns:
                rename["questions"] = "question"
            if "answers" in df.columns:
                rename["answers"] = "answer"
            if rename:
                df = df.rename(columns=rename)

            if "question" not in df.columns or "answer" not in df.columns:
                logger.warning("Skipping %s — missing question/answer columns", f)
                print(f"  [WARN] Skipping {Path(f).name} — missing question/answer columns")
                continue

            df["_source_file"] = Path(f).name
            frames.append(df)
            print(f"  loaded {Path(f).name} — {len(df)} rows")
        except Exception as exc:
            logger.warning("Failed to load %s: %s", f, exc)
            print(f"  [ERROR] Failed to load {Path(f).name}: {exc}")

    if not frames:
        raise FileNotFoundError(f"No valid CSV files with question/answer columns in {csv_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["question", "answer"])
    logger.info("Loaded %d QA rows from %d CSVs", len(combined), len(frames))
    return combined


def load_chunks(pkl_dir: Path) -> list[dict]:
    files = sorted(glob.glob(str(pkl_dir / "*_classified.pkl")))
    if not files:
        logger.warning("No classified chunk pickles found in %s", pkl_dir)
        print(f"  [WARN] No *_classified.pkl files found in {pkl_dir}")
        return []

    all_chunks = []
    for f in files:
        fname = Path(f).name
        try:
            chunks = _safe_unpickle(f)

            if not isinstance(chunks, list):
                print(f"  [WARN] {fname} — expected list, got {type(chunks).__name__}. Skipping.")
                continue

            loaded = 0
            skipped = 0
            for chunk in chunks:
                label = getattr(chunk, "label", None)
                if label is None:
                    skipped += 1
                    continue

                text = getattr(chunk, "text", "")
                if not text or not text.strip():
                    skipped += 1
                    continue

                all_chunks.append({
                    "text": text,
                    "source_file": getattr(chunk, "source_file", ""),
                    "source_section": getattr(chunk, "source_section", ""),
                    "label": label,
                    "chunk_id": getattr(chunk, "chunk_id", ""),
                })
                loaded += 1

            print(f"  {fname} — {loaded} chunks loaded"
                  + (f", {skipped} skipped (no label/text)" if skipped else ""))

        except Exception as exc:
            logger.warning("Failed to load %s: %s", f, exc)
            print(f"  [ERROR] {fname} — {type(exc).__name__}: {exc}")

    total = len(all_chunks)
    logger.info("Loaded %d classified chunks from %d pickle files", total, len(files))
    print(f"  Total: {total} classified chunks from {len(files)} files")
    return all_chunks


# ── Index construction ───────────────────────────────────────────────────

def build_qa_index(
    df: pd.DataFrame,
    encoder: SentenceTransformer,
    index_dir: Path,
) -> tuple[faiss.Index, list[dict]]:
    """Embed questions, build FAISS index, save to disk."""
    questions = df["question"].tolist()
    logger.info("Embedding %d questions …", len(questions))
    print(f"  Embedding {len(questions)} questions …")
    embeddings = encoder.encode(questions, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    meta = []
    for _, row in df.iterrows():
        meta.append({
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "label": row.get("label", ""),
            "sublabel": row.get("sublabel", ""),
            "source": row.get("source", row.get("_source_file", "")),
            "quality": _safe_float(row.get("quality")),
            "groundedness": _safe_float(row.get("groundedness")),
            "prompt_type": row.get("prompt_type", ""),
        })

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / QA_INDEX_FILE))
    (index_dir / QA_META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info("QA index saved — %d vectors, dim=%d", index.ntotal, dim)
    print(f"  QA index saved — {index.ntotal} vectors, dim={dim}")
    return index, meta


def build_chunk_index(
    chunks: list[dict],
    encoder: SentenceTransformer,
    index_dir: Path,
) -> tuple[faiss.Index, list[dict]]:
    """Embed chunk texts, build FAISS index, save to disk."""
    if not chunks:
        logger.warning("No chunks to index")
        print("  [WARN] No chunks to index — chunk index will be empty")
        dim = encoder.get_sentence_embedding_dimension()
        empty = faiss.IndexFlatIP(dim)
        return empty, []

    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d chunks …", len(texts))
    print(f"  Embedding {len(texts)} chunks …")
    embeddings = encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    meta = []
    for c in chunks:
        meta.append({
            "text": c["text"][:4000],
            "source_file": c["source_file"],
            "source_section": c["source_section"],
            "label": c["label"],
            "chunk_id": c["chunk_id"],
        })

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / CHUNK_INDEX_FILE))
    (index_dir / CHUNK_META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info("Chunk index saved — %d vectors, dim=%d", index.ntotal, dim)
    print(f"  Chunk index saved — {index.ntotal} vectors, dim={dim}")
    return index, meta


# ── Load from disk ───────────────────────────────────────────────────────

def load_qa_index(index_dir: Path) -> tuple[Optional[faiss.Index], list[dict]]:
    idx_path = index_dir / QA_INDEX_FILE
    meta_path = index_dir / QA_META_FILE
    if not idx_path.exists() or not meta_path.exists():
        return None, []
    index = faiss.read_index(str(idx_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    logger.info("QA index loaded from disk — %d vectors", index.ntotal)
    return index, meta


def load_chunk_index(index_dir: Path) -> tuple[Optional[faiss.Index], list[dict]]:
    idx_path = index_dir / CHUNK_INDEX_FILE
    meta_path = index_dir / CHUNK_META_FILE
    if not idx_path.exists() or not meta_path.exists():
        return None, []
    index = faiss.read_index(str(idx_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    logger.info("Chunk index loaded from disk — %d vectors", index.ntotal)
    return index, meta


# ── Full rebuild entry point ─────────────────────────────────────────────

def rebuild_all() -> dict:
    """Rebuild both indices from source data. Call once or when data changes."""
    encoder = _get_encoder()
    stats = {}

    print("\n  --- QA Index ---")
    df = load_qa_dataframe(QA_CSV_DIR)
    qa_index, qa_meta = build_qa_index(df, encoder, FAISS_INDEX_DIR)
    stats["qa_pairs"] = qa_index.ntotal

    print("\n  --- Chunk Index ---")
    chunks = load_chunks(CHUNK_PKL_DIR)
    chunk_index, chunk_meta = build_chunk_index(chunks, encoder, FAISS_INDEX_DIR)
    stats["chunks"] = chunk_index.ntotal

    return stats


# ── Helpers ──────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), 4)
    except (ValueError, TypeError):
        return None
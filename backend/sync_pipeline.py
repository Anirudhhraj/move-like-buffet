"""
Sync pipeline outputs into the backend data directory and rebuild FAISS indices.

This script:
  1. Copies the SINGLE deduplicated QA CSV from the pipeline's output/ folder
  2. Copies all *_classified.pkl chunk files from the pipeline's intermediate/ folder
  3. Triggers a full FAISS index rebuild

Usage:
    python sync_pipeline.py                          # uses path from .env
    python sync_pipeline.py --pipeline-dir /path/to  # explicit path

The script is idempotent — safe to re-run anytime the pipeline produces new data.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# ── Expected pipeline structure ──────────────────────────────────────────
# buffet-qa-pipeline/
#   output/
#     buffett_qa_detailed.csv    ← THE dataset (deduplicated, scored)
#     buffett_qa_dataset.csv     ← simple 3-col version (we don't need this)
#   intermediate/
#     florida_classified.pkl     ← classified chunks
#     ivey_classified.pkl
#     letters_classified.pkl
#     cunningham_classified.pkl
#     notredame_classified.pkl
#     shareholder_classified.pkl

# Files we WANT
QA_SOURCE_FILE = "buffett_qa_detailed.csv"   # the ONE deduplicated CSV
CHUNK_PKL_PATTERN = "*_classified.pkl"        # all classified chunk pickles

# Files we IGNORE (and why)
# - buffett_qa_dataset.csv      → subset of detailed, missing scoring columns
# - *_qa.csv / *_qa_detailed.csv → per-document, NOT deduplicated, WILL overlap
# - *_pairs_raw.pkl             → pre-quality-filter, includes junk pairs
# - *_pairs_filtered.pkl        → pre-dedup, includes cross-document duplicates
# - gap_pairs_*.pkl             → re-extracted from same chunks, high overlap


def find_pipeline_dir(explicit_path: str = None) -> Path:
    """Resolve the pipeline root directory."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified pipeline dir not found: {p}")

    env_path = os.getenv("PIPELINE_DIR")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        print(f"  ⚠ PIPELINE_DIR from .env not found: {p}")

    # Common locations to check
    candidates = [
        BASE_DIR.parent / "buffet-qa-pipeline",
        Path.home() / "OneDrive" / "Desktop" / "buffet-qa-pipeline",
        Path.home() / "Desktop" / "buffet-qa-pipeline",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Could not find pipeline directory. Either:\n"
        "  1. Set PIPELINE_DIR in .env\n"
        "  2. Pass --pipeline-dir /path/to/buffet-qa-pipeline"
    )


def sync_qa_csv(pipeline_dir: Path, dest_dir: Path) -> int:
    """
    Copy the SINGLE deduplicated QA CSV.
    Returns row count for verification.
    """
    source = pipeline_dir / "output" / QA_SOURCE_FILE
    if not source.exists():
        print(f"  ✗ QA CSV not found: {source}")
        print(f"    → Run notebook 06_assembly.ipynb first to generate this file")
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Remove any old CSVs to prevent loading stale per-document files
    old_csvs = list(dest_dir.glob("*.csv"))
    if old_csvs:
        for f in old_csvs:
            f.unlink()
        print(f"  ⟳ Cleared {len(old_csvs)} old CSV(s) from {dest_dir.name}/")

    dest = dest_dir / QA_SOURCE_FILE
    shutil.copy2(source, dest)

    # Count rows for verification
    row_count = sum(1 for _ in open(dest, encoding="utf-8")) - 1  # minus header
    print(f"  ✓ Copied {QA_SOURCE_FILE} → {row_count} QA pairs")
    return row_count


def sync_chunk_pickles(pipeline_dir: Path, dest_dir: Path) -> int:
    """
    Copy all *_classified.pkl files.
    Returns number of files copied.
    """
    source_dir = pipeline_dir / "intermediate"
    if not source_dir.exists():
        print(f"  ✗ Intermediate dir not found: {source_dir}")
        return 0

    pkl_files = list(source_dir.glob(CHUNK_PKL_PATTERN))
    if not pkl_files:
        print(f"  ✗ No {CHUNK_PKL_PATTERN} files found in {source_dir}")
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Clear old pickles
    old_pkls = list(dest_dir.glob("*.pkl"))
    if old_pkls:
        for f in old_pkls:
            f.unlink()
        print(f"  ⟳ Cleared {len(old_pkls)} old pickle(s) from {dest_dir.name}/")

    copied = 0
    for pkl in pkl_files:
        dest = dest_dir / pkl.name
        shutil.copy2(pkl, dest)
        size_kb = pkl.stat().st_size / 1024
        print(f"  ✓ {pkl.name} ({size_kb:.0f} KB)")
        copied += 1

    return copied


def clear_old_indices(index_dir: Path):
    """Remove stale FAISS indices so they get rebuilt on next server start."""
    index_dir.mkdir(parents=True, exist_ok=True)
    cleared = 0
    for f in index_dir.iterdir():
        f.unlink()
        cleared += 1
    if cleared:
        print(f"  ⟳ Cleared {cleared} old index file(s) — will rebuild on next start")
    else:
        print(f"  ℹ No existing indices to clear")


def rebuild_indices():
    """Run build_indices.py as a subprocess — full process isolation."""
    import subprocess
    print("\n═══ Rebuilding FAISS Indices ═══")
    script = BASE_DIR / "build_indices.py"
    if not script.exists():
        print(f"  ✗ {script.name} not found — indices will be built on server start")
        return
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")
    else:
        print(f"  ✗ Index rebuild failed:")
        for line in (result.stderr or result.stdout).strip().split("\n"):
            print(f"    {line}")
        print(f"    → Indices will be built on next server start instead")


def main():
    parser = argparse.ArgumentParser(description="Sync pipeline data into backend")
    parser.add_argument("--pipeline-dir", type=str, help="Path to buffet-qa-pipeline root")
    parser.add_argument("--no-rebuild", action="store_true", help="Skip FAISS rebuild")
    args = parser.parse_args()

    print("\n═══ Locating Pipeline ═══")
    try:
        pipeline_dir = find_pipeline_dir(args.pipeline_dir)
        print(f"  ✓ Found: {pipeline_dir}")
    except FileNotFoundError as exc:
        print(f"  ✗ {exc}")
        sys.exit(1)

    qa_dest = BASE_DIR / "data" / "csvs"
    chunk_dest = BASE_DIR / "data" / "chunks"
    index_dir = BASE_DIR / "data" / "indices"

    print("\n═══ Syncing QA Dataset ═══")
    qa_count = sync_qa_csv(pipeline_dir, qa_dest)

    print("\n═══ Syncing Chunk Pickles ═══")
    pkl_count = sync_chunk_pickles(pipeline_dir, chunk_dest)

    if qa_count == 0 and pkl_count == 0:
        print("\n⚠ No data synced. Check your pipeline directory.")
        sys.exit(1)

    print("\n═══ Clearing Old Indices ═══")
    clear_old_indices(index_dir)

    if not args.no_rebuild:
        rebuild_indices()

    print("\n═══ Summary ═══")
    print(f"  QA pairs:  {qa_count}")
    print(f"  Chunk PKLs: {pkl_count}")
    print(f"\n  Start the server:")
    print(f"  uvicorn main:app --reload --host 0.0.0.0 --port 8000\n")


if __name__ == "__main__":
    main()
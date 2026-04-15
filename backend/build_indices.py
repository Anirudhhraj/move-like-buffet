#!/usr/bin/env python3
"""
Standalone FAISS index builder.

Loads data from data/csvs and data/chunks, builds both indices,
saves to data/indices. No dependency on the rag package — imports
indexer.py directly as a file to avoid circular import chains.

Usage:
    python build_indices.py
"""

import importlib.util
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_module_from_file(name, filepath):
    """Import a .py file directly, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    sys.path.insert(0, str(BASE_DIR))

    indexer = _load_module_from_file(
        "rag_indexer", BASE_DIR / "rag" / "indexer.py"
    )

    print("--- Building FAISS Indices ---")
    print()
    stats = indexer.rebuild_all()
    print()
    print(f"OK qa_index={stats.get('qa_pairs', 0)} chunk_index={stats.get('chunks', 0)}")


if __name__ == "__main__":
    main()
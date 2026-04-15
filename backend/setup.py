#!/usr/bin/env python3
"""
First-run setup and validation.

Usage:
    python setup.py          # create directories, check deps
    python setup.py --check  # just validate without creating anything
"""

import importlib
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

REQUIRED_DIRS = [
    BASE_DIR / "data" / "csvs",
    BASE_DIR / "data" / "chunks",
    BASE_DIR / "data" / "indices",
]

REQUIRED_PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("openai", "openai"),
    ("sentence_transformers", "sentence-transformers"),
    ("faiss", "faiss-cpu"),
    ("yfinance", "yfinance"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("dotenv", "python-dotenv"),
    ("pydantic", "pydantic"),
]


def check_packages() -> list[str]:
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {pip_name}")
        except ImportError:
            print(f"  ✗ {pip_name} — NOT INSTALLED")
            missing.append(pip_name)
    return missing


def create_dirs():
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d.relative_to(BASE_DIR)}")


def check_env():
    env_file = BASE_DIR / ".env"
    example = BASE_DIR / ".env.example"
    if env_file.exists():
        print("  ✓ .env exists")
        # Check for placeholder values
        content = env_file.read_text()
        if "sk-your-deepseek-key" in content:
            print("  ⚠ .env still has placeholder DEEPSEEK_API_KEY — update it")
    elif example.exists():
        import shutil
        shutil.copy(example, env_file)
        print("  ✓ .env created from .env.example — edit it with your keys")
    else:
        print("  ✗ No .env or .env.example found")


def check_data():
    csvs = list((BASE_DIR / "data" / "csvs").glob("*.csv"))
    pkls = list((BASE_DIR / "data" / "chunks").glob("*.pkl"))
    indices = list((BASE_DIR / "data" / "indices").glob("*.index"))

    print(f"  CSV files:   {len(csvs)}")
    print(f"  Chunk PKLs:  {len(pkls)}")
    print(f"  FAISS indices: {len(indices)}")

    if not csvs:
        print("  ⚠ No CSV files in data/csvs/ — copy your pipeline output there")
    if not pkls:
        print("  ⚠ No pickle files in data/chunks/ — copy *_classified.pkl files there")
    if not indices:
        print("  ℹ No FAISS indices yet — they'll be built on first server start")


def check_syntax():
    """Import each module to catch syntax errors."""
    sys.path.insert(0, str(BASE_DIR))
    modules = ["config", "rag.indexer", "rag.retriever", "rag.router", "rag.agent",
               "stock.cache", "stock.analysis", "stock.endpoints"]
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"  ✓ {mod}")
        except Exception as exc:
            print(f"  ✗ {mod} — {exc}")


def main():
    check_only = "--check" in sys.argv

    print("\n═══ Package Check ═══")
    missing = check_packages()

    if not check_only:
        print("\n═══ Directory Setup ═══")
        create_dirs()

    print("\n═══ Environment ═══")
    check_env()

    print("\n═══ Data Files ═══")
    check_data()

    print("\n═══ Syntax Check ═══")
    check_syntax()

    if missing:
        print(f"\n⚠ Install missing packages:")
        print(f"  pip install {' '.join(missing)}")
    else:
        print("\n✓ All checks passed.")
        print("\nTo start the server:")
        print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000")

    print()


if __name__ == "__main__":
    main()
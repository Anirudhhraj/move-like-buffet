"""
Terminal chat with BuffettAgent.
Shows strategy, inline citation references, confidence, and timing.

Usage:  cd backend && python chat.py
Commands:  /quit  /clear  /stats
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from rag.agent import BuffettAgent


SEP = "=" * 70


def show_tools(tools):
    for t in tools:
        d = vars(t)
        parts = [f"{d.get('results_count', 0)} results"]
        if d.get("best_score") is not None:
            parts.append(f"best={d['best_score']:.3f}")
        if d.get("labels_hit"):
            parts.append(f"labels=[{', '.join(d['labels_hit'])}]")
        print(f"    {d.get('tool', '?')}: {', '.join(parts)}")


def show_sources(sources):
    if not sources:
        print("    (none)")
        return
    for s in sources:
        d = vars(s)
        idx = d.get("ref_idx", 0)
        stype = d.get("source_type", "?")
        label = d.get("label", "")
        sublabel = d.get("sublabel", "")
        sfile = d.get("source_file", "")
        section = d.get("source_section", "")
        sim = d.get("similarity", 0)
        qual = d.get("quality")

        loc = label
        if sublabel:
            loc += f" / {sublabel}"
        if sfile:
            loc += f"  ({sfile}"
            if section:
                loc += f" > {section[:60]}"
            loc += ")"

        scores = [f"sim={sim:.3f}"]
        if qual is not None:
            scores.append(f"quality={qual:.2f}")

        print(f"    [{idx}] {stype}: {loc}")
        print(f"        {', '.join(scores)}")


def main():
    print(f"\n{SEP}")
    print("  BUFFETT AGENT — Terminal Chat")
    print(SEP)
    print("\nLoading agent...\n")

    agent = BuffettAgent()
    s = agent.index_stats
    print(f"Ready: {s['qa_vectors']} QA pairs, {s['chunk_vectors']} chunks\n")
    print("Type a question. Commands: /quit  /clear  /stats\n")

    history = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("/quit", "/exit", "/q"):
            print("Bye.")
            break
        if query.lower() == "/clear":
            history.clear()
            print("History cleared.\n")
            continue
        if query.lower() == "/stats":
            s = agent.index_stats
            print(f"  QA: {s['qa_vectors']}  Chunks: {s['chunk_vectors']}\n")
            continue

        result = agent.answer(query, history=history)

        # Answer
        print(f"\n{SEP}\n")
        print(f"Buffett: {result.answer}\n")

        # Show enriched query if it differs from original
        if result.enriched_query:
            print(f"  Searched as: {result.enriched_query}")

        # Metadata
        print(f"  Strategy:    {result.strategy}")
        print(f"  Reason:      {result.reason}")
        print(f"  Confidence:  {result.confidence}")
        print(f"  Time:        {result.duration_ms}ms")

        # Tools (skip for rejections)
        if result.tools_called:
            print("\n  Retrieval:")
            show_tools(result.tools_called)

        # Sources with citation numbers matching the answer's [1], [2], etc.
        if result.sources:
            print("\n  Citation map:")
            show_sources(result.sources)

        print(f"\n{SEP}\n")

        # Update history for multi-turn
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": result.answer})


if __name__ == "__main__":
    main()
"""
Terminal chat with BuffettAgent.
Streams research events + LLM tokens live, validates citations.

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
    print("  BUFFETT AGENT — Terminal Chat (streaming)")
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

        # ── Streaming callbacks ──────────────────────────────────────
        token_count = 0

        def on_research(event):
            """Display research + citation events inline."""
            if event.step == "generating":
                # Signal: next output will be streamed tokens
                print(f"  [{event.step:>10}] {event.detail}", flush=True)
            elif event.step == "citation":
                # Citation issues appear after streaming finishes
                print(f"  [  citation] {event.detail}")
            else:
                print(f"  [{event.step:>10}] {event.detail}", flush=True)

        def on_token(token):
            """Print each token as it arrives from the LLM."""
            nonlocal token_count
            if token_count == 0:
                print(f"\nBuffett: ", end="", flush=True)
            print(token, end="", flush=True)
            token_count += 1

        # ── Call agent ───────────────────────────────────────────────
        print()  # blank line before events
        result = agent.answer(
            query,
            history=history,
            on_event=on_research,
            on_token=on_token,
        )

        # ── Display answer ───────────────────────────────────────────
        if token_count > 0:
            # Tokens were streamed — just add newline
            print("\n")
        else:
            # No streaming (rejection or error) — print answer directly
            print(f"\nBuffett: {result.answer}\n")

        # ── Metadata ─────────────────────────────────────────────────
        if result.enriched_query:
            print(f"  Searched as: {result.enriched_query}")

        print(f"  Strategy:    {result.strategy}")
        print(f"  Reason:      {result.reason}")
        print(f"  Confidence:  {result.confidence}")
        print(f"  Time:        {result.duration_ms}ms")

        if result.tools_called:
            print("\n  Initial retrieval:")
            show_tools(result.tools_called)

        if result.sources:
            print("\n  Citation map:")
            show_sources(result.sources)

        print(f"\n{SEP}\n")

        # ── Update history ───────────────────────────────────────────
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": result.answer})


if __name__ == "__main__":
    main()
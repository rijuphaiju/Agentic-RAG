"""
Batch Test Runner — Agentic RAG System
Reads all_test_questions.txt, runs every question through Stage 4,
and saves a full results report to test_results.json.

Usage:
    python run_tests.py
    python run_tests.py --questions "path/to/questions.txt"
    python run_tests.py --limit 20          # run first 20 questions only
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from Stage_1_RAG_Pipeline import (
    load_faiss_index, build_faiss_index, load_hotpotqa_passages,
    INDEX_PATH, PASSAGES_PATH,
)
from Stage_2_Verifier_GPU import load_verifier, VERIFIER_PATH
from Stage_4_Agentic_Loop import agentic_query

DEFAULT_QUESTIONS_FILE = r"c:\Users\pmans\OneDrive\Desktop\all_test_questions.txt"
OUTPUT_FILE = "test_results.json"


# ─────────────────────────────────────────────
# PARSE QUESTION FILE
# ─────────────────────────────────────────────
def parse_questions(filepath):
    """
    Returns a list of (category_name, question) tuples.
    Skips comment lines and blank lines.
    Groups questions under the most recent CATEGORY header comment.
    Deduplicates questions that appear in multiple categories.
    """
    entries = []
    seen = set()
    current_category = "General"

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                upper = line.upper()
                if "CATEGORY" in upper:
                    # e.g. "# CATEGORY 3: COMPARISON QUESTIONS"
                    parts = line.lstrip("# ").split(":", 1)
                    current_category = parts[1].strip() if len(parts) > 1 else parts[0].strip()
                continue
            # It's a question
            if line not in seen:
                seen.add(line)
                entries.append((current_category, line))

    return entries


# ─────────────────────────────────────────────
# RUN ALL QUESTIONS
# ─────────────────────────────────────────────
def run_all(entries, index, embedder, passages, verifier_model, verifier_tokenizer):
    results = []
    total = len(entries)
    category_stats = {}

    for i, (category, question) in enumerate(entries, 1):
        print(f"\n[{i}/{total}] [{category}]")
        print(f"  Q: {question}")

        t0 = time.time()
        try:
            result = agentic_query(
                question, index, embedder, passages,
                verifier_model, verifier_tokenizer,
                verbose=False,
            )
            elapsed = time.time() - t0
            entry = {
                "question":    question,
                "category":    category,
                "answer":      result["answer"],
                "status":      result["status"],
                "query_type":  result["query_type"],
                "iterations":  result["iterations"],
                "time_s":      round(elapsed, 2),
            }
            status_icon = {
                "SUPPORTED":   "✅",
                "PARTIAL":     "⚠️",
                "BEST_EFFORT": "🔁",
            }.get(result["status"], "❓")
            print(f"  {status_icon} {result['status']} | type={result['query_type']} "
                  f"| iter={result['iterations']} | {elapsed:.1f}s")
            print(f"  A: {result['answer']}")

        except Exception as exc:
            elapsed = time.time() - t0
            entry = {
                "question":   question,
                "category":   category,
                "answer":     f"ERROR: {exc}",
                "status":     "ERROR",
                "query_type": "UNKNOWN",
                "iterations": 0,
                "time_s":     round(elapsed, 2),
            }
            print(f"  ❌ ERROR: {exc}")

        results.append(entry)

        # Update per-category stats
        stats = category_stats.setdefault(category, {
            "total": 0, "SUPPORTED": 0, "PARTIAL": 0,
            "BEST_EFFORT": 0, "ERROR": 0,
        })
        stats["total"] += 1
        stats[entry["status"]] = stats.get(entry["status"], 0) + 1

    return results, category_stats


# ─────────────────────────────────────────────
# PRINT FINAL REPORT
# ─────────────────────────────────────────────
def print_report(results, category_stats):
    total = len(results)
    counts = {"SUPPORTED": 0, "PARTIAL": 0, "BEST_EFFORT": 0, "ERROR": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print(f"\n{'='*70}")
    print("OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"  Total questions : {total}")
    for status, n in counts.items():
        pct = n / total * 100 if total else 0
        print(f"  {status:<15}: {n:>3}  ({pct:.1f}%)")

    print(f"\n{'='*70}")
    print("PER-CATEGORY BREAKDOWN")
    print(f"{'='*70}")
    for cat, stats in category_stats.items():
        t = stats["total"]
        sup = stats.get("SUPPORTED", 0)
        par = stats.get("PARTIAL", 0)
        be  = stats.get("BEST_EFFORT", 0)
        err = stats.get("ERROR", 0)
        print(f"\n  {cat}")
        print(f"    Total={t}  SUPPORTED={sup}  PARTIAL={par}  BEST_EFFORT={be}  ERROR={err}")

    # List every BEST_EFFORT + ERROR for easy review
    print(f"\n{'='*70}")
    print("NEEDS REVIEW (BEST_EFFORT / ERROR)")
    print(f"{'='*70}")
    flagged = [r for r in results if r["status"] in ("BEST_EFFORT", "ERROR")]
    if not flagged:
        print("  None — all answers verified!")
    for r in flagged:
        print(f"\n  [{r['status']}] {r['question']}")
        print(f"    Answer: {r['answer']}")
        print(f"    Type: {r['query_type']} | Iterations: {r['iterations']}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS_FILE,
                        help="Path to the questions text file")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N questions (0 = all)")
    args = parser.parse_args()

    if not os.path.exists(args.questions):
        print(f"Questions file not found: {args.questions}")
        sys.exit(1)

    # ── Load resources ──
    print("Loading FAISS index...")
    if os.path.exists(INDEX_PATH) and os.path.exists(PASSAGES_PATH):
        index, embedder, passages = load_faiss_index()
    else:
        passages = load_hotpotqa_passages()
        index, embedder, passages = build_faiss_index(passages)

    print("Loading verifier...")
    if not os.path.exists(VERIFIER_PATH):
        print("Verifier not found. Run: python Stage_2_Verifier_GPU.py --mode train")
        sys.exit(1)
    verifier_model, verifier_tokenizer = load_verifier(VERIFIER_PATH)

    # ── Parse questions ──
    entries = parse_questions(args.questions)
    if args.limit:
        entries = entries[:args.limit]
    print(f"\n{len(entries)} unique questions loaded. Starting tests...\n")

    # ── Run ──
    t_start = time.time()
    results, category_stats = run_all(
        entries, index, embedder, passages, verifier_model, verifier_tokenizer
    )
    total_time = time.time() - t_start

    # ── Report ──
    print_report(results, category_stats)
    print(f"\nTotal time: {total_time/60:.1f} min")

    # ── Save ──
    output = {
        "timestamp":    datetime.now().isoformat(),
        "total":        len(results),
        "time_minutes": round(total_time / 60, 1),
        "summary":      {k: v for k, v in {
            "SUPPORTED":   sum(1 for r in results if r["status"] == "SUPPORTED"),
            "PARTIAL":     sum(1 for r in results if r["status"] == "PARTIAL"),
            "BEST_EFFORT": sum(1 for r in results if r["status"] == "BEST_EFFORT"),
            "ERROR":       sum(1 for r in results if r["status"] == "ERROR"),
        }.items()},
        "by_category":  category_stats,
        "results":      results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()

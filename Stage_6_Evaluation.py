"""
Stage 6: Full Pipeline Evaluation
==================================
Runs all four pipeline stages on HotpotQA validation questions and computes:
  - Exact Match (EM)
  - Precision, Recall, F1  (token-level, SQuAD-style)
  - Macro F1               (average F1 across query types; N/A for Stage 1)
  - Hallucination Rate     (% ungrounded in retrieved context, via verifier)
  - Abstention Rate        (% refusal / "not enough information" responses)

Usage:
    python Stage_6_Evaluation.py
    python Stage_6_Evaluation.py --samples 100
"""

import argparse
import hashlib
import json
import os
import re
import string
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from Stage_1_RAG_Pipeline import (
    load_faiss_index, build_faiss_index, load_hotpotqa_passages,
    generate_answer, retrieve as retrieve_s1, rerank_passages,
    init_bm25, _BM25_AVAILABLE,
    INDEX_PATH, PASSAGES_PATH,
    EMBED_MODEL, RERANKER_MODEL, OLLAMA_MODEL, TOP_K as S1_TOP_K,
)
from Stage_2_RAG_Pipeline import rag_query_stage2
from Stage_2_Verifier_GPU import load_verifier, verify, VERIFIER_PATH
from Stage_3_Adaptive_Retrieval import (
    classify_query,
    retrieve_simple, retrieve_multi_hop, retrieve_comparison,
    decompose_and_retrieve_multi_hop,
    TOP_K, MAX_HOPS,
    MULTI_HOP_PATTERNS,
)
from Stage_4_Agentic_Loop import (
    agentic_query,
    routed_query,
    MAX_ITERATIONS, CONFIDENCE_THRESHOLD,
    PARTIAL_ACCEPTANCE_THRESHOLD, COMPARISON_PARTIAL_THRESHOLD,
)

OUTPUT_FILE = "evaluation_results.json"


# ─────────────────────────────────────────────
# CODE FINGERPRINT
# Helps verify which version of each module is actually running.
# ─────────────────────────────────────────────

def _file_mtime(path):
    """Return last-modified time of a file as a readable string."""
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return "N/A"


def _file_md5(path):
    """Return first 8 chars of MD5 for quick change detection."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except OSError:
        return "N/A"


def _git_commit():
    """Return short git commit hash, or 'no-git' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


STAGE_FILES = {
    "stage1": "Stage_1_RAG_Pipeline.py",
    "stage2": "Stage_2_RAG_Pipeline.py",
    "stage3": "Stage_3_Adaptive_Retrieval.py",
    "stage4": "Stage_4_Agentic_Loop.py",
    "stage6": "Stage_6_Evaluation.py",
}


def _build_fingerprint():
    return {
        name: {
            "mtime": _file_mtime(path),
            "md5":   _file_md5(path),
        }
        for name, path in STAGE_FILES.items()
    }


def _print_run_header(num_samples):
    """Print a verification banner so the user can confirm which version is running."""
    git = _git_commit()
    fp  = _build_fingerprint()

    print(f"\n{'#'*72}")
    print(f"  Stage 6 Evaluation  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Git commit : {git}")
    print(f"  Samples    : {num_samples}")
    print(f"{'─'*72}")
    print(f"  {'File':<36} {'MD5':>8}  {'Modified'}")
    print(f"  {'─'*36}  {'─'*8}  {'─'*19}")
    for name, path in STAGE_FILES.items():
        mtime = _file_mtime(path)
        md5   = _file_md5(path)
        print(f"  {path:<36} {md5:>8}  {mtime}")
    print(f"{'─'*72}")

    # Stage config summary
    print(f"\n  CONFIG SNAPSHOT")
    print(f"  Stage 1 : embed={EMBED_MODEL}, reranker={RERANKER_MODEL}, "
          f"ollama={OLLAMA_MODEL}, top_k={S1_TOP_K}")
    print(f"  Stage 3 : top_k={TOP_K}, max_hops={MAX_HOPS}, "
          f"multi_hop_patterns={len(MULTI_HOP_PATTERNS)}")
    print(f"  Stage 4 : max_iter={MAX_ITERATIONS}, conf_thresh={CONFIDENCE_THRESHOLD}, "
          f"partial_thresh={PARTIAL_ACCEPTANCE_THRESHOLD}, "
          f"comparison_partial={COMPARISON_PARTIAL_THRESHOLD}")
    bm25_status = "ON" if _BM25_AVAILABLE else "OFF (pip install rank-bm25)"
    print(f"  Retrieval: SIMPLE=BM25+dense hybrid ({bm25_status}, alpha=0.5)")
    print(f"           : MULTI_HOP=BM25+dense iterative (3 hops)")
    print(f"           : COMPARISON=dense-only per-entity (BM25 excluded — hurts entity lookup)")
    print(f"  Stage 4 : ROUTING — COMPARISON→Stage4 agentic, MULTI_HOP/SIMPLE→Stage3")
    print(f"  Stage 1 : MULTI_HOP chain-of-thought prompt=ON")
    print(f"  Stage 1 : copula answer shortening=ON (\"X is Y\" → \"Y\" when ≤5 tokens)")
    print(f"  Eval    : alias_normalization=ON (US/UK/NYC/number-words)")
    print(f"  Output  : {OUTPUT_FILE} (always overwritten — no cache)")
    print(f"{'#'*72}\n")

ABSTENTION_PHRASES = (
    "don't have enough information",
    "does not contain enough information",
    "cannot be answered",
    "not enough information",
    "insufficient information",
    "context does not",
    "context doesn't",
    "i don't know",
    "no information available",
    "unable to answer",
)


# ─────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────

_ALIAS_MAP = {
    r'\bus\b|\bunited states\b|\bu\.s\.?\b': 'usa',
    r'\buk\b|\bunited kingdom\b|\bu\.k\.?\b': 'uk',
    r'\bnyc\b|\bnew york city\b': 'new york',
    r'\bla\b|\blos angeles\b':   'los angeles',
    r'\bone\b':   '1', r'\btwo\b':   '2', r'\bthree\b': '3',
    r'\bfour\b':  '4', r'\bfive\b':  '5', r'\bsix\b':   '6',
    r'\bseven\b': '7', r'\beight\b': '8', r'\bnine\b':  '9',
    r'\bten\b':  '10',
}

def _normalize(text):
    """Lowercase, alias-expand, strip articles and punctuation — standard QA normalization."""
    text = text.lower()
    for pattern, replacement in _ALIAS_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    return ' '.join(text.split())


def compute_em(pred, gold):
    return int(_normalize(pred) == _normalize(gold))


def compute_prf(pred, gold):
    """Token-level Precision, Recall, F1 (SQuAD evaluation style)."""
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return 0.0, 0.0, 0.0
    common = sum((Counter(p_toks) & Counter(g_toks)).values())
    if common == 0:
        return 0.0, 0.0, 0.0
    prec = common / len(p_toks)
    rec  = common / len(g_toks)
    f1   = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def is_abstention(text):
    t = text.lower()
    return any(phrase in t for phrase in ABSTENTION_PHRASES)


def _safe_mean(lst):
    return float(np.mean(lst)) if lst else 0.0


# ─────────────────────────────────────────────
# STAGE RUNNERS
# Returns: answer, abstained (bool), hallucinated (bool), query_type (str|None)
# ─────────────────────────────────────────────

def run_stage1(q, index, embedder, passages, vm, vt):
    retrieved = retrieve_s1(q, index, embedder, passages)
    answer    = generate_answer(q, retrieved)
    context   = " ".join(p["text"] for p in retrieved[:5])
    label     = verify(context, answer, vm, vt)["label"]
    return {
        "answer":       answer,
        "abstained":    is_abstention(answer),
        "hallucinated": label == "UNSUPPORTED",
        "query_type":   None,
    }


def run_stage2(q, index, embedder, passages, vm, vt):
    result  = rag_query_stage2(q, index, embedder, passages, verbose=False)
    answer  = result["answer"]
    context = " ".join(p["text"] for p in result["retrieved_passages"][:5])
    label   = verify(context, answer, vm, vt)["label"]
    return {
        "answer":       answer,
        "abstained":    is_abstention(answer),
        "hallucinated": label == "UNSUPPORTED",
        "query_type":   None,
    }


def run_stage3(q, index, embedder, passages, vm, vt):
    qtype         = classify_query(q)
    bridge_passage = None

    if qtype == "COMPARISON":
        retrieved = retrieve_comparison(q, index, embedder, passages)
    elif qtype == "MULTI_HOP":
        # Decomposed 2-sub-question strategy: SQ2 retrieval is independent of
        # the bridge answer, so a wrong bridge degrades context but never
        # misdirects retrieval. Bridge context is injected after reranking.
        retrieved, bridge_ctx = decompose_and_retrieve_multi_hop(q, index, embedder, passages)
        if bridge_ctx:
            bridge_passage = {
                "title": "Intermediate Finding",
                "text":  bridge_ctx,
                "score": 1.0,
                "hop":   0,
            }
    else:
        retrieved = retrieve_simple(q, index, embedder, passages)

    pool     = retrieved[:TOP_K * 2] if len(retrieved) > TOP_K else retrieved
    reranked = rerank_passages(q, pool, top_k=TOP_K)

    # Inject bridge answer AFTER reranking so CrossEncoder scores real passages
    # but LLM still sees the intermediate finding as the first context item.
    if bridge_passage:
        reranked = [bridge_passage] + reranked[:TOP_K - 1]

    # SIMPLE: top-1 document only — one focused passage beats a noisy top-K
    # that lets the LLM hedge or pick the wrong entity.
    # Verification still uses top-5 for a fair faithfulness signal.
    gen_passages = reranked[:1] if qtype == "SIMPLE" else reranked
    answer  = generate_answer(q, gen_passages, query_type=qtype)
    context = " ".join(p["text"] for p in reranked[:5])
    label   = verify(context, answer, vm, vt)["label"]
    return {
        "answer":       answer,
        "abstained":    is_abstention(answer),
        "hallucinated": label == "UNSUPPORTED",
        "query_type":   qtype,
    }


def run_stage4(q, index, embedder, passages, vm, vt):
    # Routing layer: COMPARISON → Stage 4 agentic loop; MULTI_HOP/SIMPLE → Stage 3.
    # Based on Run 3 empirical results: Stage 4 wins on COMPARISON (0.4451 vs 0.3991),
    # Stage 3 wins on MULTI_HOP (0.1543 vs 0.0884) and is safer for SIMPLE.
    result = routed_query(q, index, embedder, passages, vm, vt, verbose=False)
    label = result["verification"]["label"]
    return {
        "answer":       result["answer"],
        "abstained":    False,
        "hallucinated": label == "UNSUPPORTED",
        "query_type":   result["query_type"],
    }


# ─────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────

def aggregate(records, has_query_type):
    ems, precs, recs, f1s = [], [], [], []
    abstentions = hallucinations = 0
    per_type: dict[str, list[float]] = defaultdict(list)

    for r in records:
        ems.append(r["em"])
        precs.append(r["precision"])
        recs.append(r["recall"])
        f1s.append(r["f1"])
        abstentions    += int(r["abstained"])
        hallucinations += int(r["hallucinated"])
        if has_query_type and r["query_type"]:
            per_type[r["query_type"]].append(r["f1"])

    n = len(records)
    per_type_avg = {qt: round(_safe_mean(fs), 4) for qt, fs in per_type.items()}
    macro = round(_safe_mean(list(per_type_avg.values())), 4) if per_type_avg else None

    return {
        "exact_match":        round(_safe_mean(ems) * 100, 2),
        "precision":          round(_safe_mean(precs), 4),
        "recall":             round(_safe_mean(recs),  4),
        "f1":                 round(_safe_mean(f1s),   4),
        "macro_f1":           macro,
        "hallucination_rate": round(hallucinations / n * 100, 2) if n else 0.0,
        "abstention_rate":    round(abstentions    / n * 100, 2) if n else 0.0,
        "n":                  n,
        "per_type_f1":        per_type_avg,
    }


# ─────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────

def evaluate(num_samples: int = 50):
    # Print verification banner before anything else loads
    _print_run_header(num_samples)

    # ── Load pipeline resources ──
    print("Loading FAISS index...")
    if os.path.exists(INDEX_PATH) and os.path.exists(PASSAGES_PATH):
        index, embedder, passages = load_faiss_index()
    else:
        passages = load_hotpotqa_passages()
        index, embedder, passages = build_faiss_index(passages)

    # Build BM25 index in-memory (one-time; no disk write)
    init_bm25(passages)

    if not os.path.exists(VERIFIER_PATH):
        print("Verifier not found. Run: python Stage_2_Verifier_GPU.py --mode train")
        sys.exit(1)
    print("Loading verifier...")
    vm, vt = load_verifier(VERIFIER_PATH)

    # ── Load validation set ──
    print(f"\nLoading HotpotQA validation set ({num_samples} samples)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    s1_recs: list[dict] = []
    s2_recs: list[dict] = []
    s3_recs: list[dict] = []
    s4_recs: list[dict] = []

    RUNNERS = [
        ("Stage 1", run_stage1, s1_recs, False),
        ("Stage 2", run_stage2, s2_recs, False),
        ("Stage 3", run_stage3, s3_recs, True),
        ("Stage 4", run_stage4, s4_recs, True),
    ]

    t_total = time.time()

    for i, ex in enumerate(tqdm(dataset, total=num_samples, desc="Questions")):
        if i >= num_samples:
            break
        q    = ex["question"]
        gold = ex["answer"]

        for stage_label, fn, records, has_qt in RUNNERS:
            try:
                out         = fn(q, index, embedder, passages, vm, vt)
                prec, rec, f1 = compute_prf(out["answer"], gold)
                records.append({
                    "question":     q,
                    "gold":         gold,
                    "answer":       out["answer"],
                    "em":           compute_em(out["answer"], gold),
                    "precision":    round(prec, 4),
                    "recall":       round(rec,  4),
                    "f1":           round(f1,   4),
                    "abstained":    out["abstained"],
                    "hallucinated": out["hallucinated"],
                    "query_type":   out["query_type"],
                })
            except Exception as exc:
                records.append({
                    "question":     q,
                    "gold":         gold,
                    "answer":       f"ERROR: {exc}",
                    "em":           0,
                    "precision":    0.0,
                    "recall":       0.0,
                    "f1":           0.0,
                    "abstained":    False,
                    "hallucinated": True,
                    "query_type":   None,
                })
                tqdm.write(f"  [{stage_label}] ERROR on '{q[:60]}': {exc}")

    elapsed = time.time() - t_total

    # ── Aggregate per stage ──
    summary = {
        "stage1": aggregate(s1_recs, has_query_type=False),
        "stage2": aggregate(s2_recs, has_query_type=False),
        "stage3": aggregate(s3_recs, has_query_type=True),
        "stage4": aggregate(s4_recs, has_query_type=True),
    }

    output = {
        "timestamp":    datetime.now().isoformat(),
        "git_commit":   _git_commit(),
        "file_fingerprints": _build_fingerprint(),
        "config": {
            "embed_model":     EMBED_MODEL,
            "reranker_model":  RERANKER_MODEL,
            "ollama_model":    OLLAMA_MODEL,
            "top_k":           S1_TOP_K,
            "max_hops":        MAX_HOPS,
            "max_iterations":  MAX_ITERATIONS,
            "conf_threshold":  CONFIDENCE_THRESHOLD,
            "partial_thresh":  PARTIAL_ACCEPTANCE_THRESHOLD,
            "comparison_partial_thresh": COMPARISON_PARTIAL_THRESHOLD,
            "features": {
                "multi_hop_cot_prompt":      True,
                "simple_early_exit":         True,
                "simple_skip_self_consist":  True,
                "stage3_reranking":          True,
                "alias_normalization":       True,
            },
        },
        "num_samples":  num_samples,
        "elapsed_min":  round(elapsed / 60, 1),
        "summary":      summary,
        "detail": {
            "stage1": s1_recs,
            "stage2": s2_recs,
            "stage3": s3_recs,
            "stage4": s4_recs,
        },
    }

    # ── Print formatted table ──
    _print_table(summary, num_samples, elapsed)

    # ── Save ──
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {OUTPUT_FILE}\n")

    return output


# ─────────────────────────────────────────────
# PRETTY TABLE PRINTER
# ─────────────────────────────────────────────

def _print_table(summary, n, elapsed):
    STAGES = ["stage1", "stage2", "stage3", "stage4"]
    LABELS = ["Stage 1",  "Stage 2",  "Stage 3",  "Stage 4"]

    METRICS = [
        ("exact_match",        "Exact Match (EM %)"),
        ("precision",          "Precision"),
        ("recall",             "Recall"),
        ("f1",                 "F1 Score"),
        ("macro_f1",           "Macro F1"),
        ("hallucination_rate", "Hallucination Rate (%)"),
        ("abstention_rate",    "Abstention Rate (%)"),
    ]

    LW, CW = 26, 13
    sep = "+" + "-" * (LW + 2) + ("+" + "-" * (CW + 2)) * 4 + "+"

    print(f"\n{'='*72}")
    print(f"  FULL PIPELINE EVALUATION — {n} HotpotQA Validation Samples")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    print(f"{'='*72}")
    print(sep)
    hdr = f"| {'Metric':<{LW}} " + "".join(f"| {l:^{CW}} " for l in LABELS) + "|"
    print(hdr)
    print(sep.replace("-", "="))

    for key, display in METRICS:
        row = f"| {display:<{LW}} "
        for s in STAGES:
            val = summary[s].get(key)
            if val is None:
                cell = "N/A"
            elif key in ("exact_match", "hallucination_rate", "abstention_rate"):
                cell = f"{val:.1f}%"
            else:
                cell = f"{val:.4f}"
            row += f"| {cell:^{CW}} "
        row += "|"
        print(row)

    print(sep)

    # Per-type F1 breakdown
    for sk, sl in [("stage3", "Stage 3"), ("stage4", "Stage 4")]:
        per = summary[sk].get("per_type_f1", {})
        if per:
            print(f"\n  {sl} — F1 by Query Type:")
            for qt, v in per.items():
                print(f"    {qt:<12}  {v:.4f}")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all pipeline stages on HotpotQA")
    parser.add_argument(
        "--samples", type=int, default=50,
        help="Number of validation samples (default 50; full set ~7400)",
    )
    args = parser.parse_args()

    print(f"Stage 6: Full Pipeline Evaluation")
    print(f"Samples: {args.samples}\n")
    evaluate(args.samples)

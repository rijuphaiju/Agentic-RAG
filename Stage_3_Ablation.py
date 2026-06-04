"""
Stage 3 Ablation Study
======================
Measures the contribution of each Stage 3 component to EM / F1 / hallucination rate
on a stratified sample of HotpotQA validation questions.

Seven configurations tested on the SAME questions:

  A1  Stage 2 baseline          (simple retrieval + verifier, no adaptive strategy)
  A2  Stage 3 − entity retrieval (Phase 1 uses full-query only; no entity searches)
  A3  Stage 3 − expansion        (max_expansions=0 for all budgets)
  A4  Stage 3 − reformulation    (max_hops=1; skip Phase 2 iterative hops)
  A5  Stage 3 − type override    (use classifier; ignore HotpotQA dataset label)
  A6  Stage 3 − escalation gate  (always run MULTI_HOP; skip confidence check)
  A7  Stage 3 full               (all components active — current implementation)

Usage:
    python Stage_3_Ablation.py                    # 100 stratified questions
    python Stage_3_Ablation.py --samples 200      # larger sample
    python Stage_3_Ablation.py --configs A1,A7    # run only specific configs
    python Stage_3_Ablation.py --output ablation.json
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from datasets import load_dataset
from tqdm import tqdm

from Stage_1_RAG_Pipeline import (
    load_faiss_index, build_faiss_index, load_hotpotqa_passages,
    generate_answer, rerank_passages, retrieve_hybrid,
    init_bm25, INDEX_PATH, PASSAGES_PATH, OLLAMA_MODEL,
)
from Stage_2_RAG_Pipeline import rag_query_stage2
from Stage_2_Verifier_GPU import load_verifier, verify, build_verify_context, VERIFIER_PATH
from Stage_3_Adaptive_Retrieval import (
    classify_query, retrieve_simple, retrieve_multi_hop,
    retrieve_comparison, decompose_and_retrieve_multi_hop,
    rerank_passages as s3_rerank,
    adaptive_retrieve_with_coverage_check,
    estimate_retrieval_confidence,
    _retrieve_for_type, _BUDGET, _DEFAULT_BUDGET,
    TOP_K, LOW_SUPPORT_THRESHOLD,
    adaptive_rag_query,
)

OUTPUT_FILE = "ablation_results.json"


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    return ' '.join(text.split())


def compute_em(pred: str, gold: str) -> int:
    return int(_normalize(pred) == _normalize(gold))


def compute_f1(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return 0.0
    common = sum((Counter(p_toks) & Counter(g_toks)).values())
    if common == 0:
        return 0.0
    prec = common / len(p_toks)
    rec  = common / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def is_abstention(answer: str) -> bool:
    _pats = re.compile(
        r'\b(cannot provide|i cannot|do not appear|does not appear|'
        r'no information|not mentioned|not in the context|cannot be found|'
        r'not available|i don\'t have|does not contain|'
        r'not found|not determined|not specified|cannot determine)\b',
        re.IGNORECASE,
    )
    return bool(_pats.search(answer))


# ─────────────────────────────────────────────
# ABLATION RUNNERS
# Each runner returns: answer, hallucinated (bool), query_type_used
# ─────────────────────────────────────────────

def run_A1_stage2(q, index, embedder, passages, vm, vt, **kw):
    """Baseline: Stage 2 (simple retrieval + verifier, no adaptive strategy)."""
    result = rag_query_stage2(q, index, embedder, passages, vm, vt, verbose=False)
    label  = result["label"]
    return result["answer"], label in ("PARTIAL", "UNSUPPORTED"), "SIMPLE"


def run_A2_no_entity(q, index, embedder, passages, vm, vt,
                     hotpot_type=None, hotpot_level=None, **kw):
    """Stage 3 without entity-anchored Phase 1: only full-question search + hops."""
    if hotpot_type == "bridge":   query_type = "MULTI_HOP"
    elif hotpot_type == "comparison": query_type = "COMPARISON"
    else: query_type = classify_query(q)

    complexity = hotpot_level or "medium"
    budget = _BUDGET.get((query_type, complexity), _DEFAULT_BUDGET[query_type])

    if query_type == "MULTI_HOP":
        # Phase 1: only full-question search (no entity-specific calls)
        pool = list(retrieve_simple(q, index, embedder, passages, top_k=budget["top_k"]))
        # Phase 2: standard hops if anchor exists
        seen = {p["title"] for p in pool}
        if budget["max_hops"] > 1 and pool:
            from Stage_3_Adaptive_Retrieval import _reformulate_query
            top_text = pool[0]["text"][:300]
            hop_q    = _reformulate_query(q, top_text)
            if hop_q.strip().lower() != q.strip().lower():
                for p in retrieve_simple(hop_q, index, embedder, passages,
                                         top_k=budget["top_k"] // 2):
                    if p["title"] not in seen:
                        pool.append(p)
                        seen.add(p["title"])
        bridge_ctx = None
    elif query_type == "COMPARISON":
        pool = retrieve_comparison(q, index, embedder, passages, top_k=budget["top_k"])
        bridge_ctx = None
    else:
        pool = retrieve_simple(q, index, embedder, passages, top_k=budget["top_k"])
        bridge_ctx = None

    reranked = rerank_passages(q, pool, top_k=TOP_K)
    ctx_passages = list(reranked)
    if bridge_ctx:
        ctx_passages = [{"title": "Bridge", "text": bridge_ctx}] + ctx_passages
    answer = generate_answer(q, ctx_passages, query_type=query_type)

    ctx  = build_verify_context(reranked, answer)
    verif = verify(ctx, answer, vm, vt, question=q)
    return answer, verif["label"] in ("PARTIAL", "UNSUPPORTED"), query_type


def run_A3_no_expansion(q, index, embedder, passages, vm, vt,
                        hotpot_type=None, hotpot_level=None, **kw):
    """Stage 3 without coverage expansion (max_expansions=0)."""
    return _run_with_budget_override(
        q, index, embedder, passages, vm, vt,
        hotpot_type, hotpot_level,
        budget_override={"max_expansions": 0},
    )


def run_A4_no_reformulation(q, index, embedder, passages, vm, vt,
                            hotpot_type=None, hotpot_level=None, **kw):
    """Stage 3 without iterative reformulation (max_hops=1)."""
    return _run_with_budget_override(
        q, index, embedder, passages, vm, vt,
        hotpot_type, hotpot_level,
        budget_override={"max_hops": 1},
    )


def run_A5_no_override(q, index, embedder, passages, vm, vt,
                       hotpot_type=None, hotpot_level=None, **kw):
    """Stage 3 without dataset type override: always use classify_query()."""
    result = adaptive_rag_query(
        q, index, embedder, passages, vm, vt,
        verbose=False,
        query_type_override=None,     # ← disables dataset override
        level_override=hotpot_level,
    )
    verif = result.get("verification") or {}
    label = verif.get("label", "UNSUPPORTED")
    return result["answer"], label in ("PARTIAL", "UNSUPPORTED"), result["query_type"]


def run_A6_no_escalation(q, index, embedder, passages, vm, vt,
                         hotpot_type=None, hotpot_level=None, **kw):
    """Stage 3 without confidence-gated escalation: always run full MULTI_HOP."""
    if hotpot_type == "bridge":   query_type = "MULTI_HOP"
    elif hotpot_type == "comparison": query_type = "COMPARISON"
    else: query_type = classify_query(q)

    complexity = hotpot_level or "medium"
    budget = _BUDGET.get((query_type, complexity), _DEFAULT_BUDGET[query_type])

    # Skip the escalation gate — always run full multi-hop
    pool, reranked, bridge_ctx, _ = adaptive_retrieve_with_coverage_check(
        q, query_type, index, embedder, passages, budget, False,
    )
    ctx_passages = list(reranked)
    if bridge_ctx:
        ctx_passages = [{"title": "Bridge", "text": bridge_ctx}] + ctx_passages
    answer = generate_answer(q, ctx_passages, query_type=query_type)

    ctx   = build_verify_context(reranked, answer)
    verif = verify(ctx, answer, vm, vt, question=q)
    return answer, verif["label"] in ("PARTIAL", "UNSUPPORTED"), query_type


def run_A7_full(q, index, embedder, passages, vm, vt,
                hotpot_type=None, hotpot_level=None, **kw):
    """Stage 3 with all components active (current implementation)."""
    result = adaptive_rag_query(
        q, index, embedder, passages, vm, vt,
        verbose=False,
        query_type_override=hotpot_type,
        level_override=hotpot_level,
    )
    verif = result.get("verification") or {}
    label = verif.get("label", "UNSUPPORTED")
    return result["answer"], label in ("PARTIAL", "UNSUPPORTED"), result["query_type"]


# ─────────────────────────────────────────────
# HELPER: run with budget override
# ─────────────────────────────────────────────

def _run_with_budget_override(q, index, embedder, passages, vm, vt,
                               hotpot_type, hotpot_level, budget_override):
    if hotpot_type == "bridge":       query_type = "MULTI_HOP"
    elif hotpot_type == "comparison": query_type = "COMPARISON"
    else:                             query_type = classify_query(q)

    complexity = hotpot_level or "medium"
    budget = dict(_BUDGET.get((query_type, complexity), _DEFAULT_BUDGET[query_type]))
    budget.update(budget_override)

    pool, reranked, bridge_ctx, _ = adaptive_retrieve_with_coverage_check(
        q, query_type, index, embedder, passages, budget, False,
    )
    ctx_passages = list(reranked)
    if bridge_ctx:
        ctx_passages = [{"title": "Bridge", "text": bridge_ctx}] + ctx_passages
    answer = generate_answer(q, ctx_passages, query_type=query_type)

    ctx   = build_verify_context(reranked, answer)
    verif = verify(ctx, answer, vm, vt, question=q)
    return answer, verif["label"] in ("PARTIAL", "UNSUPPORTED"), query_type


# ─────────────────────────────────────────────
# CONFIGS REGISTRY
# ─────────────────────────────────────────────

ALL_CONFIGS = {
    "A1": ("Stage 2 baseline",              run_A1_stage2,         False),
    "A2": ("Stage 3 − entity retrieval",    run_A2_no_entity,      True),
    "A3": ("Stage 3 − expansion",           run_A3_no_expansion,   True),
    "A4": ("Stage 3 − reformulation",       run_A4_no_reformulation, True),
    "A5": ("Stage 3 − type override",       run_A5_no_override,    True),
    "A6": ("Stage 3 − escalation gate",     run_A6_no_escalation,  True),
    "A7": ("Stage 3 full (all components)", run_A7_full,           True),
}


# ─────────────────────────────────────────────
# AGGREGATE
# ─────────────────────────────────────────────

def aggregate(records):
    if not records:
        return {}
    ems  = [r["em"]  for r in records]
    f1s  = [r["f1"]  for r in records]
    hals = [r["hallucinated"] for r in records]
    n = len(records)

    # per-type F1
    per_type = defaultdict(list)
    for r in records:
        per_type[r["query_type"]].append(r["f1"])
    per_type_f1 = {qt: round(sum(v)/len(v), 4) for qt, v in per_type.items()}
    macro = round(sum(per_type_f1.values()) / len(per_type_f1), 4) if per_type_f1 else None

    return {
        "n":                  n,
        "exact_match_pct":    round(sum(ems) / n * 100, 2),
        "f1":                 round(sum(f1s) / n,       4),
        "macro_f1":           macro,
        "hallucination_pct":  round(sum(hals) / n * 100, 2),
        "per_type_f1":        per_type_f1,
    }


# ─────────────────────────────────────────────
# STRATIFIED SAMPLING
# ─────────────────────────────────────────────

def stratified_sample(dataset, n: int):
    """
    Return up to n questions stratified by (type × level).
    Bridges: 60%, Comparisons: 40% (matches HotpotQA distribution).
    Within bridges: easy/medium/hard proportional.
    """
    buckets = defaultdict(list)
    for ex in dataset:
        buckets[(ex["type"], ex["level"])].append(ex)

    n_bridge     = int(n * 0.60)
    n_comparison = n - n_bridge

    sample = []
    for ds_type, count in [("bridge", n_bridge), ("comparison", n_comparison)]:
        levels    = [l for (t, l) in buckets if t == ds_type]
        per_level = max(1, count // len(levels)) if levels else count
        for level in sorted(set(levels)):
            pool = buckets[(ds_type, level)]
            sample.extend(pool[:per_level])
        # Fill remaining slots if rounding left gaps
        all_of_type = [ex for (t, _), pool in buckets.items()
                       if t == ds_type for ex in pool]
        seen_qs = {ex["question"] for ex in sample}
        for ex in all_of_type:
            if len([s for s in sample if s["type"] == ds_type]) >= count:
                break
            if ex["question"] not in seen_qs:
                sample.append(ex)
                seen_qs.add(ex["question"])

    return sample[:n]


# ─────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────

def run_ablation(num_samples: int = 100, config_keys: list = None,
                 output_file: str = OUTPUT_FILE):
    if config_keys is None:
        config_keys = list(ALL_CONFIGS.keys())

    configs = {k: ALL_CONFIGS[k] for k in config_keys if k in ALL_CONFIGS}
    if not configs:
        print(f"No valid configs in: {config_keys}")
        return

    print(f"Loading FAISS index...")
    if os.path.exists(INDEX_PATH) and os.path.exists(PASSAGES_PATH):
        index, embedder, passages = load_faiss_index()
    else:
        passages = load_hotpotqa_passages()
        index, embedder, passages = build_faiss_index(passages)
    init_bm25(passages)

    if not os.path.exists(VERIFIER_PATH):
        print("Verifier not found. Run: python Stage_2_Verifier_GPU.py --mode train")
        sys.exit(1)
    print("Loading verifier...")
    vm, vt = load_verifier(VERIFIER_PATH)

    print(f"Loading HotpotQA validation ({num_samples} stratified samples)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    sample  = stratified_sample(list(dataset), num_samples)

    type_dist = Counter(ex["type"] for ex in sample)
    print(f"Sample distribution: {dict(type_dist)}")

    all_records = {k: [] for k in configs}
    t_start = time.time()

    for ex in tqdm(sample, desc="Questions"):
        q, gold = ex["question"], ex["answer"]
        ht, hl  = ex.get("type"), ex.get("level")

        for cfg_key, (label, runner, needs_type) in configs.items():
            kw = {"hotpot_type": ht, "hotpot_level": hl} if needs_type else {}
            try:
                answer, hallucinated, qtype = runner(
                    q, index, embedder, passages, vm, vt, **kw
                )
            except Exception as exc:
                tqdm.write(f"  [{cfg_key}] ERROR on '{q[:55]}': {exc}")
                answer, hallucinated, qtype = f"ERROR: {exc}", True, ht or "UNKNOWN"

            all_records[cfg_key].append({
                "question":    q,
                "gold":        gold,
                "answer":      answer,
                "em":          compute_em(answer, gold),
                "f1":          round(compute_f1(answer, gold), 4),
                "hallucinated": hallucinated,
                "query_type":  qtype,
                "hotpot_type": ht,
                "hotpot_level": hl,
            })

    elapsed = time.time() - t_start

    # ── Aggregate ──
    summary = {}
    for cfg_key, (label, _, _) in configs.items():
        summary[cfg_key] = {"label": label, **aggregate(all_records[cfg_key])}

    _print_table(summary, num_samples, elapsed)

    output = {
        "timestamp":    datetime.now().isoformat(),
        "num_samples":  num_samples,
        "elapsed_min":  round(elapsed / 60, 1),
        "sample_dist":  dict(type_dist),
        "configs":      {k: v[0] for k, v in configs.items()},
        "summary":      summary,
        "detail":       all_records,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {output_file}")
    return output


# ─────────────────────────────────────────────
# PRETTY TABLE
# ─────────────────────────────────────────────

def _print_table(summary, n, elapsed):
    print(f"\n{'='*90}")
    print(f"  STAGE 3 ABLATION STUDY — {n} HotpotQA Validation Samples  "
          f"(elapsed: {elapsed/60:.1f} min)")
    print(f"{'='*90}")

    HDR = f"{'Config':<6} {'Description':<36} {'EM%':>6} {'F1':>7} "
    HDR += f"{'MacroF1':>8} {'Halluc%':>8}"
    print(HDR)
    print("-" * 90)

    # Sort by F1 descending
    rows = sorted(summary.items(), key=lambda kv: kv[1].get("f1", 0), reverse=True)
    for cfg_key, s in rows:
        em   = s.get("exact_match_pct", 0)
        f1   = s.get("f1", 0)
        mf1  = s.get("macro_f1") or 0
        hal  = s.get("hallucination_pct", 0)
        lbl  = s.get("label", "")[:35]
        print(f"{cfg_key:<6} {lbl:<36} {em:>5.1f}% {f1:>7.4f} "
              f"{mf1:>8.4f} {hal:>7.1f}%")

    print("-" * 90)
    print("\nPer-type F1 breakdown:")
    for cfg_key, s in rows:
        ptf = s.get("per_type_f1", {})
        if ptf:
            breakdown = "  ".join(f"{qt}={v:.4f}" for qt, v in sorted(ptf.items()))
            print(f"  {cfg_key}: {breakdown}")

    print()
    print("Interpretation guide:")
    print("  A7 − A1 = total Stage 3 gain over Stage 2 baseline")
    print("  A7 − A2 = value added by entity-anchored Phase 1 retrieval")
    print("  A7 − A3 = value added by coverage expansion")
    print("  A7 − A4 = value added by iterative reformulation")
    print("  A7 − A5 = value added by dataset type override vs classifier")
    print("  A7 − A6 = value added by confidence-gated escalation")
    print("  Negative delta = component HURTS performance (should be restricted/removed)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 ablation study on HotpotQA")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of stratified samples (default 100)")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config keys, e.g. A1,A5,A7")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help=f"Output JSON file (default {OUTPUT_FILE})")
    args = parser.parse_args()

    cfg_keys = [c.strip() for c in args.configs.split(",")] if args.configs else None
    run_ablation(num_samples=args.samples, config_keys=cfg_keys,
                 output_file=args.output)

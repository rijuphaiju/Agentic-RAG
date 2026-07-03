"""
Stage 6: Full Pipeline Evaluation Framework
=============================================
Runs all four pipeline stages on HotpotQA validation questions under the
official distractor protocol (every question builds its own temporary corpus
via build_example_corpus() — no global FAISS/BM25 — used identically by all
four stages, then discarded before the next question) and produces a
research-grade evaluation report.

Core metrics (unchanged from the original evaluation):
  - Exact Match (EM)
  - Precision, Recall, F1  (token-level, SQuAD-style)
  - Macro F1               (average F1 across query types; N/A for Stage 1/2)
  - Hallucination Rate     (PARTIAL/UNSUPPORTED per Eq. 6.7)
  - Abstention Rate

Extended metrics: ROC-AUC / PR-AUC of verifier confidence vs. actual
correctness, confusion matrices, difficulty-wise and query-type-wise
breakdowns, stage-to-stage improvement deltas, per-stage latency, Stage 3
retrieval statistics (pipeline usage, coverage before/after expansion), and
Stage 4 agent statistics (action distribution, recovery rate).

Structure: metric computation / stage execution / aggregation / reporting /
JSON export are kept as separate, independently testable functions rather
than one large procedural loop.

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
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from Stage_1_RAG_Pipeline import (
    build_example_corpus,
    rag_query as s1_rag_query,
    EMBED_MODEL, RERANKER_MODEL, OLLAMA_MODEL, TOP_K as S1_TOP_K,
)
from Stage_2_Verifier import load_verifier, verify, legacy_scores, VERIFIER_PATH
from Stage_3_Adaptive_Retrieval import (
    adaptive_rag_query,
    TOP_K, MAX_HOPS, MULTI_HOP_PATTERNS,
)
from Stage_4_Agentic_Loop import (
    agentic_query,
    MAX_ACTIONS, CONFIDENCE_THRESHOLD,
)

OUTPUT_FILE = "evaluation_results.json"

_LABELS = ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
_STAGES = ["stage1", "stage2", "stage3", "stage4"]
_STAGE_LABELS = {"stage1": "Stage 1", "stage2": "Stage 2", "stage3": "Stage 3", "stage4": "Stage 4"}


# ═════════════════════════════════════════════
# CODE FINGERPRINT
# Helps verify which version of each module is actually running.
# ═════════════════════════════════════════════

def _file_mtime(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return "N/A"


def _file_md5(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except OSError:
        return "N/A"


def _git_commit() -> str:
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
    "stage2": "Stage_2_Verifier.py",
    "stage3": "Stage_3_Adaptive_Retrieval.py",
    "stage4": "Stage_4_Agentic_Loop.py",
    "stage6": "Stage_6_Evaluation.py",
}


def _build_fingerprint() -> dict:
    return {name: {"mtime": _file_mtime(path), "md5": _file_md5(path)}
            for name, path in STAGE_FILES.items()}


def _print_run_header(num_samples: int) -> None:
    """Print a verification banner so the user can confirm which version is running."""
    git = _git_commit()

    print(f"\n{'#'*72}")
    print(f"  Stage 6 Evaluation  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Git commit : {git}")
    print(f"  Samples    : {num_samples}")
    print(f"{'─'*72}")
    print(f"  {'File':<36} {'MD5':>8}  {'Modified'}")
    print(f"  {'─'*36}  {'─'*8}  {'─'*19}")
    for name, path in STAGE_FILES.items():
        print(f"  {path:<36} {_file_md5(path):>8}  {_file_mtime(path)}")
    print(f"{'─'*72}")

    print(f"\n  CONFIG SNAPSHOT")
    print(f"  Architecture : official HotpotQA distractor protocol — every question "
          f"builds its own temporary corpus (build_example_corpus); no global FAISS/BM25.")
    print(f"  Stage 1 : embed={EMBED_MODEL}, reranker={RERANKER_MODEL}, "
          f"ollama={OLLAMA_MODEL}, top_k={S1_TOP_K}")
    print(f"  Stage 3 : top_k={TOP_K}, max_hops={MAX_HOPS}, "
          f"nine adaptive pipelines (type × difficulty), "
          f"multi_hop_patterns={len(MULTI_HOP_PATTERNS)}")
    print(f"  Stage 4 : max_actions={MAX_ACTIONS}, conf_thresh={CONFIDENCE_THRESHOLD}, "
          f"diagnosis-driven agent (KEEP/EXPAND/COMPARE/DECOMPOSE/REWRITE/ABSTAIN)")
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
    "cannot confidently answer",
)


# ═════════════════════════════════════════════
# METRIC COMPUTATION
# ═════════════════════════════════════════════

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


def _normalize(text: str) -> str:
    """Lowercase, alias-expand, strip articles and punctuation — standard QA normalization."""
    text = text.lower()
    for pattern, replacement in _ALIAS_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    return ' '.join(text.split())


def compute_em(pred: str, gold: str) -> int:
    return int(_normalize(pred) == _normalize(gold))


def compute_prf(pred: str, gold: str) -> tuple:
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


def is_abstention(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in ABSTENTION_PHRASES)


def _safe_mean(lst: list) -> float:
    return float(np.mean(lst)) if lst else 0.0


def compute_roc_pr_auc(records: list) -> dict:
    """
    ROC-AUC / PR-AUC of the verifier's own P(SUPPORTED) score as a predictor
    of ACTUAL answer correctness (Exact Match against the HotpotQA gold
    answer) — i.e. how well verifier confidence tracks real correctness,
    not just the verifier's internal self-consistency. HotpotQA has no
    independent SUPPORTED/PARTIAL/UNSUPPORTED ground truth, so EM is used as
    the correctness proxy, matching the same ground-truth convention this
    project already uses for hallucination-rate scoring.

    Returns {"roc_auc": None, "pr_auc": None, "n": ...} (not a fabricated
    number) when fewer than 2 distinct EM classes are present — sklearn's
    metrics are undefined in that degenerate case.
    """
    y_true, y_score = [], []
    for r in records:
        if r.scores and "SUPPORTED" in r.scores:
            y_true.append(r.em)
            y_score.append(r.scores["SUPPORTED"])

    if len(set(y_true)) < 2:
        return {"roc_auc": None, "pr_auc": None, "n": len(y_true)}

    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc":  round(float(average_precision_score(y_true, y_score)), 4),
        "n":       len(y_true),
    }


def compute_confusion_matrix(records: list) -> dict:
    """
    Verifier label cross-tabulated against actual correctness (EM) — the
    closest meaningful "confusion matrix" available given that HotpotQA has
    no independent SUPPORTED/PARTIAL/UNSUPPORTED ground truth to compare
    against. Answers where the verifier said label X are split into
    "actually correct" (em=1) vs. "actually incorrect" (em=0) counts.
    """
    matrix = {lbl: {"em_correct": 0, "em_incorrect": 0} for lbl in _LABELS}
    for r in records:
        if r.label in matrix:
            matrix[r.label]["em_correct" if r.em else "em_incorrect"] += 1
    return matrix


# ═════════════════════════════════════════════
# PER-EXAMPLE RECORD
# One dataclass shared by all four stages — fields that don't apply to a
# given stage (e.g. Stage 3's `pipeline`, Stage 4's `actions_selected`) stay
# None rather than needing four separate record types.
# ═════════════════════════════════════════════

@dataclass
class EvalRecord:
    question: str
    gold: str
    answer: str
    em: int
    precision: float
    recall: float
    f1: float
    abstained: bool
    hallucinated: bool

    query_type: str | None = None
    level: str | None = None
    label: str | None = None                 # verifier label: SUPPORTED/PARTIAL/UNSUPPORTED
    scores: dict | None = None                # verifier probabilities for this example
    latency: dict | None = None
    num_retrieved: int | None = None

    # Stage 3 only
    pipeline: str | None = None
    coverage: float | None = None
    coverage_before: float | None = None
    expanded: bool | None = None

    # Stage 4 only
    status: str | None = None                 # SUPPORTED / BEST_EFFORT / ABSTAINED
    iterations: int | None = None
    actions_selected: list | None = None
    recovered_via: str | None = None

    error: str | None = None                  # populated only if the stage raised an exception


def _error_record(example: dict, exc: Exception) -> EvalRecord:
    return EvalRecord(
        question=example["question"], gold=example["answer"],
        answer=f"ERROR: {exc}", em=0, precision=0.0, recall=0.0, f1=0.0,
        abstained=False, hallucinated=True, level=example.get("level"),
        error=str(exc),
    )


# ═════════════════════════════════════════════
# STAGE EXECUTION
# Every runner uses the SAME per-question corpus (built once, shared across
# all four stages for that question) — no global FAISS/BM25 anywhere.
# ═════════════════════════════════════════════

def run_stage1(example: dict, ex_index, embedder, ex_passages, ex_bm25, vm, vt) -> EvalRecord:
    t0 = time.time()
    result = s1_rag_query(example["question"], ex_index, embedder, ex_passages, bm25=ex_bm25)
    t1 = time.time()
    answer = result["answer"]
    verification = verify(example["question"], answer, result["retrieved_passages"], vm)
    t2 = time.time()

    label = verification["overall_status"]
    scores = legacy_scores(label, verification["support_score"])
    prec, rec, f1 = compute_prf(answer, example["answer"])
    return EvalRecord(
        question=example["question"], gold=example["answer"], answer=answer,
        em=compute_em(answer, example["answer"]),
        precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
        abstained=is_abstention(answer),
        hallucinated=label in ("PARTIAL", "UNSUPPORTED"),
        level=example.get("level"), label=label, scores=scores,
        num_retrieved=len(result["retrieved_passages"]),
        latency={"pipeline": round(t1 - t0, 4), "verify": round(t2 - t1, 4), "total": round(t2 - t0, 4)},
    )


def run_stage2(s1_record: EvalRecord) -> EvalRecord:
    """
    Stage 2 = Stage 1's own retrieval + generation, with the verifier's label
    as its contribution — Stage 2 does not run separate retrieval (matches
    the established convention already used by Stage 4/5, since re-retrieving
    would make Stage 2 incomparable to Stage 1 on the same evidence).
    """
    return EvalRecord(
        question=s1_record.question, gold=s1_record.gold, answer=s1_record.answer,
        em=s1_record.em, precision=s1_record.precision, recall=s1_record.recall, f1=s1_record.f1,
        abstained=s1_record.abstained, hallucinated=s1_record.hallucinated,
        level=s1_record.level, label=s1_record.label, scores=s1_record.scores,
        num_retrieved=s1_record.num_retrieved, latency=s1_record.latency,
    )


def run_stage3(example: dict, ex_index, embedder, ex_passages, ex_bm25, vm, vt) -> EvalRecord:
    t0 = time.time()
    result = adaptive_rag_query(
        example["question"], ex_index, embedder, ex_passages, vm, vt, verbose=False,
        query_type_override=example.get("type"), level_override=example.get("level"),
        bm25=ex_bm25,
    )
    t1 = time.time()

    answer = result["answer"]
    # Stage 3 now returns Stage 2 V2's native report directly (overall_status/
    # support_score, no more legacy label/scores) — same shape Stage 4 already
    # returns, so both stages are read identically here.
    verification = result.get("verification") or {"overall_status": "UNSUPPORTED", "support_score": 0.0}
    label = verification.get("overall_status", "UNSUPPORTED")
    scores = legacy_scores(label, verification.get("support_score", 0.0))
    stats = result.get("retrieval_stats") or {}
    prec, rec, f1 = compute_prf(answer, example["answer"])

    return EvalRecord(
        question=example["question"], gold=example["answer"], answer=answer,
        em=compute_em(answer, example["answer"]),
        precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
        abstained=is_abstention(answer),
        hallucinated=label in ("PARTIAL", "UNSUPPORTED"),
        query_type=result.get("query_type"), level=result.get("level"),
        label=label, scores=scores,
        num_retrieved=result.get("num_retrieved"),
        pipeline=result.get("retrieval_strategy"),
        coverage=stats.get("coverage"), coverage_before=stats.get("coverage_before"),
        expanded=stats.get("expanded"),
        latency={"total": round(t1 - t0, 4)},
    )


def run_stage4(example: dict, ex_index, embedder, ex_passages, ex_bm25, vm, vt) -> EvalRecord:
    t0 = time.time()
    result = agentic_query(
        example["question"], ex_index, embedder, ex_passages, vm, vt, verbose=False,
        query_type_override=example.get("type"), level_override=example.get("level"),
        bm25=ex_bm25,
    )
    t1 = time.time()

    answer = result["answer"]
    # Stage 4 returns Stage 2 V2's report shape directly (overall_status/
    # support_score, not label/scores) — {} when every action in the
    # episode was UNSUPPORTED and the agent abstained from the start.
    verification = result.get("verification") or {}
    label = verification.get("overall_status", "UNSUPPORTED")
    scores = legacy_scores(label, verification.get("support_score", 0.0))
    abstained = result.get("abstained", False)
    prec, rec, f1 = compute_prf(answer, example["answer"])

    # Hallucination per Eq. 6.7: PARTIAL/UNSUPPORTED count as hallucinations;
    # abstentions do not (no answer returned). BEST_EFFORT (a PARTIAL best
    # candidate, per the agent's explicit PARTIAL-handling policy) still
    # counts — only a true "SUPPORTED" status or an abstention does not.
    hallucinated = (not abstained) and (result["status"] != "SUPPORTED")

    return EvalRecord(
        question=example["question"], gold=example["answer"], answer=answer,
        em=compute_em(answer, example["answer"]),
        precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
        abstained=abstained, hallucinated=hallucinated,
        query_type=result.get("query_type"), level=result.get("level"),
        label=label, scores=scores,
        status=result.get("status"), iterations=result.get("iterations"),
        actions_selected=result.get("actions_selected"), recovered_via=result.get("recovered_via"),
        latency={"total": round(t1 - t0, 4)},
    )


# ═════════════════════════════════════════════
# AGGREGATION
# ═════════════════════════════════════════════

def _metrics_from_records(records: list) -> dict:
    """Core EM/Precision/Recall/F1/Hallucination/Abstention computation,
    shared by overall, difficulty-wise, and query-type-wise aggregation so
    it's never duplicated."""
    n = len(records)
    ems   = [r.em for r in records]
    precs = [r.precision for r in records]
    recs  = [r.recall for r in records]
    f1s   = [r.f1 for r in records]
    abstentions    = sum(int(r.abstained) for r in records)
    hallucinations = sum(int(r.hallucinated) for r in records)

    return {
        "exact_match":        round(_safe_mean(ems) * 100, 2),
        "precision":          round(_safe_mean(precs), 4),
        "recall":             round(_safe_mean(recs), 4),
        "f1":                 round(_safe_mean(f1s), 4),
        "hallucination_rate": round(hallucinations / n * 100, 2) if n else 0.0,
        "abstention_rate":    round(abstentions / n * 100, 2) if n else 0.0,
        "n": n,
    }


def aggregate_overall(records: list) -> dict:
    metrics = _metrics_from_records(records)

    per_type = defaultdict(list)
    for r in records:
        if r.query_type:
            per_type[r.query_type].append(r.f1)
    per_type_avg = {qt: round(_safe_mean(fs), 4) for qt, fs in per_type.items()}

    metrics["macro_f1"] = round(_safe_mean(list(per_type_avg.values())), 4) if per_type_avg else None
    metrics["per_type_f1"] = per_type_avg
    return metrics


def aggregate_by_group(records: list, group_fn) -> dict:
    """Generic grouping aggregator — used for both difficulty-wise and
    query-type-wise breakdowns so metric computation is never duplicated
    per grouping dimension."""
    groups: dict = defaultdict(list)
    for r in records:
        key = group_fn(r)
        if key:
            groups[key].append(r)
    return {key: _metrics_from_records(recs) for key, recs in groups.items()}


def compute_stage_improvements(summaries: dict) -> dict:
    """Delta EM / F1 / Hallucination Rate between each consecutive stage."""
    pairs = [("stage1", "stage2"), ("stage2", "stage3"), ("stage3", "stage4")]
    improvements = {}
    for prev, curr in pairs:
        p, c = summaries[prev], summaries[curr]
        improvements[f"{curr}_vs_{prev}"] = {
            "delta_em":            round(c["exact_match"] - p["exact_match"], 2),
            "delta_f1":            round(c["f1"] - p["f1"], 4),
            "delta_hallucination": round(c["hallucination_rate"] - p["hallucination_rate"], 2),
        }
    return improvements


def compute_latency_stats(records: list) -> dict:
    totals = [r.latency["total"] for r in records if r.latency and "total" in r.latency]
    stats = {"avg_total_sec": round(_safe_mean(totals), 3), "n": len(totals)}
    # Stage 1 uniquely exposes a pipeline/verify split (Stage 6 calls verify()
    # itself for Stage 1; Stages 2-4 verify internally, so that split isn't
    # separately observable from Stage 6 without instrumenting those files).
    pipeline_times = [r.latency["pipeline"] for r in records if r.latency and "pipeline" in r.latency]
    verify_times   = [r.latency["verify"] for r in records if r.latency and "verify" in r.latency]
    if pipeline_times:
        stats["avg_pipeline_sec"] = round(_safe_mean(pipeline_times), 3)
        stats["avg_verify_sec"]   = round(_safe_mean(verify_times), 3)
    return stats


def compute_stage3_statistics(records: list) -> dict:
    """
    Stage 3 retrieval statistics: pipeline usage, coverage before/after
    expansion, expansion rate, average retrieved-passage count.

    Note: "average retrieval hops" is not included — Stage 3's
    adaptive_rag_query() return value does not expose a numeric hop count
    (only the pipeline name, e.g. "BRIDGE_HARD"), and fabricating one would
    not be a real measurement. Pipeline usage frequency is the closest
    available, honest proxy for retrieval depth.
    """
    n = len(records)
    pipeline_counts = Counter(r.pipeline for r in records if r.pipeline)
    coverage_before = [r.coverage_before for r in records if r.coverage_before is not None]
    coverage_after  = [r.coverage for r in records if r.coverage is not None]
    num_retrieved   = [r.num_retrieved for r in records if r.num_retrieved is not None]
    expanded_flags  = [r.expanded for r in records if r.expanded is not None]

    return {
        "pipeline_usage":           dict(pipeline_counts),
        "pipeline_usage_pct":       {k: round(v / n * 100, 1) for k, v in pipeline_counts.items()} if n else {},
        "avg_coverage_before_expansion": round(_safe_mean(coverage_before), 4),
        "avg_coverage_after_expansion":  round(_safe_mean(coverage_after), 4),
        "avg_num_retrieved":        round(_safe_mean(num_retrieved), 2),
        "expansion_rate_pct":       round(_safe_mean([1 if e else 0 for e in expanded_flags]) * 100, 2) if expanded_flags else 0.0,
        "n": n,
    }


def compute_stage4_statistics(records: list) -> dict:
    """
    Stage 4 agent statistics: action distribution (KEEP/EXPAND/COMPARE/
    DECOMPOSE/REWRITE/ABSTAIN counts and percentages), final status counts
    (SUPPORTED/BEST_EFFORT/ABSTAINED), and recovery counts (how often the
    final answer improved over Stage 3's own initial answer, broken down by
    which action produced the improvement).
    """
    all_actions = []
    for r in records:
        if r.actions_selected:
            all_actions.extend(r.actions_selected)
    action_counts = Counter(all_actions)
    n_actions = len(all_actions)

    status_counts    = Counter(r.status for r in records if r.status)
    recovered_counts = Counter(r.recovered_via for r in records if r.recovered_via)
    iterations       = [r.iterations for r in records if r.iterations is not None]
    action_lengths   = [len(r.actions_selected) for r in records if r.actions_selected]

    return {
        "avg_iterations":          round(_safe_mean(iterations), 2),
        "avg_actions_per_question": round(_safe_mean(action_lengths), 2),
        "action_counts":           dict(action_counts),
        "action_distribution_pct": {k: round(v / n_actions * 100, 1) for k, v in action_counts.items()} if n_actions else {},
        "status_counts":           dict(status_counts),
        "recovered_via_counts":    dict(recovered_counts),
        "recovery_rate_pct":       round(sum(recovered_counts.values()) / len(records) * 100, 2) if records else 0.0,
        "n": len(records),
    }


# ═════════════════════════════════════════════
# EVALUATION LOOP
# ═════════════════════════════════════════════

def evaluate(num_samples: int = 50) -> dict:
    _print_run_header(num_samples)

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Loading verifier (Stage 2 V2 — pretrained NLI, no local checkpoint needed)...")
    vm, vt = load_verifier(VERIFIER_PATH)

    print(f"\nLoading HotpotQA validation set ({num_samples} samples)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    records: dict = {s: [] for s in _STAGES}
    skipped = 0
    t_total = time.time()

    for i, ex in enumerate(tqdm(dataset, total=num_samples, desc="Questions")):
        if i >= num_samples:
            break

        # Official distractor protocol: one temporary corpus per question,
        # shared identically by all four stages, discarded afterward.
        ex_index, ex_passages, ex_bm25 = build_example_corpus(ex, embedder)
        if ex_index is None:
            skipped += 1
            continue

        try:
            s1_rec = run_stage1(ex, ex_index, embedder, ex_passages, ex_bm25, vm, vt)
        except Exception as exc:
            s1_rec = _error_record(ex, exc)
            tqdm.write(f"  [Stage 1] ERROR on '{ex['question'][:60]}': {exc}")
        records["stage1"].append(s1_rec)
        records["stage2"].append(run_stage2(s1_rec))

        try:
            s3_rec = run_stage3(ex, ex_index, embedder, ex_passages, ex_bm25, vm, vt)
        except Exception as exc:
            s3_rec = _error_record(ex, exc)
            tqdm.write(f"  [Stage 3] ERROR on '{ex['question'][:60]}': {exc}")
        records["stage3"].append(s3_rec)

        try:
            s4_rec = run_stage4(ex, ex_index, embedder, ex_passages, ex_bm25, vm, vt)
        except Exception as exc:
            s4_rec = _error_record(ex, exc)
            tqdm.write(f"  [Stage 4] ERROR on '{ex['question'][:60]}': {exc}")
        records["stage4"].append(s4_rec)

    elapsed = time.time() - t_total

    # ── Aggregation ──
    summary = {s: aggregate_overall(records[s]) for s in _STAGES}

    difficulty_statistics = {
        s: aggregate_by_group(records[s], lambda r: r.level) for s in _STAGES
    }
    query_type_statistics = {
        s: aggregate_by_group(records[s], lambda r: r.query_type) for s in _STAGES
    }
    latency_statistics    = {s: compute_latency_stats(records[s]) for s in _STAGES}
    confusion_matrices    = {s: compute_confusion_matrix(records[s]) for s in _STAGES}
    roc_pr                = {s: compute_roc_pr_auc(records[s]) for s in _STAGES}
    stage_improvements    = compute_stage_improvements(summary)
    retrieval_statistics  = compute_stage3_statistics(records["stage3"])
    agent_statistics      = compute_stage4_statistics(records["stage4"])
    stage_statistics      = {
        s: {"n": summary[s]["n"], "avg_latency_sec": latency_statistics[s]["avg_total_sec"]}
        for s in _STAGES
    }

    output = {
        # ── Backward-compatible top-level keys (unchanged shape) ──
        "timestamp":    datetime.now().isoformat(),
        "git_commit":   _git_commit(),
        "file_fingerprints": _build_fingerprint(),
        "config": {
            "embed_model":     EMBED_MODEL,
            "reranker_model":  RERANKER_MODEL,
            "ollama_model":    OLLAMA_MODEL,
            "top_k":           S1_TOP_K,
            "max_hops":        MAX_HOPS,
            "max_actions":     MAX_ACTIONS,
            "conf_threshold":  CONFIDENCE_THRESHOLD,
            "architecture":    "per-question temporary corpus (official distractor protocol)",
        },
        "num_samples":  num_samples,
        "skipped":      skipped,
        "elapsed_min":  round(elapsed / 60, 1),
        "summary":      summary,
        "detail": {s: [asdict(r) for r in records[s]] for s in _STAGES},

        # ── New, additive top-level keys ──
        "metrics":                summary,
        "stage_statistics":       stage_statistics,
        "difficulty_statistics":  difficulty_statistics,
        "query_type_statistics":  query_type_statistics,
        "latency":                latency_statistics,
        "confusion_matrices":     confusion_matrices,
        "roc_auc":                {s: roc_pr[s]["roc_auc"] for s in _STAGES},
        "pr_auc":                 {s: roc_pr[s]["pr_auc"] for s in _STAGES},
        "stage_improvements":     stage_improvements,
        "agent_statistics":       agent_statistics,
        "retrieval_statistics":   retrieval_statistics,
    }

    # ── Report ──
    _print_main_table(summary, num_samples, elapsed)
    _print_difficulty_table(difficulty_statistics)
    _print_query_type_table(query_type_statistics)
    _print_latency_table(latency_statistics)
    _print_roc_pr_table(roc_pr)
    _print_retrieval_statistics(retrieval_statistics)
    _print_agent_statistics(agent_statistics)
    _print_stage_improvements(stage_improvements)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {OUTPUT_FILE}\n")

    return output


# ═════════════════════════════════════════════
# REPORTING — terminal tables
# ═════════════════════════════════════════════

def _print_main_table(summary: dict, n: int, elapsed: float) -> None:
    LABELS = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
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
    print(f"| {'Metric':<{LW}} " + "".join(f"| {l:^{CW}} " for l in LABELS) + "|")
    print(sep.replace("-", "="))

    for key, display in METRICS:
        row = f"| {display:<{LW}} "
        for s in _STAGES:
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

    for sk in ("stage3", "stage4"):
        per = summary[sk].get("per_type_f1", {})
        if per:
            print(f"\n  {_STAGE_LABELS[sk]} — F1 by Query Type:")
            for qt, v in per.items():
                print(f"    {qt:<12}  {v:.4f}")


def _print_difficulty_table(difficulty_statistics: dict) -> None:
    print(f"\n{'─'*72}")
    print("  DIFFICULTY BREAKDOWN")
    print(f"{'─'*72}")
    for level in ("easy", "medium", "hard"):
        row_has_data = any(level in difficulty_statistics[s] for s in _STAGES)
        if not row_has_data:
            continue
        print(f"\n  Level: {level}")
        print(f"    {'Stage':<10} {'N':>5} {'EM%':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Halluc%':>9}")
        for s in _STAGES:
            m = difficulty_statistics[s].get(level)
            if not m:
                continue
            print(f"    {_STAGE_LABELS[s]:<10} {m['n']:>5} {m['exact_match']:>7.1f}% "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
                  f"{m['hallucination_rate']:>8.1f}%")


def _print_query_type_table(query_type_statistics: dict) -> None:
    print(f"\n{'─'*72}")
    print("  QUERY-TYPE BREAKDOWN")
    print(f"{'─'*72}")
    all_types = sorted({qt for s in _STAGES for qt in query_type_statistics[s]})
    for qt in all_types:
        print(f"\n  Type: {qt}")
        print(f"    {'Stage':<10} {'N':>5} {'EM%':>8} {'F1':>8} {'Halluc%':>9}")
        for s in _STAGES:
            m = query_type_statistics[s].get(qt)
            if not m:
                continue
            print(f"    {_STAGE_LABELS[s]:<10} {m['n']:>5} {m['exact_match']:>7.1f}% "
                  f"{m['f1']:>8.4f} {m['hallucination_rate']:>8.1f}%")


def _print_latency_table(latency_statistics: dict) -> None:
    print(f"\n{'─'*72}")
    print("  LATENCY (average seconds per question)")
    print(f"{'─'*72}")
    print(f"  {'Stage':<10} {'Total':>10} {'Pipeline':>10} {'Verify':>10}")
    for s in _STAGES:
        st = latency_statistics[s]
        pipeline = f"{st['avg_pipeline_sec']:.3f}" if "avg_pipeline_sec" in st else "n/a*"
        verify_t = f"{st['avg_verify_sec']:.3f}" if "avg_verify_sec" in st else "n/a*"
        print(f"  {_STAGE_LABELS[s]:<10} {st['avg_total_sec']:>10.3f} {pipeline:>10} {verify_t:>10}")
    print("  * Stages 2-4 verify internally, so a separate pipeline/verify split "
          "isn't observable from Stage 6 without instrumenting those files.")


def _print_roc_pr_table(roc_pr: dict) -> None:
    print(f"\n{'─'*72}")
    print("  VERIFIER ROC-AUC / PR-AUC  (P(SUPPORTED) predicting actual EM correctness)")
    print(f"{'─'*72}")
    print(f"  {'Stage':<10} {'ROC-AUC':>10} {'PR-AUC':>10} {'N':>6}")
    for s in _STAGES:
        r = roc_pr[s]
        roc = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "N/A"
        pr  = f"{r['pr_auc']:.4f}" if r["pr_auc"] is not None else "N/A"
        print(f"  {_STAGE_LABELS[s]:<10} {roc:>10} {pr:>10} {r['n']:>6}")


def _print_retrieval_statistics(stats: dict) -> None:
    print(f"\n{'─'*72}")
    print("  STAGE 3 — RETRIEVAL STATISTICS")
    print(f"{'─'*72}")
    print(f"  Avg coverage before expansion : {stats['avg_coverage_before_expansion']:.4f}")
    print(f"  Avg coverage after expansion  : {stats['avg_coverage_after_expansion']:.4f}")
    print(f"  Expansion rate                : {stats['expansion_rate_pct']:.1f}%")
    print(f"  Avg passages retrieved        : {stats['avg_num_retrieved']:.2f}")
    print(f"  Pipeline usage:")
    for pipeline, pct in sorted(stats["pipeline_usage_pct"].items(), key=lambda kv: -kv[1]):
        count = stats["pipeline_usage"][pipeline]
        print(f"    {pipeline:<20} {count:>4}  ({pct:.1f}%)")


def _print_agent_statistics(stats: dict) -> None:
    print(f"\n{'─'*72}")
    print("  STAGE 4 — AGENT STATISTICS")
    print(f"{'─'*72}")
    print(f"  Avg iterations (incl. Stage 3 pass) : {stats['avg_iterations']:.2f}")
    print(f"  Avg actions taken beyond Stage 3     : {stats['avg_actions_per_question']:.2f}")
    print(f"  Recovery rate (improved over Stage 3): {stats['recovery_rate_pct']:.1f}%")
    print(f"  Action distribution:")
    for action, pct in sorted(stats["action_distribution_pct"].items(), key=lambda kv: -kv[1]):
        count = stats["action_counts"][action]
        print(f"    {action:<12} {count:>4}  ({pct:.1f}%)")
    print(f"  Final status counts:")
    for status, count in stats["status_counts"].items():
        print(f"    {status:<12} {count:>4}")
    if stats["recovered_via_counts"]:
        print(f"  Recovered via:")
        for action, count in stats["recovered_via_counts"].items():
            print(f"    {action:<12} {count:>4}")


def _print_stage_improvements(improvements: dict) -> None:
    print(f"\n{'─'*72}")
    print("  STAGE-TO-STAGE IMPROVEMENTS")
    print(f"{'─'*72}")
    print(f"  {'Comparison':<20} {'ΔEM':>10} {'ΔF1':>10} {'ΔHalluc':>10}")
    for name, delta in improvements.items():
        print(f"  {name:<20} {delta['delta_em']:>+9.2f}% {delta['delta_f1']:>+10.4f} "
              f"{delta['delta_hallucination']:>+9.2f}%")
    print()


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

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

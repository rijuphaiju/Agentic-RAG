"""
Stage 2: Automatic Label Generator
====================================
Project: HARA — Hallucination-Aware Retrieval Agent

Reads verifier_dataset.jsonl (produced by Stage_2_Build_Verifier_Dataset.py —
real Stage 1 outputs, no labels) and writes verifier_dataset_labeled.jsonl,
assigning SUPPORTED / PARTIAL / UNSUPPORTED via a deterministic-first decision
cascade. An LLM is used only as a minority fallback for genuinely ambiguous
cases — it is never the primary labeling authority.

Label derivation (exactly as designed):
    Decision 1 — is the core answer BOTH correct AND grounded?
    Decision 2 — does the answer contain any extra claim that is NOT grounded?

        core incorrect or ungrounded          -> UNSUPPORTED
        core correct+grounded, no bad extras   -> SUPPORTED
        core correct+grounded, >=1 bad extra    -> PARTIAL

Every decision records which tier/signal resolved it, so every label is
reproducible and auditable (see `label_metadata` in the output schema).

Tier structure per decision (cheapest/most-deterministic first):
    Correctness : 1) exact match / containment  2) token F1
                  3) embedding cosine similarity (fixed BGE model — a static
                     metric, not a generative LLM judgment)
                  4) LLM semantic-equivalence question (minority fallback)
    Grounding   : 1) verbatim/near-verbatim span match against the passages
                     actually given to the LLM (DPR/RAG-HAT-style
                     distant-supervision "has_answer" check)
                  2) entity/number/date token-coverage against context
                  3) llm_judge_supported() (minority fallback, reused as-is
                     from Stage_1_RAG_Pipeline.py)
    Extra claims: 1) deterministic sentence/conjunction clause splitting
                  2) LLM decomposition (minority fallback, only when the
                     regex split can't confidently decide)

Train/val/test split is assigned deterministically by question_id (a hash,
not a random shuffle), so re-running this script or extending the dataset
later never reshuffles previously assigned splits.

Usage:
    python Stage_2_Label_Generator.py --input verifier_dataset.jsonl \\
        --output verifier_dataset_labeled.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import faiss
import ollama
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from Stage_1_RAG_Pipeline import (
    normalize_answer,
    token_f1,
    llm_judge_supported,
    EMBED_MODEL,
    OLLAMA_MODEL,
)

logger = logging.getLogger("stage2_label_generator")

DEFAULT_INPUT       = "verifier_dataset.jsonl"
DEFAULT_OUTPUT      = "verifier_dataset_labeled.jsonl"
DEFAULT_AUDIT_CSV   = "verifier_dataset_audit_sample.csv"
DEFAULT_AUDIT_SIZE  = 200

# Thresholds are starting points — calibrate against the manual audit sample
# before trusting them at full scale (see the design doc / thesis writeup).
F1_HIGH        = 0.70   # token F1 >= this -> confidently correct
F1_LOW         = 0.30   # token F1 <= this (with low embedding sim too) -> confidently incorrect
EMBED_SIM_HIGH = 0.85   # cosine sim >= this -> confidently correct (paraphrase)
EMBED_SIM_LOW  = 0.55   # cosine sim <= this -> confidently incorrect
COVERAGE_HIGH  = 0.80   # entity/token coverage >= this -> confidently grounded
COVERAGE_LOW   = 0.30   # entity/token coverage <= this -> confidently ungrounded

_SALIENT_TOKEN_RE = re.compile(r'\b[A-Z][a-zA-Z]{2,}\b|\b\d{3,4}\b')
_CLAUSE_SPLIT_RE = re.compile(
    r'(?:(?<=[.!?])\s+|\s+and also\s+|,\s+and also\s+|\s+as well as\s+|,\s+and\s+|\s+and\s+|,\s+)',
    re.IGNORECASE,
)
_LEADING_CONJ_RE = re.compile(r'^(?:and|also|but|while|with|as well as)\s+', re.IGNORECASE)
MIN_EXTRA_CLAIM_WORDS = 3   # below this (after stripping a leading conjunction),
                            # treat a split fragment as truncation noise, not a
                            # genuine extra claim — see decompose_claims().


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


def _record_key(record: Dict[str, Any]) -> str:
    """Unique key for one labelable unit. The multi-candidate dataset writes
    several rows per question_id, so resume tracking must be per-candidate
    (via `candidate_id`) — falling back to `question_id` alone only for the
    older single-candidate dataset format, where that's already unique."""
    return record.get("candidate_id") or record["question_id"]


def load_labeled_keys(output_path: str) -> Set[str]:
    """Scans an existing labeled-output JSONL file and returns the set of
    candidate keys already labeled, so a re-run resumes without re-labeling
    or silently skipping a question's other, not-yet-labeled candidates."""
    keys: Set[str] = set()
    if not os.path.exists(output_path):
        return keys
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(_record_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


# ─────────────────────────────────────────────
# SPLIT ASSIGNMENT
# ─────────────────────────────────────────────
def assign_split(question_id: str, val_pct: int = 10, test_pct: int = 10) -> str:
    """Deterministic train/val/test split by question_id (hash, not a random
    shuffle) — reproducible across reruns, and guarantees that if multiple
    records ever shared a question_id, they'd land in the same bucket."""
    bucket = int(hashlib.md5(question_id.encode()).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


# ─────────────────────────────────────────────
# LOW-LEVEL SIGNALS
# ─────────────────────────────────────────────
def extract_salient_tokens(text: str) -> Set[str]:
    """Deterministic entity/number/date extraction — capitalized words (3+
    letters) and 3-4 digit numbers (catches years). Same regex-based-entity
    style already used in Stage_1_RAG_Pipeline (_get_other_entity)."""
    return {t.lower() for t in _SALIENT_TOKEN_RE.findall(text)}


def _cosine_sim(embedder: SentenceTransformer, a: str, b: str) -> float:
    """Fixed pretrained-embedding cosine similarity — a static metric (same
    category as BERTScore), not a generative LLM judgment."""
    vecs = embedder.encode([a, b], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vecs)
    return float(np.dot(vecs[0], vecs[1]))


def _llm_equivalence_check(question: str, claim: str, gold_answer: str) -> bool:
    """Decision 1, Tier 4 — minority fallback only, invoked when lexical F1
    and embedding similarity are both ambiguous."""
    prompt = (
        f"Question: {question}\n"
        f"Candidate answer: {claim}\n"
        f"Reference answer: {gold_answer}\n\n"
        f"Is the candidate answer semantically equivalent to the reference "
        f"answer as a response to this question? Reply with only YES or NO."
    )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 5},
        )
        return response["message"]["content"].strip().upper().startswith("YES")
    except Exception:
        logger.exception("LLM equivalence check failed — defaulting to False")
        return False


def _llm_decompose(answer: str) -> List[str]:
    """Decision 3, minority fallback — invoked only when deterministic
    clause splitting can't confidently decide on a long, unstructured answer."""
    prompt = (
        f"List the distinct factual claims made in the following sentence, "
        f"one per line, as short phrases. If it makes only one claim, "
        f"return just that one line.\n\nSentence: {answer}\n\nClaims:"
    )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 80},
        )
        lines = [l.strip("-• \t") for l in response["message"]["content"].splitlines() if l.strip()]
        return lines if lines else [answer]
    except Exception:
        logger.exception("LLM decomposition failed — treating answer as a single claim")
        return [answer]


# ─────────────────────────────────────────────
# DECISION 1: CORE CORRECTNESS CASCADE
# ─────────────────────────────────────────────
def assess_correctness(
    claim: str, gold_answer: str, question: str, embedder: SentenceTransformer,
) -> Tuple[bool, int, str]:
    """Is `claim` correct relative to `gold_answer`? Returns (is_correct, tier, signal)."""
    norm_claim = normalize_answer(claim)
    norm_gold = normalize_answer(gold_answer)

    if not norm_gold:
        return True, 1, "empty_gold"

    # Tier 1 — exact match / containment (free, deterministic)
    if norm_claim == norm_gold:
        return True, 1, "exact_match"
    if norm_gold in norm_claim or norm_claim in norm_gold:
        return True, 1, "containment"

    # Tier 2 — token F1 (free, deterministic)
    f1 = token_f1(claim, gold_answer)
    if f1 >= F1_HIGH:
        return True, 2, f"token_f1={f1:.2f}"

    # Tier 3 — embedding cosine similarity (fixed model, not an LLM judgment)
    sim = _cosine_sim(embedder, claim, gold_answer)
    if sim >= EMBED_SIM_HIGH:
        return True, 3, f"embedding_sim={sim:.2f}"
    if sim <= EMBED_SIM_LOW and f1 <= F1_LOW:
        return False, 3, f"embedding_sim={sim:.2f},token_f1={f1:.2f}"

    # Tier 4 — LLM equivalence, minority fallback only
    is_equiv = _llm_equivalence_check(question, claim, gold_answer)
    return is_equiv, 4, "llm_equivalence"


# ─────────────────────────────────────────────
# DECISION 2: GROUNDING CASCADE
# ─────────────────────────────────────────────
def assess_grounding(
    claim: str,
    reranked_passages: List[Dict[str, Any]],
    supporting_titles: Set[str],
    question: str,
) -> Tuple[bool, int, str]:
    """Is `claim` evidenced by the passages actually given to the LLM?
    Returns (is_grounded, tier, signal)."""
    if not normalize_answer(claim):
        return False, 1, "empty_claim"

    norm_claim = normalize_answer(claim)
    context_concat = " ".join(p["text"] for p in reranked_passages)

    # Tier 1 — verbatim/near-verbatim span match (DPR/RAG-HAT-style
    # distant-supervision "has_answer" check), corroborated by whether the
    # matching passage is one of the gold supporting-fact titles.
    matching_titles = {
        p["title"] for p in reranked_passages
        if norm_claim and norm_claim in normalize_answer(p["text"])
    }
    if matching_titles:
        if matching_titles & supporting_titles:
            return True, 1, "verbatim_span_supporting_title"
        return True, 1, "verbatim_span_other_title"

    # Tier 2 — entity/number/date token coverage (free, deterministic)
    claim_tokens = extract_salient_tokens(claim)
    if claim_tokens:
        context_tokens = extract_salient_tokens(context_concat)
        coverage = len(claim_tokens & context_tokens) / len(claim_tokens)
        if coverage >= COVERAGE_HIGH:
            return True, 2, f"entity_coverage={coverage:.2f}"
        if coverage <= COVERAGE_LOW:
            return False, 2, f"entity_coverage={coverage:.2f}"
    else:
        # No proper nouns/numbers to check (e.g. "Yes"/"No") — token overlap
        # with the context is a weaker but still free, deterministic proxy.
        overlap = token_f1(claim, context_concat)
        if overlap >= COVERAGE_HIGH:
            return True, 2, f"token_overlap={overlap:.2f}"
        if overlap <= COVERAGE_LOW:
            return False, 2, f"token_overlap={overlap:.2f}"

    # Tier 3 — LLM grounding judge, minority fallback only (reused as-is)
    is_grounded = llm_judge_supported(question, claim, reranked_passages)
    return is_grounded, 3, "llm_judge_supported"


# ─────────────────────────────────────────────
# DECISION 3: EXTRA-CLAIM DECOMPOSITION
# ─────────────────────────────────────────────
def _is_meaningful_claim(text: str) -> bool:
    """Filters out truncation artifacts / dangling fragments (e.g. a lone
    "but", or "with its publication beginning" left over after an aggressive
    upstream answer truncation) from being counted as genuine extra claims.
    Without this, a truncated sentence stub gets graded as "ungrounded"
    purely because it has no evaluable content — manufacturing a spurious
    PARTIAL label rather than reflecting a real additional fact."""
    stripped = _LEADING_CONJ_RE.sub("", text).strip()
    return len(stripped.split()) >= MIN_EXTRA_CLAIM_WORDS


def decompose_claims(processed_answer: str, gold_answer: str) -> Tuple[str, List[str], int, str]:
    """Splits `processed_answer` into a core claim (best correctness match
    against gold_answer) and zero or more extra claims, via deterministic
    sentence/conjunction splitting. Falls back to a single LLM decomposition
    call only for a long, unstructured single clause with no detectable
    split point and no clean gold match. Dangling fragments below
    MIN_EXTRA_CLAIM_WORDS are dropped as truncation noise, not extra claims.

    Returns (core_claim, extra_claims, tier, signal).
    """
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(processed_answer) if p.strip()]
    if not parts:
        return processed_answer.strip(), [], 1, "no_split_needed"

    if len(parts) == 1:
        only = parts[0]
        norm_only, norm_gold = normalize_answer(only), normalize_answer(gold_answer)
        looks_like_single_claim = (
            len(only.split()) <= 15
            or (norm_gold and (norm_gold in norm_only or norm_only in norm_gold))
        )
        if looks_like_single_claim:
            return only, [], 1, "single_clause"
        decomposed = _llm_decompose(only)
        if len(decomposed) > 1:
            extras = [c for c in decomposed[1:] if _is_meaningful_claim(c)]
            return decomposed[0], extras, 2, "llm_decomposition"
        return only, [], 1, "single_clause"

    # Multiple clauses found — the "core" one is whichever best matches
    # gold_answer (cheap containment/F1 scoring, no LLM call needed).
    def _match_score(clause: str) -> float:
        norm_c, norm_g = normalize_answer(clause), normalize_answer(gold_answer)
        if norm_g and (norm_g in norm_c or norm_c in norm_g):
            return 1.0
        return token_f1(clause, gold_answer)

    best_idx = max(range(len(parts)), key=lambda i: _match_score(parts[i]))
    core = parts[best_idx]
    extras = [p for i, p in enumerate(parts) if i != best_idx and _is_meaningful_claim(p)]
    return core, extras, 1, "clause_split"


# ─────────────────────────────────────────────
# LABEL ORCHESTRATION
# ─────────────────────────────────────────────
def label_record(record: Dict[str, Any], embedder: SentenceTransformer) -> Dict[str, Any]:
    """Runs the full Decision 1/2/3 cascade on one dataset record and returns
    the record augmented with `label`, `label_metadata`, and `split`.

    Works against the multi-candidate schema (`candidate_answer` / `context`)
    produced by the current Stage_2_Build_Verifier_Dataset.py, falling back
    to the older single-candidate field names (`processed_answer` /
    `reranked_passages`) so any pre-existing dataset file in that format can
    still be labeled without a separate code path.
    """
    question = record["question"]
    gold_answer = record["gold_answer"]
    processed_answer = record.get("candidate_answer", record.get("processed_answer"))
    reranked_passages = record.get("context", record.get("reranked_passages"))
    supporting_titles = {sf["title"] for sf in record["supporting_facts"]}

    core_claim, extra_claims, decomp_tier, decomp_signal = decompose_claims(processed_answer, gold_answer)

    core_correct, correct_tier, correct_signal = assess_correctness(
        core_claim, gold_answer, question, embedder
    )
    core_grounded, grounded_tier, grounded_signal = assess_grounding(
        core_claim, reranked_passages, supporting_titles, question
    )

    extra_claim_results = []
    for extra in extra_claims:
        grounded, tier, signal = assess_grounding(extra, reranked_passages, supporting_titles, question)
        extra_claim_results.append({"text": extra, "grounded": grounded, "tier": tier, "signal": signal})

    if not (core_correct and core_grounded):
        label = "UNSUPPORTED"
    elif all(c["grounded"] for c in extra_claim_results):
        label = "SUPPORTED"
    else:
        label = "PARTIAL"

    out = dict(record)
    out["label"] = label
    out["label_metadata"] = {
        "core_claim": core_claim,
        "core_correct": core_correct,
        "core_correct_tier": correct_tier,
        "core_correct_signal": correct_signal,
        "core_grounded": core_grounded,
        "core_grounded_tier": grounded_tier,
        "core_grounded_signal": grounded_signal,
        "decomposition_tier": decomp_tier,
        "decomposition_signal": decomp_signal,
        "extra_claims": extra_claim_results,
    }
    out["split"] = assign_split(record["question_id"])
    return out


# ─────────────────────────────────────────────
# AUDIT SAMPLE EXPORT
# ─────────────────────────────────────────────
def export_audit_sample(labeled_path: str, csv_path: str, sample_size: int, seed: int = 42) -> None:
    """Writes a random sample of labeled records to a flat CSV for manual
    review — the calibration/validation set for the deterministic thresholds
    above, not a labeling mechanism itself."""
    records = []
    with open(labeled_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    random.Random(seed).shuffle(records)
    sample = records[:sample_size]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id", "source", "question", "gold_answer", "candidate_answer",
            "label", "core_correct", "core_correct_signal",
            "core_grounded", "core_grounded_signal", "num_extra_claims",
            "human_verdict_agrees",  # left blank for manual annotation
        ])
        for r in sample:
            md = r["label_metadata"]
            candidate_answer = r.get("candidate_answer", r.get("processed_answer"))
            writer.writerow([
                r["question_id"], r.get("source", "stage1"), r["question"], r["gold_answer"], candidate_answer,
                r["label"], md["core_correct"], md["core_correct_signal"],
                md["core_grounded"], md["core_grounded_signal"], len(md["extra_claims"]),
                "",
            ])
    logger.info(f"Audit sample ({len(sample)} records) written to {csv_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: assign SUPPORTED/PARTIAL/UNSUPPORTED labels to a real Stage 1 output dataset."
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-csv", type=str, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--audit-size", type=int, default=DEFAULT_AUDIT_SIZE)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger.info(f"Stage 2 label generation starting: {args.input} -> {args.output}")

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    processed_keys = load_labeled_keys(args.output)
    if processed_keys:
        logger.info(f"Resuming: {len(processed_keys)} records already labeled in {args.output}")

    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    with open(args.input, "r", encoding="utf-8") as f:
        input_records = [json.loads(line) for line in f if line.strip()]
    logger.info(f"{len(input_records)} records loaded from {args.input}")

    label_counts: Counter = Counter()
    tier_counts: Counter = Counter()
    llm_fallback_count = 0
    done = len(processed_keys)

    pbar = tqdm(total=len(input_records), initial=done, desc="Labeling")
    try:
        with open(args.output, "a", encoding="utf-8") as out_f:
            for record in input_records:
                key = _record_key(record)
                if key in processed_keys:
                    continue
                qid = record["question_id"]
                try:
                    labeled = label_record(record, embedder)
                except Exception:
                    logger.exception(f"Error labeling question {qid} — skipping")
                    continue

                out_f.write(json.dumps(labeled) + "\n")
                out_f.flush()

                label_counts[labeled["label"]] += 1
                md = labeled["label_metadata"]
                tier_counts[f"correct_tier_{md['core_correct_tier']}"] += 1
                tier_counts[f"grounded_tier_{md['core_grounded_tier']}"] += 1
                if md["core_correct_tier"] == 4 or md["core_grounded_tier"] == 3:
                    llm_fallback_count += 1

                done += 1
                pbar.update(1)
    except KeyboardInterrupt:
        logger.warning(f"Interrupted. {done} records labeled so far in {args.output}. Re-run to resume.")
    finally:
        pbar.close()

    logger.info(f"Done. {done} records labeled -> {args.output}")
    logger.info(f"Label distribution: {dict(label_counts)}")
    logger.info(f"Tier usage: {dict(tier_counts)}")
    if done:
        logger.info(f"LLM fallback rate: {llm_fallback_count}/{done} ({100*llm_fallback_count/done:.1f}%)")

    export_audit_sample(args.output, args.audit_csv, args.audit_size)


if __name__ == "__main__":
    main()

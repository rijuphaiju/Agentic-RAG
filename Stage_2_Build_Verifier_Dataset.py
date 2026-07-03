"""
Stage 2: Real Verifier Dataset Generator
=========================================
Project: HARA — Hallucination-Aware Retrieval Agent

Replaces the old synthetic template-based training data generator
(Stage_2_Verifier_GPU.py's build_training_data()). This script produces NO
labels — it only runs the existing, unmodified Stage 1 pipeline over the
HotpotQA TRAIN split and records exactly what it produced, so that a
downstream labeling step (Stage_2_Label_Generator.py) can assign
SUPPORTED/PARTIAL/UNSUPPORTED from real RAG behaviour instead of fabricated
templates.

Pipeline per question, mirroring Stage_1_RAG_Pipeline.evaluate() exactly:
    build_example_corpus() -> retrieve_hybrid() -> rerank_passages()
    -> generate_answer()
Stage 1 itself is never modified or reimplemented — every step above is
imported directly from Stage_1_RAG_Pipeline.py.

`generated_answer` is generate_answer()'s return value (Stage 1's own
_extract_answer post-processing is already baked into it — that is the only
answer text Stage 1 ever exposes to anything downstream, including Stage 3/4).
`processed_answer` additionally applies _distill_for_verify(), the extra
shortening step verify() applies at inference time — this is the text a
labeling/training pipeline should treat as "what the verifier will actually
see."

Only the HotpotQA TRAIN split is read here. The official validation split is
never touched by this script, so it stays uncontaminated for Stage 1/3/4's
own downstream evaluation metrics.

Usage:
    python Stage_2_Build_Verifier_Dataset.py --num-samples 12000 --output verifier_dataset.jsonl
    (re-running the same command resumes from the last completed example)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from Stage_1_RAG_Pipeline import (
    build_example_corpus,
    retrieve_hybrid,
    rerank_passages,
    generate_answer,
    compute_recall_at_k,
    compute_recall_at_k_titles,
    EMBED_MODEL,
    RERANK_POOL,
    TOP_K,
    CHUNK_SIZE,
)
from Stage_2_Verifier_GPU import _distill_for_verify

logger = logging.getLogger("stage2_dataset_builder")

DEFAULT_OUTPUT      = "verifier_dataset.jsonl"
DEFAULT_NUM_SAMPLES = 12000


def setup_logging(verbose: bool = False) -> None:
    """Configures console logging for this script only (does not touch the
    root logger's handlers set up by any imported module)."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


def load_processed_ids(output_path: str) -> Set[str]:
    """Scans an existing JSONL output file (if any) and returns the set of
    question_ids already written, so a re-run resumes instead of restarting.

    Tolerates a malformed trailing line (e.g. a hard interrupt mid-write) by
    skipping it rather than crashing the resume scan.
    """
    processed: Set[str] = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                processed.add(record["question_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def _strip_passage(passage: Dict[str, Any]) -> Dict[str, Any]:
    """Keeps only the JSON-relevant fields from a Stage 1 passage dict,
    rounding floats for a smaller, cleaner file."""
    out: Dict[str, Any] = {"title": passage["title"], "text": passage["text"]}
    if "score" in passage:
        out["score"] = round(float(passage["score"]), 4)
    if "chunk_id" in passage:
        out["chunk_id"] = passage["chunk_id"]
    if "rerank_score" in passage:
        out["rerank_score"] = round(float(passage["rerank_score"]), 4)
    return out


def build_record(
    example: Dict[str, Any],
    embedder: SentenceTransformer,
    chunk_size: int = CHUNK_SIZE,
) -> Optional[Dict[str, Any]]:
    """Runs the existing, unmodified Stage 1 pipeline for one HotpotQA train
    example and packages the result into a JSONL-ready record.

    Returns None if the example's context is empty (nothing to build a
    per-example corpus from) — the caller should skip such examples.
    """
    question    = example["question"]
    gold_answer = example["answer"]
    gold_titles = list(dict.fromkeys(example["supporting_facts"]["title"]))

    ex_index, ex_passages, ex_bm25 = build_example_corpus(example, embedder, chunk_size=chunk_size)
    if ex_index is None:
        return None

    # Mirrors Stage_1_RAG_Pipeline.evaluate()'s pool-sizing convention exactly.
    recall_pool = min(max(RERANK_POOL, 20), len(ex_passages))
    pool = retrieve_hybrid(
        question, ex_index, embedder, ex_passages,
        top_k=recall_pool, bm25=ex_bm25,
    )
    reranked = rerank_passages(question, pool, top_k=TOP_K)

    # Baseline Stage 1 generation — no query_type, no prompt variation.
    generated_answer = generate_answer(question, reranked)
    processed_answer = _distill_for_verify(generated_answer)

    pool_titles = [p["title"] for p in pool]
    supporting_facts = [
        {"title": t, "sent_id": s}
        for t, s in zip(
            example["supporting_facts"]["title"],
            example["supporting_facts"]["sent_id"],
        )
    ]

    return {
        "question_id": example["id"],
        "question": question,
        "question_type": example.get("type", "bridge"),
        "level": example.get("level", "unknown"),
        "gold_answer": gold_answer,
        "supporting_facts": supporting_facts,
        "retrieved_passages": [_strip_passage(p) for p in pool],
        "reranked_passages": [_strip_passage(p) for p in reranked],
        "generated_answer": generated_answer,
        "processed_answer": processed_answer,
        "retrieval_metrics": {
            "recall_at_2": round(compute_recall_at_k_titles(pool_titles, gold_titles, 2), 4),
            "recall_at_5": round(compute_recall_at_k(pool_titles, gold_titles, 5), 4),
            "recall_at_10": round(compute_recall_at_k(pool_titles, gold_titles, 10), 4),
            "recall_at_20": round(compute_recall_at_k(pool_titles, gold_titles, 20), 4),
        },
        "corpus_size": len(ex_passages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: build a verifier training dataset from real Stage 1 outputs (no labels)."
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                         help=f"Number of HotpotQA train questions to process (default: {DEFAULT_NUM_SAMPLES})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                         help=f"JSONL output path; re-running with the same path resumes (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                         help="Sentences per passage chunk, passed to build_example_corpus")
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger.info("Stage 2 dataset generation starting")
    logger.info(f"Target: {args.num_samples} examples -> {args.output}")

    processed_ids = load_processed_ids(args.output)
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} examples already present in {args.output}")

    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    logger.info("Loading HotpotQA train split (distractor)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="train")

    done = len(processed_ids)
    skipped_errors = 0
    skipped_empty = 0

    pbar = tqdm(total=args.num_samples, initial=done, desc="Building verifier dataset")
    try:
        with open(args.output, "a", encoding="utf-8") as f:
            for example in dataset:
                if done >= args.num_samples:
                    break
                qid = example["id"]
                if qid in processed_ids:
                    continue
                try:
                    record = build_record(example, embedder, chunk_size=args.chunk_size)
                except Exception:
                    logger.exception(f"Error processing question {qid} — skipping")
                    skipped_errors += 1
                    continue
                if record is None:
                    skipped_empty += 1
                    continue
                f.write(json.dumps(record) + "\n")
                f.flush()
                done += 1
                pbar.update(1)
    except KeyboardInterrupt:
        logger.warning(
            f"Interrupted by user. {done} examples saved to {args.output}. "
            f"Re-run the same command to resume from where it left off."
        )
    finally:
        pbar.close()

    logger.info(
        f"Done. {done} examples in {args.output} "
        f"({skipped_errors} errors, {skipped_empty} empty-context skips)."
    )


if __name__ == "__main__":
    main()

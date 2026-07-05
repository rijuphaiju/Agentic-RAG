"""
Stage 2: Adaptive Feedback — Generator Learns From the Validator
====================================================================
Project: HARA — Hallucination-Aware Retrieval Agent

Point 5 of the HARA architecture: "the generator learns from the validator
and is adaptive on the next run" — implemented WITHOUT any change to
Stage_1_RAG_Pipeline.py or any other stage's file.

How: Stage_1_RAG_Pipeline.generate_answer() calls `ollama.chat(model=
OLLAMA_MODEL, ...)` where OLLAMA_MODEL is a plain local model TAG ("llama3.2").
Ollama lets a local tag's SYSTEM prompt be redefined in place via
`ollama create <tag> -f Modelfile`, without touching any Python call site
that references that tag by name. So:

    1. Stage_2_Verifier.verify() optionally logs every non-SUPPORTED verdict
       to FEEDBACK_LOG_PATH (opt-in via HARA_ADAPTIVE_FEEDBACK=1, off by
       default — see Stage_2_Verifier.py).
    2. This script reads that log, distills recurring failure patterns into
       a short corrective system prompt, and re-registers the local Ollama
       tag with it.
    3. Every subsequent `ollama.chat(model="llama3.2", ...)` call from any
       stage picks up the corrective system prompt automatically — the
       generator is "adaptive on the next run" with zero code changes
       elsewhere.

This is a manual/periodic tool, not wired into any stage's live request
path — you decide when to apply accumulated feedback (e.g. after a batch of
evaluation runs), so a live serving process never has its model mutated
mid-run.

Usage:
    python Stage_2_AdaptiveFeedback.py --dry-run     # show the distilled prompt only
    python Stage_2_AdaptiveFeedback.py --apply       # write Modelfile and run `ollama create`
    python Stage_2_AdaptiveFeedback.py --reset       # restore the tag to its vanilla base model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
from collections import Counter
from typing import Any, Dict, List

from Stage_1_RAG_Pipeline import OLLAMA_MODEL
from Stage_2_Verifier import FEEDBACK_LOG_PATH

logger = logging.getLogger("stage2_adaptive_feedback")

# The plain upstream model this tag is layered on top of. Ollama model tags
# without a ":" suffix pull from the library by this same name, so re-using
# OLLAMA_MODEL itself as the FROM target re-derives from the vanilla weights
# each time rather than compounding onto a previously-adapted version.
BASE_MODEL_FOR_MODELFILE = OLLAMA_MODEL
MAX_PATTERNS_IN_PROMPT = 8
MIN_OCCURRENCES_TO_INCLUDE = 2

_REASON_GUIDANCE = {
    "ANSWER_TYPE_MISMATCH": "Answer with the exact type of thing the question asks for (a year, a name, a place, etc.) — not a related but differently-typed fact.",
    "QUESTION_NOT_ANSWERED": "Make sure your answer directly addresses what was asked, not just a related topic.",
    "WRONG_ENTITY": "Double-check that the entity you name actually matches the role the question asks about, not just any entity mentioned nearby in the context.",
    "WRONG_YEAR": "Double-check dates/years against the context carefully before answering — do not guess or approximate.",
    "WRONG_NUMBER": "Double-check numeric values against the context carefully before answering — do not guess or approximate.",
    "MISSING_REQUIRED_ENTITY": "Include the specific named entity the question requires, not a vague reference to it.",
    "MISSING_REQUIRED_NUMBER": "Include the specific number the question requires.",
    "MISSING_REQUIRED_DATE": "Include the specific date the question requires.",
    "PARTIAL_COMPARISON": "For comparison questions, commit to one side — do not hedge with 'both' or 'unclear'.",
    "CONTRADICTED_BY_EVIDENCE": "Only state facts that are directly supported by the given context — do not state the opposite of what the context says.",
    "INSUFFICIENT_EVIDENCE": "If the context doesn't clearly support an answer, say so rather than guessing.",
    "NO_RELEVANT_PASSAGE": "Only answer using the provided context — do not rely on outside knowledge that isn't in the passages.",
    "MULTIPLE_CONFLICTING_PASSAGES": "When passages disagree, prefer the one that most directly discusses the entity/topic named in the question.",
}

BASE_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant for a Wikipedia-grounded "
    "retrieval system. Answer using ONLY the provided context."
)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


def load_feedback(log_path: str) -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(log_path):
        return records
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def distill_system_prompt(records: List[Dict[str, Any]]) -> str:
    """Turns recurring failure_reason patterns into a short, concrete list of
    corrective instructions — no per-question memorization, since HotpotQA
    train questions won't recur verbatim; the goal is to correct systematic
    generation habits the validator keeps catching."""
    reason_counts = Counter(r.get("failure_reason") for r in records if r.get("failure_reason"))
    lines = [BASE_SYSTEM_PROMPT]

    if reason_counts:
        lines.append("")
        lines.append(
            "A validator has reviewed your previous answers on this task and found "
            "these recurring mistakes — correct for them:"
        )
        for reason, count in reason_counts.most_common(MAX_PATTERNS_IN_PROMPT):
            if count < MIN_OCCURRENCES_TO_INCLUDE:
                continue
            guidance = _REASON_GUIDANCE.get(reason)
            if guidance:
                lines.append(f"- {guidance}")

    return "\n".join(lines)


def write_modelfile(system_prompt: str, path: str) -> None:
    escaped = system_prompt.replace('"""', '\\"\\"\\"')
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'FROM {BASE_MODEL_FOR_MODELFILE}\n')
        f.write(f'SYSTEM """{escaped}"""\n')


def apply_modelfile(tag: str, modelfile_path: str) -> None:
    logger.info(f"Running: ollama create {tag} -f {modelfile_path}")
    result = subprocess.run(
        ["ollama", "create", tag, "-f", modelfile_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"ollama create failed:\n{result.stderr}")
        raise RuntimeError(f"ollama create failed with exit code {result.returncode}")
    logger.info(f"Ollama tag '{tag}' updated.\n{result.stdout}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill Stage 2 verifier feedback into a corrective system prompt for the local Ollama model tag."
    )
    parser.add_argument("--feedback-log", type=str, default=FEEDBACK_LOG_PATH)
    parser.add_argument("--tag", type=str, default=OLLAMA_MODEL,
                         help=f"Local Ollama model tag to update (default: {OLLAMA_MODEL})")
    parser.add_argument("--dry-run", action="store_true", help="Print the distilled system prompt without applying it")
    parser.add_argument("--apply", action="store_true", help="Write the Modelfile and run `ollama create`")
    parser.add_argument("--reset", action="store_true", help="Restore the tag to the vanilla base model (no system prompt)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.reset:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False) as tf:
            tf.write(f"FROM {BASE_MODEL_FOR_MODELFILE}\n")
            modelfile_path = tf.name
        apply_modelfile(args.tag, modelfile_path)
        os.unlink(modelfile_path)
        return

    records = load_feedback(args.feedback_log)
    logger.info(f"Loaded {len(records)} feedback records from {args.feedback_log}")
    if not records:
        logger.warning(
            "No feedback recorded yet. Feedback logging is opt-in — set "
            "HARA_ADAPTIVE_FEEDBACK=1 before running Stage 1/3/4/5/6 to accumulate it."
        )
        return

    system_prompt = distill_system_prompt(records)
    logger.info(f"Distilled system prompt:\n{'-'*60}\n{system_prompt}\n{'-'*60}")

    if args.dry_run or not args.apply:
        logger.info("Dry run only (pass --apply to update the Ollama tag).")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False) as tf:
        modelfile_path = tf.name
    write_modelfile(system_prompt, modelfile_path)
    apply_modelfile(args.tag, modelfile_path)
    os.unlink(modelfile_path)


if __name__ == "__main__":
    main()

"""
Stage 2 V2: Evidence-Grounded Self-Verification — Orchestrator
================================================================
Project: HARA — Hallucination-Aware Retrieval Agent

Public entry point for Stage 2 V2. Wires together:

    Stage_2_ClaimExtractor   -> decompose the answer into atomic claims
    Stage_2_EvidenceMatcher  -> rank the already-retrieved passages per claim
    Stage_2_NLIVerifier      -> score each (claim, best evidence) pair, zero-shot
    Stage_2_Aggregator       -> per-claim diagnosis + overall verification report

This replaces Stage_2_Verifier_GPU.py's fine-tuned classifier entirely. The
call-site shape is deliberately similar to the old verify() (same
question / passages / answer inputs) so Stage 3/4/5 can be rewired to it
with a small, mechanical change — but the return value is now the full
structured VerificationReport (see Stage_2_Aggregator.VerificationReport),
not a bare label, and no HotpotQA-derived label of any kind is used at
inference time.

Usage:
    from Stage_2_Verifier import verify, load_nli_verifier

    nli_verifier = load_nli_verifier()
    report = verify(question, generated_answer, reranked_passages, nli_verifier)
    report["overall_status"]        # SUPPORTED | PARTIAL | UNSUPPORTED
    report["recommended_action"]    # KEEP | EXPAND | REWRITE | DECOMPOSE | COMPARE | ABSTAIN
    report["failure_reason"]        # explainable reason, or None
"""

from __future__ import annotations

from typing import Any, Dict, List

from Stage_2_Aggregator import aggregate, build_claim_verifications
from Stage_2_ClaimExtractor import extract_claims
from Stage_2_EvidenceMatcher import DEFAULT_TOP_K, match_evidence_batch
from Stage_2_NLIVerifier import NLIVerifier, load_nli_verifier  # re-exported

__all__ = [
    "verify", "load_nli_verifier", "NLIVerifier",
    "verify_legacy", "load_verifier", "build_verify_context", "VERIFIER_PATH",
    "normalize_verification", "legacy_scores",
]


def verify(
    question: str,
    generated_answer: str,
    passages: List[Dict[str, Any]],
    nli_verifier: NLIVerifier,
    top_k_evidence: int = DEFAULT_TOP_K,
    weights: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Evidence-grounded, question-aware self-verification for one (question,
    answer) pair — five dimensions (evidence support, question coverage,
    answer completeness, answer type match, contradiction), see
    Stage_2_Aggregator for the weighting and decision policy.

    `passages` should be the passages actually used for generation (Stage
    1/3/4's reranked top-k) — evidence matching ranks within this set per
    claim rather than issuing new retrieval, per the efficiency requirement
    that no claim be compared against a full corpus.

    `weights` optionally overrides Stage_2_Aggregator.DEFAULT_WEIGHTS.

    Returns a JSON-serializable dict — see Stage_2_Aggregator.VerificationReport
    for the exact field list.
    """
    claims = extract_claims(generated_answer, question=question)
    claim_texts = [c.text for c in claims]

    evidence_matches = match_evidence_batch(claim_texts, passages, top_k=top_k_evidence)

    premises = [matches[0].text if matches else "" for matches in evidence_matches]
    nli_pairs = list(zip(premises, claim_texts))
    nli_results = nli_verifier.score_batch(nli_pairs)

    all_evidence_text = "\n".join(p["text"] for p in passages)
    claim_verifications = build_claim_verifications(
        claims, evidence_matches, nli_results, all_evidence_text, question=question,
    )

    report = aggregate(claim_verifications, question=question, answer=generated_answer, weights=weights)
    return report.to_dict()


# ─────────────────────────────────────────────
# LEGACY COMPATIBILITY LAYER
# Lets Stage 3/4/5/6 switch off Stage_2_Verifier_GPU by changing only their
# import line, not their call-site logic, during the migration. New code
# should call verify()/load_nli_verifier() directly instead.
# ─────────────────────────────────────────────

VERIFIER_PATH = "verifier_model.pt"  # unused by the new pipeline; kept only
                                      # so `load_verifier(VERIFIER_PATH)` call
                                      # sites don't need to change


def load_verifier(model_path: str = None):
    """Old call sites do `verifier_model, verifier_tokenizer = load_verifier(
    VERIFIER_PATH)`, then gate verification on `verifier_tokenizer is not
    None` (Stage 3/4/5 all do this). Returns the same NLIVerifier instance
    in both slots — verify_legacy() below only ever uses the first — so
    that gate stays truthy and those call sites need no other change."""
    nli_verifier = load_nli_verifier()
    return nli_verifier, nli_verifier


_LEGACY_PASSAGE_SEP = "\n---PASSAGE---\n"


def build_verify_context(passages: List[Dict[str, Any]], answer: str, top_n: int = 5) -> str:
    """Same reordering behavior as Stage_2_Verifier_GPU.build_verify_context()
    (answer-containing passages first), returned as a single string so old
    call sites (`ctx = build_verify_context(...)`, passed straight into
    verify()/verify_legacy(), never inspected otherwise) need no change.

    Passages are joined with a recoverable separator rather than a plain
    space: verify_legacy() splits on it to restore individual passage
    boundaries. Concatenating unrelated passages into one blob measurably
    hurts the NLI model — a short claim can flip from ENTAILED to
    CONTRADICTED once an unrelated trailing sentence about a different
    entity gets appended to the same premise (confirmed empirically during
    Stage 3 integration testing). Keeping passages separate avoids that.
    """
    a_lower = answer.lower().strip()
    containing = [p for p in passages if a_lower in p["text"].lower()]
    others = [p for p in passages if a_lower not in p["text"].lower()]
    ordered = (containing + others)[:top_n]
    return _LEGACY_PASSAGE_SEP.join(p["text"] for p in ordered)


def legacy_scores(label: str, support_score: float) -> Dict[str, float]:
    """Synthesizes a SUPPORTED/PARTIAL/UNSUPPORTED probability triple from
    support_score for old call sites that read verification["scores"] (only
    the "SUPPORTED" entry is ever actually consumed downstream, for
    LOW_SUPPORT_THRESHOLD comparisons and ROC-AUC — the PARTIAL/UNSUPPORTED
    split below is a reasonable, non-authoritative fill-in)."""
    remaining = max(0.0, 1.0 - support_score)
    partial_share = {"UNSUPPORTED": 0.25, "PARTIAL": 0.75, "SUPPORTED": 0.5}[label]
    return {
        "SUPPORTED": round(support_score, 4),
        "PARTIAL": round(remaining * partial_share, 4),
        "UNSUPPORTED": round(remaining * (1 - partial_share), 4),
    }


def verify_legacy(
    context: str,
    answer: str,
    model: NLIVerifier,
    tokenizer: Any = None,
    question: str = None,
) -> Dict[str, Any]:
    """
    Drop-in replacement for Stage_2_Verifier_GPU.verify(context, answer,
    model, tokenizer, question=...). `model` must be an NLIVerifier (from
    load_verifier() above); `tokenizer` is accepted only for positional
    compatibility and ignored.

    `context` here is whatever build_verify_context() produced. It's split
    back into individual passages on that function's separator, so
    per-claim evidence matching still ranks among the real distinct
    passages instead of one concatenated blob — prefer verify() directly
    with the real passage list for new call sites regardless, since this
    round-trip is a compatibility convenience, not the primary path.

    Returns the old shape (label/confidence/scores) plus two additive keys
    (support_score, hallucination_probability) already in the shape the
    integration brief asked for, and the full new report under
    "verification_report" for callers ready to start reading it.
    """
    segments = [s.strip() for s in context.split(_LEGACY_PASSAGE_SEP) if s.strip()]
    passages = [{"title": f"context_{i}", "text": s} for i, s in enumerate(segments)] or [{"title": "context", "text": context}]
    report = verify(question or "", answer, passages, model)

    label = report["overall_status"]
    support_score = report["support_score"]
    return {
        "label": label,
        "confidence": report["overall_confidence"],
        "support_score": support_score,
        "hallucination_probability": round(1 - support_score, 4),
        "scores": legacy_scores(label, support_score),
        "verification_report": report,
    }


def normalize_verification(verification: dict) -> dict:
    """
    Normalizes a verification dict to the new report shape (overall_status /
    overall_confidence / support_score / coverage / failure_reason /
    recommended_action / claims), whether it came directly from verify()
    (already this shape) or from verify_legacy() — which may carry a
    "verification_report" nested inside it, and whose top-level label/
    confidence can reflect a demotion (e.g. Stage 3's LLM-judge role-check)
    applied *after* verify_legacy() returned, which the nested report has no
    way to know about. When unwrapping a legacy dict, this takes label/
    confidence from the OUTER (possibly-demoted) fields, not the stale
    nested report, so a demotion already applied is never silently lost —
    the exact class of bug this project already hit once before with the
    old verifier.

    Shared by any caller that might receive either shape (Stage 4, Stage 5).
    """
    if "overall_status" in verification:
        return verification

    nested = verification.get("verification_report") or {}
    label = verification["label"]
    demoted = label != nested.get("overall_status")
    return {
        **nested,
        "overall_status": label,
        "overall_confidence": verification["confidence"],
        "support_score": verification.get("support_score", nested.get("support_score", 0.0)),
        "coverage": nested.get("coverage", 1.0),
        "claims": nested.get("claims", []),
        "failure_reason": ("WRONG_ENTITY" if demoted else nested.get("failure_reason")),
        "recommended_action": ("COMPARE" if demoted else nested.get("recommended_action", "KEEP")),
    }

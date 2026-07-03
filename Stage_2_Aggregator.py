"""
Stage 2 V2.1: Aggregator
=========================
Project: HARA — Hallucination-Aware Retrieval Agent

Combines FIVE independent dimensions into one VerificationReport:

    1. Evidence Support     (40%) — per-claim NLI entailment (V2's original mechanism)
    2. Question Coverage    (25%) — does the answer's content address what was asked
    3. Answer Completeness  (20%) — does it contain all required parts (comparison/list)
    4. Answer Type Match    (10%) — does the answer's surface form match the expected type
    5. Contradiction         (5%) — penalty if any claim is actively contradicted

V2's original weakness: evidence-only aggregation could label an answer
SUPPORTED when it's factually entailed but doesn't answer the question (a
YEAR question answered with a person's name, if that name happens to be
entailed by some passage) — or reject a correct answer over a narrow NLI
miss. Dimensions 2-5 exist specifically to catch what pure entailment
can't: the existing per-claim NLI diagnosis is the FOUNDATION (unchanged),
and question-aware checks act as corrective overrides layered on top —
never used to redeem a genuine contradiction, always able to catch a
type/coverage miss the NLI signal alone would pass.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from Stage_2_AnswerTypeValidator import score_answer_type
from Stage_2_ClaimExtractor import Claim, extract_entities, extract_numbers, extract_years
from Stage_2_EvidenceMatcher import MIN_RELEVANCE_SCORE, EvidenceMatch
from Stage_2_NLIVerifier import NLIResult
from Stage_2_QuestionCoverageScorer import score_completeness, score_question_coverage
from Stage_2_QuestionIntent import QuestionIntent, analyze_question

logger = logging.getLogger("stage2_v2")

AMBIGUITY_RELEVANCE_DELTA = 1.0   # top-2 evidence scores this close, from
                                  # different passages, suggests ambiguity

# Part 5 — configurable dimension weights, must sum to 1.0.
DEFAULT_WEIGHTS = {
    "evidence_support": 0.40,
    "question_coverage": 0.25,
    "answer_completeness": 0.20,
    "answer_type": 0.10,
    "contradiction": 0.05,
}

# Escalation thresholds — a whole-answer dimension score below these
# overrides the per-claim NLI verdict, since a well-entailed claim that
# doesn't answer the question or has the wrong type is still a bad answer.
ANSWER_TYPE_ESCALATION_THRESHOLD = 0.30
COVERAGE_ESCALATION_THRESHOLD = 0.30
COMPLETENESS_PARTIAL_THRESHOLD = 0.70

# Part 6 — expanded failure taxonomy, each with a recommended_action (Part 9).
_ACTION_BY_REASON = {
    "ANSWER_TYPE_MISMATCH": "REWRITE",
    "QUESTION_NOT_ANSWERED": "REWRITE",
    "MISSING_REQUIRED_ENTITY": "EXPAND",
    "MISSING_REQUIRED_NUMBER": "EXPAND",
    "MISSING_REQUIRED_DATE": "EXPAND",
    "WRONG_ENTITY": "REWRITE",
    "WRONG_YEAR": "REWRITE",
    "WRONG_NUMBER": "REWRITE",
    "PARTIAL_COMPARISON": "COMPARE",
    "CONTRADICTED_BY_EVIDENCE": "ABSTAIN",
    "INSUFFICIENT_EVIDENCE": "EXPAND",
    "NO_RELEVANT_PASSAGE": "EXPAND",
    "MULTIPLE_CONFLICTING_PASSAGES": "COMPARE",
    "UNKNOWN": "REWRITE",
    None: "KEEP",
}


@dataclass
class ClaimVerification:
    claim: str
    kind: str
    best_evidence: Optional[str]
    label: str            # SUPPORTED | PARTIAL | UNSUPPORTED
    confidence: float
    reason: str
    verdict: str           # raw ENTAILED | NEUTRAL | CONTRADICTED
    entailment_prob: float
    contradiction_prob: float = 0.0
    evidence_score: Optional[float] = None


@dataclass
class VerificationReport:
    overall_status: str
    overall_confidence: float
    support_score: float
    coverage: float
    claims_total: int
    claims_supported: int
    claims_partial: int
    claims_unsupported: int
    claims: List[ClaimVerification]
    failure_reason: Optional[str]
    recommended_action: str
    explanation: str

    # Additive (Stage 2 V2.1) — the five dimensions and what drove them.
    evidence_support_score: float = 0.0
    question_coverage_score: float = 0.0
    answer_completeness_score: float = 0.0
    answer_type_score: float = 0.0
    contradiction_score: float = 1.0
    question_intent: Optional[Dict[str, Any]] = None
    weights_used: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────
# PER-CLAIM DIAGNOSIS
# ─────────────────────────────────────────────

def _diagnose_contradiction(claim: Claim, best_match: Optional[EvidenceMatch]) -> str:
    if best_match is None:
        return "CONTRADICTED_BY_EVIDENCE"
    ev_text = best_match.text

    claim_years, ev_years = extract_years(claim.text), extract_years(ev_text)
    if claim_years and ev_years and set(claim_years) - set(ev_years):
        return "WRONG_YEAR"

    claim_nums, ev_nums = extract_numbers(claim.text), extract_numbers(ev_text)
    if claim_nums and ev_nums and set(claim_nums) - set(ev_nums):
        return "WRONG_NUMBER"

    claim_entities = set(extract_entities(claim.text))
    ev_entities = set(extract_entities(ev_text))
    if claim_entities and not (claim_entities & ev_entities):
        return "WRONG_ENTITY"

    return "CONTRADICTED_BY_EVIDENCE"


def _diagnose_neutral(claim: Claim, best_match: Optional[EvidenceMatch], all_evidence_text: str) -> str:
    """Distinguishes which specific kind of information is missing —
    entity, number, or date — rather than one generic reason, per Part 6."""
    years = extract_years(claim.text)
    nums = extract_numbers(claim.text)
    entities = extract_entities(claim.text)

    has_relevant_evidence = best_match is not None and best_match.relevance_score >= MIN_RELEVANCE_SCORE
    if not has_relevant_evidence:
        return "NO_RELEVANT_PASSAGE"

    ev_lower = all_evidence_text.lower()
    if years and not any(y in all_evidence_text for y in years):
        return "MISSING_REQUIRED_DATE"
    if nums and not any(n in all_evidence_text for n in nums):
        return "MISSING_REQUIRED_NUMBER"
    if entities and not any(e.lower() in ev_lower for e in entities):
        return "MISSING_REQUIRED_ENTITY"

    return "INSUFFICIENT_EVIDENCE"


def _detect_ambiguity(matches: List[EvidenceMatch]) -> bool:
    if len(matches) < 2:
        return False
    top, second = matches[0], matches[1]
    if top.relevance_score <= 0:
        return False
    close = abs(top.relevance_score - second.relevance_score) < AMBIGUITY_RELEVANCE_DELTA
    different_source = top.title != second.title
    return close and different_source


def diagnose_claim(
    claim: Claim,
    matches: List[EvidenceMatch],
    nli_result: NLIResult,
    all_evidence_text: str,
    question: str = "",
) -> ClaimVerification:
    """Assigns a per-claim SUPPORTED/PARTIAL/UNSUPPORTED label and an
    explainable failure reason (Evidence Support dimension only — question
    coverage/completeness/type checks happen once, whole-answer, in
    aggregate())."""
    best_match = matches[0] if matches else None

    if nli_result.verdict == "ENTAILED":
        label, reason = "SUPPORTED", None
    elif nli_result.verdict == "CONTRADICTED":
        label, reason = "UNSUPPORTED", _diagnose_contradiction(claim, best_match)
    else:  # NEUTRAL
        reason = _diagnose_neutral(claim, best_match, all_evidence_text)
        label = "PARTIAL" if reason == "INSUFFICIENT_EVIDENCE" else "UNSUPPORTED"

    if label != "SUPPORTED" and _detect_ambiguity(matches):
        reason = "MULTIPLE_CONFLICTING_PASSAGES"

    return ClaimVerification(
        claim=claim.text,
        kind=claim.kind,
        best_evidence=(f"{best_match.title}: {best_match.text[:200]}" if best_match else None),
        label=label,
        confidence=nli_result.confidence,
        reason=reason,
        verdict=nli_result.verdict,
        entailment_prob=nli_result.entailment_prob,
        contradiction_prob=nli_result.contradiction_prob,
        evidence_score=(best_match.relevance_score if best_match else None),
    )


def build_claim_verifications(
    claims: List[Claim],
    evidence_matches: List[List[EvidenceMatch]],
    nli_results: List[NLIResult],
    all_evidence_text: str,
    question: str = "",
) -> List[ClaimVerification]:
    return [
        diagnose_claim(claim, matches, nli_result, all_evidence_text, question)
        for claim, matches, nli_result in zip(claims, evidence_matches, nli_results)
    ]


# ─────────────────────────────────────────────
# OVERALL AGGREGATION
# ─────────────────────────────────────────────

def _base_status_and_reason(claim_verifications: List[ClaimVerification]):
    """Evidence Support dimension's own severity-based verdict — unchanged
    from V2's original policy. Any claim CONTRADICTED is treated as severe
    regardless of whether it's the primary or a secondary/elaboration claim
    (Part 9: contradiction -> ABSTAIN is a stronger response than a mere
    downgrade), which is a deliberate change from V2's original behavior of
    only downgrading a contradicted secondary claim to PARTIAL."""
    primary = claim_verifications[0]
    secondary = claim_verifications[1:]

    contradicted = [c for c in claim_verifications if c.verdict == "CONTRADICTED"]
    if contradicted:
        deciding = max(contradicted, key=lambda c: c.contradiction_prob)
        return "UNSUPPORTED", (deciding.reason or "CONTRADICTED_BY_EVIDENCE"), deciding

    if primary.label == "SUPPORTED":
        bad_secondary = [c for c in secondary if c.label != "SUPPORTED"]
        if not bad_secondary:
            return "SUPPORTED", None, primary
        deciding = min(bad_secondary, key=lambda c: c.confidence)
        return "PARTIAL", deciding.reason, deciding

    return "UNSUPPORTED", primary.reason, primary


def aggregate(
    claim_verifications: List[ClaimVerification],
    question: str = "",
    answer: str = "",
    weights: Optional[Dict[str, float]] = None,
) -> VerificationReport:
    weights = weights or DEFAULT_WEIGHTS
    intent = analyze_question(question) if question else QuestionIntent(
        expected_type="UNKNOWN", focus_phrase="", requires_comparison=False,
        requires_multi_hop=False, expected_cardinality=1,
    )

    if not claim_verifications:
        report = VerificationReport(
            overall_status="UNSUPPORTED", overall_confidence=0.0, support_score=0.0,
            coverage=0.0, claims_total=0, claims_supported=0, claims_partial=0,
            claims_unsupported=0, claims=[], failure_reason="NO_RELEVANT_PASSAGE",
            recommended_action="ABSTAIN", explanation="No claims could be extracted from the answer.",
            evidence_support_score=0.0, question_coverage_score=0.0,
            answer_completeness_score=0.0, answer_type_score=0.0, contradiction_score=1.0,
            question_intent=intent.to_dict(), weights_used=dict(weights),
        )
        _log_verification(question, answer, report)
        return report

    primary = claim_verifications[0]
    secondary = claim_verifications[1:]

    base_status, base_reason, deciding = _base_status_and_reason(claim_verifications)

    # ── Whole-answer dimensions (independent of per-claim evidence) ──
    answer_type_score = score_answer_type(intent.expected_type, answer)
    question_coverage_score = score_question_coverage(intent, answer)
    completeness_score = score_completeness(intent, answer)
    max_contradiction_prob = max((c.contradiction_prob for c in claim_verifications), default=0.0)
    contradiction_score = 1.0 - max_contradiction_prob

    if secondary:
        secondary_support = sum(c.entailment_prob for c in secondary) / len(secondary)
        evidence_support_score = 0.7 * primary.entailment_prob + 0.3 * secondary_support
    else:
        evidence_support_score = primary.entailment_prob

    # ── Corrective overrides: a well-entailed claim that's the wrong type or
    # doesn't address the question is still a bad answer — these can only
    # escalate severity, never redeem an already-detected contradiction. ──
    overall_status, failure_reason = base_status, base_reason
    if base_status != "UNSUPPORTED" or base_reason != "CONTRADICTED_BY_EVIDENCE":
        if answer_type_score < ANSWER_TYPE_ESCALATION_THRESHOLD:
            overall_status, failure_reason = "UNSUPPORTED", "ANSWER_TYPE_MISMATCH"
        elif question_coverage_score < COVERAGE_ESCALATION_THRESHOLD:
            overall_status, failure_reason = "UNSUPPORTED", "QUESTION_NOT_ANSWERED"
        elif (intent.requires_comparison and completeness_score < COMPLETENESS_PARTIAL_THRESHOLD
              and base_status == "SUPPORTED"):
            overall_status, failure_reason = "PARTIAL", "PARTIAL_COMPARISON"

    if failure_reason is None and base_reason is not None:
        failure_reason = base_reason

    claims_supported = sum(1 for c in claim_verifications if c.label == "SUPPORTED")
    claims_partial = sum(1 for c in claim_verifications if c.label == "PARTIAL")
    claims_unsupported = sum(1 for c in claim_verifications if c.label == "UNSUPPORTED")
    non_supported_count = claims_partial + claims_unsupported

    if overall_status == "SUPPORTED":
        recommended_action = "KEEP"
    elif failure_reason == "CONTRADICTED_BY_EVIDENCE":
        recommended_action = "ABSTAIN"
    elif non_supported_count >= 2:
        recommended_action = "ABSTAIN"
    else:
        recommended_action = _ACTION_BY_REASON.get(failure_reason, "REWRITE")

    claims_with_evidence = sum(1 for c in claim_verifications if c.evidence_score is not None)
    evidence_coverage = claims_with_evidence / len(claim_verifications)

    # Part 7 — composite confidence, weighted across all five dimensions.
    overall_confidence = round(
        weights["evidence_support"] * evidence_support_score
        + weights["question_coverage"] * question_coverage_score
        + weights["answer_completeness"] * completeness_score
        + weights["answer_type"] * answer_type_score
        + weights["contradiction"] * contradiction_score,
        4,
    )

    explanation = _build_explanation(
        overall_status, deciding, failure_reason, answer_type_score, question_coverage_score,
    )

    report = VerificationReport(
        overall_status=overall_status,
        overall_confidence=overall_confidence,
        support_score=round(evidence_support_score, 4),
        coverage=round(evidence_coverage, 4),
        claims_total=len(claim_verifications),
        claims_supported=claims_supported,
        claims_partial=claims_partial,
        claims_unsupported=claims_unsupported,
        claims=claim_verifications,
        failure_reason=failure_reason,
        recommended_action=recommended_action,
        explanation=explanation,
        evidence_support_score=round(evidence_support_score, 4),
        question_coverage_score=round(question_coverage_score, 4),
        answer_completeness_score=round(completeness_score, 4),
        answer_type_score=round(answer_type_score, 4),
        contradiction_score=round(contradiction_score, 4),
        question_intent=intent.to_dict(),
        weights_used=dict(weights),
    )
    _log_verification(question, answer, report)
    return report


def _build_explanation(
    overall_status: str, deciding: ClaimVerification, failure_reason: Optional[str],
    answer_type_score: float, question_coverage_score: float,
) -> str:
    if overall_status == "SUPPORTED":
        return f"All claims are entailed by retrieved evidence (e.g. \"{deciding.claim}\")."
    if failure_reason == "ANSWER_TYPE_MISMATCH":
        return f"The answer's type does not match what the question expects (type score {answer_type_score:.2f})."
    if failure_reason == "QUESTION_NOT_ANSWERED":
        return f"The answer does not address what was asked (coverage score {question_coverage_score:.2f})."
    role = "core" if deciding.kind == "primary" else "supporting"
    return (
        f"The {role} claim \"{deciding.claim}\" was judged {deciding.verdict.lower()} "
        f"by the evidence (confidence {deciding.confidence:.2f}); reason: {failure_reason}."
    )


# ─────────────────────────────────────────────
# LOGGING (Part 8 — debugging only, not used for control flow)
# ─────────────────────────────────────────────

def _log_verification(question: str, answer: str, report: VerificationReport) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "question=%r answer=%r claims=%s top_evidence=%s nli=%s "
        "coverage=%.2f type_score=%.2f contradiction=%.2f final_score=%.2f "
        "reason=%s action=%s",
        question, answer,
        [c.claim for c in report.claims],
        [c.best_evidence for c in report.claims],
        [(c.claim, c.verdict, round(c.entailment_prob, 3)) for c in report.claims],
        report.question_coverage_score, report.answer_type_score, report.contradiction_score,
        report.overall_confidence, report.failure_reason, report.recommended_action,
    )

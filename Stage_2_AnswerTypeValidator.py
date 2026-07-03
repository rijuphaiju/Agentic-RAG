"""
Stage 2 V2.1: Answer Type Validator
=====================================
Project: HARA — Hallucination-Aware Retrieval Agent

Scores whether a candidate answer's surface form matches the type the
question expects (per Stage_2_QuestionIntent.QuestionIntent) — e.g. a YEAR
question answered with "Pop Warner" scores near zero regardless of whether
that answer is entailed by evidence somewhere. Pure regex/heuristic, no LLM,
reusing the entity classifier already built for Stage 2 V2's claim
diagnosis so person/organization/location detection is defined once.
"""

from __future__ import annotations

import re

from Stage_2_ClaimExtractor import classify_entity, extract_dates, extract_entities, extract_numbers, extract_years

_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}
_REFUSAL_RE = re.compile(
    r"\b(i\s+cannot|i\s+can't|cannot\s+confidently|not\s+specified|no\s+information|"
    r"cannot\s+determine|not\s+clear|not\s+available|unclear|insufficient)\b",
    re.IGNORECASE,
)


def is_refusal_like(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer))


def score_answer_type(expected_type: str, answer: str) -> float:
    """Returns a 0.0-1.0 heuristic score for whether `answer` surface-matches
    `expected_type`. 1.0 = clearly the right shape, 0.0 = clearly wrong."""
    answer = (answer or "").strip()
    if not answer:
        return 0.0
    if is_refusal_like(answer):
        return 0.0

    a_lower = answer.lower()

    if expected_type == "YEAR":
        return 1.0 if extract_years(answer) else (0.3 if extract_numbers(answer) else 0.0)

    if expected_type == "MONTH":
        return 1.0 if any(m in a_lower for m in _MONTHS) else 0.0

    if expected_type in ("NUMBER", "COUNT"):
        return 1.0 if (extract_numbers(answer) or extract_years(answer)) else 0.0

    if expected_type == "DATE":
        if extract_dates(answer):
            return 1.0
        if any(m in a_lower for m in _MONTHS) or extract_years(answer):
            return 0.6
        return 0.0

    if expected_type == "BOOLEAN":
        return 1.0 if a_lower.rstrip(".") in ("yes", "no") else 0.3

    if expected_type == "PERSON":
        ents = extract_entities(answer)
        types = {classify_entity(e) for e in ents}
        if "person" in types:
            return 1.0
        if "organization" in types:
            # "Who did X?" grammatically allows an agent that isn't a human
            # ("Who formed WINNER?" -> "YG Entertainment") — a near-match,
            # not a near-miss.
            return 0.85
        return 0.4 if ents else 0.1

    if expected_type == "ORGANIZATION":
        ents = extract_entities(answer)
        if any(classify_entity(e) == "organization" for e in ents):
            return 1.0
        return 0.4 if ents else 0.1

    if expected_type in ("LOCATION", "CITY", "COUNTRY"):
        ents = extract_entities(answer)
        if any(classify_entity(e) == "location" for e in ents):
            return 1.0
        return 0.4 if ents else 0.1

    if expected_type in ("TITLE", "EVENT", "LIST", "COMPARISON"):
        return 1.0 if answer else 0.5

    return 0.5  # UNKNOWN — no strong basis to penalize or reward

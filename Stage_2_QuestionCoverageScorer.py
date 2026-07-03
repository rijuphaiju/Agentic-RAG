"""
Stage 2 V2.1: Question Coverage & Completeness Scorer
=======================================================
Project: HARA — Hallucination-Aware Retrieval Agent

Two related, independent-of-evidence dimensions:

  score_question_coverage() — does the answer's CONTENT address what was
    asked (population question needs a number; month question needs a
    month), regardless of whether that content is actually true? Builds on
    AnswerTypeValidator's type match, refusal-checked, with a comparison-
    specific floor.

  score_completeness() — does the answer contain ALL required parts for
    THIS question's shape? A single correct fact is complete for a simple
    factoid question; a comparison needs a resolved side, a LIST needs
    multiple items. This is deliberately distinct from type/coverage: those
    ask "is it the right kind of content", this asks "is there enough of it".
"""

from __future__ import annotations

import re

from Stage_2_AnswerTypeValidator import is_refusal_like, score_answer_type
from Stage_2_QuestionIntent import QuestionIntent

_LIST_SPLIT_RE = re.compile(r",|\band\b", re.IGNORECASE)
_HEDGE_BOTH_RE = re.compile(r"\b(both|unclear|not\s+sure|either|ambiguous)\b", re.IGNORECASE)


def score_question_coverage(intent: QuestionIntent, answer: str) -> float:
    """0.0-1.0: does the answer's content address the informational need?
    Independent of whether that content is evidence-supported."""
    answer = (answer or "").strip()
    if not answer or is_refusal_like(answer):
        return 0.0

    base = score_answer_type(intent.expected_type, answer)

    if intent.requires_comparison:
        # A comparison question needs an attempted resolution, not just any
        # entity mention — an unresolved hedge ("both", "unclear") covers
        # the topic without answering the question.
        if _HEDGE_BOTH_RE.search(answer):
            return min(base, 0.3)
        return max(base, 0.6) if answer else 0.0

    return base


def score_completeness(intent: QuestionIntent, answer: str) -> float:
    """0.0-1.0: does the answer contain enough parts for this question's
    shape? Single-fact questions are complete with just one correct fact —
    this only penalizes comparison/list questions missing a required part."""
    answer = (answer or "").strip()
    if not answer:
        return 0.0
    if is_refusal_like(answer):
        return 0.0

    if intent.requires_comparison:
        if _HEDGE_BOTH_RE.search(answer):
            return 0.4  # names the topic but doesn't commit to one side
        return 1.0

    if intent.expected_type == "LIST" or intent.expected_cardinality > 1:
        items = [p.strip() for p in _LIST_SPLIT_RE.split(answer) if p.strip()]
        found = len(items)
        return min(1.0, found / intent.expected_cardinality) if intent.expected_cardinality else 1.0

    return 1.0

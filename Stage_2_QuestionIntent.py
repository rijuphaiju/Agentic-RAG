"""
Stage 2 V2.1: Question Intent Analyzer
========================================
Project: HARA — Hallucination-Aware Retrieval Agent

Infers what a question is actually asking for — expected_type, the focus
phrase, and whether it needs a comparison or multi-hop resolution — purely
via regex/heuristics. No LLM, no training. Used by AnswerTypeValidator and
QuestionCoverageScorer to judge an answer against the question itself, not
just against retrieved evidence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

EXPECTED_TYPES = (
    "PERSON", "ORGANIZATION", "LOCATION", "COUNTRY", "CITY", "DATE", "YEAR",
    "MONTH", "NUMBER", "COUNT", "BOOLEAN", "TITLE", "EVENT", "COMPARISON",
    "LIST", "UNKNOWN",
)

_COMPARATIVE_WORDS = (
    "older", "younger", "first", "last", "earlier", "later", "before", "after",
    "more", "less", "higher", "lower", "longer", "shorter", "bigger", "smaller",
    "taller", "same", "both",
)
_LEADING_AUX_RE = re.compile(
    r"^(were|was|is|are|did|does|do|have|has|can|could|would|will)\b", re.IGNORECASE
)
_OR_COMPARISON_RE = re.compile(r"\b[A-Z][\w.']*(?:\s+[A-Z][\w.']*)*\s+or\s+[A-Z][\w.']*")

_WH_TYPE_RULES = (
    # (regex, expected_type)
    (r"\bwhat\s+year\b|\bwhich\s+year\b|\bin\s+what\s+year\b", "YEAR"),
    (r"\bwhat\s+month\b|\bwhich\s+month\b", "MONTH"),
    (r"\bhow\s+many\b", "COUNT"),
    (r"\bwhat\s+population\b|\bhow\s+much\b|\bhow\s+(?:tall|long|far|old|high|wide|deep|large|big|small)\b", "NUMBER"),
    (r"\bwhich\s+country\b|\bwhat\s+country\b", "COUNTRY"),
    (r"\bwhich\s+city\b|\bwhat\s+city\b", "CITY"),
    (r"\bwhich\s+organi[sz]ation\b|\bwhat\s+organi[sz]ation\b|\bwhich\s+company\b|\bwhat\s+company\b|"
     r"\bwhich\s+band\b|\bwhich\s+team\b", "ORGANIZATION"),
    (r"\bwhere\b", "LOCATION"),
    (r"\bwho\b|\bwhom\b|\bwhose\b", "PERSON"),
    (r"\bwhat\s+(?:is|was)\s+the\s+title\b|\btitle\s+of\b|\bwhat.{0,15}(?:film|movie|book|song|album|novel)\b.{0,15}"
     r"(?:title|name|called)\b", "TITLE"),
    (r"\bwhat\s+event\b|\bwhich\s+event\b", "EVENT"),
    (r"\bwhat\s+are\b|\bwhich\s+are\b|\bname\s+all\b|\blist\s+the\b", "LIST"),
    (r"\bwhen\b|\bwhat\s+date\b", "DATE"),
)

_FOCUS_RE = re.compile(
    r"\b(?:what|which|who|whom|whose|where|when|how\s+many|how\s+much)\s+"
    r"((?:[a-z]+\s+){0,3}[a-z]+)", re.IGNORECASE,
)


@dataclass
class QuestionIntent:
    expected_type: str
    focus_phrase: str
    requires_comparison: bool
    requires_multi_hop: bool
    expected_cardinality: int

    def to_dict(self) -> dict:
        return asdict(self)


def _is_comparison(question: str) -> bool:
    q = question.strip()
    if _OR_COMPARISON_RE.search(q):
        return True
    if _LEADING_AUX_RE.match(q) and any(w in q.lower() for w in _COMPARATIVE_WORDS):
        return True
    return False


def _detect_type(question: str, comparison: bool) -> str:
    if comparison:
        return "COMPARISON"
    q_lower = question.lower()
    # Prefer whichever wh-pattern matches EARLIEST in the question, not the
    # first rule in priority order — HotpotQA bridge questions often embed a
    # second "who"/"which" inside a relative clause well after the real
    # question word ("What government position was held by the woman WHO
    # portrayed..."), and that trailing one must not win.
    best_pos, best_type = None, None
    for pattern, etype in _WH_TYPE_RULES:
        m = re.search(pattern, q_lower)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos, best_type = m.start(), etype
    if best_type is not None:
        return best_type
    if _LEADING_AUX_RE.match(question.strip()):
        return "BOOLEAN"
    return "UNKNOWN"


def _extract_focus(question: str) -> str:
    m = _FOCUS_RE.search(question)
    if not m:
        return ""
    words = m.group(1).split()
    return " ".join(words[:3])


def _requires_multi_hop(question: str) -> bool:
    relative_markers = len(re.findall(r"\b(who|which|that)\b", question, re.IGNORECASE))
    return relative_markers >= 2 or len(question.split()) > 15


def analyze_question(question: str) -> QuestionIntent:
    """Deterministic, regex-only question-intent inference — no LLM."""
    comparison = _is_comparison(question)
    expected_type = _detect_type(question, comparison)
    return QuestionIntent(
        expected_type=expected_type,
        focus_phrase=_extract_focus(question),
        requires_comparison=comparison,
        requires_multi_hop=_requires_multi_hop(question),
        expected_cardinality=2 if comparison else (2 if expected_type == "LIST" else 1),
    )

"""
Stage 2 V2: Atomic Claim Extractor
===================================
Project: HARA — Hallucination-Aware Retrieval Agent

Splits a generated answer into independent, checkable atomic claims so that
Stage 2 V2 can verify each one against evidence separately, instead of
scoring an entire (possibly compound) answer as one unit.

Example (from the Stage 2 V2 design brief):
    "YG Entertainment formed WINNER in 2014."
    -> Claim("YG Entertainment formed WINNER.", kind="primary")
    -> Claim("WINNER formed in 2014.", kind="secondary", qualifier_type="temporal")

Decomposition is deterministic-first: conjunction splitting, then a small
registry of trailing-qualifier splitters (temporal, locative) that peel a
date/place phrase off the end of a clause and re-anchor it to the last
entity mentioned. An LLM decomposition call is used only as a fallback for
long or structurally complex answers no deterministic splitter reduces —
the same minority-fallback philosophy used throughout this project.

No dependency parser is used (none is part of this project's stack).
Secondary claims are therefore templated propositions ("WINNER formed in
2014") rather than fully reconstructed natural sentences ("WINNER was
formed in 2014") — NLI models are robust to this since they score
propositional content, not fluency (verified empirically against the
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli checkpoint used by Stage 2 V2).

This module also exposes the symbolic entity/number/date/year extraction
helpers used both for qualifier-splitting here and for failure-reason
diagnosis in Stage_2_Aggregator.py, so that logic is defined once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import ollama

from Stage_1_RAG_Pipeline import OLLAMA_MODEL

# ─────────────────────────────────────────────
# SYMBOLIC EXTRACTION HELPERS
# (shared with Stage_2_Aggregator.py for failure diagnosis)
# ─────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER_RE = re.compile(r"\b\d+\b")
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}\b"
)
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z'.]*(?:\s+[A-Z][a-zA-Z'.]*){0,3}\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS_LEADING = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "He", "She",
    "They", "We", "You", "His", "Her", "Their", "In", "On", "At", "Final", "Answer",
}

_ORG_SUFFIXES = (
    "University", "College", "Company", "Corporation", "Inc", "Ltd",
    "Party", "Organization", "Institute", "Association", "Church",
    "League", "Studio", "Records", "Films", "Productions", "Band", "Group",
    "Team", "Council", "Committee", "Entertainment", "Media", "Pictures",
    "Industries", "Holdings", "Enterprises", "Networks", "Broadcasting",
    "Agency", "Airlines", "Bank", "Publishing",
)
_LOCATION_SUFFIXES = (
    "City", "County", "Province", "State", "Island", "River", "Mountain",
    "Republic", "Kingdom", "Bay", "Lake", "Valley", "Coast",
)


def extract_entities(text: str) -> List[str]:
    """Heuristic proper-noun span extraction (no NER model dependency).
    Runs per-sentence so a span never merges the end of one sentence with
    the start of the next."""
    found: List[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        for m in _ENTITY_RE.finditer(sentence):
            span = m.group(0).strip().rstrip(".")
            words = span.split()
            while words and words[0] in _STOPWORDS_LEADING:
                words = words[1:]
            while words and words[-1] in _STOPWORDS_LEADING:
                words = words[:-1]
            if not words:
                continue
            span = " ".join(words)
            if len(span) < 3:
                continue
            found.append(span)
    return list(dict.fromkeys(found))


def extract_years(text: str) -> List[str]:
    return list(dict.fromkeys(_YEAR_RE.findall(text)))


def extract_numbers(text: str) -> List[str]:
    return list(dict.fromkeys(n for n in _NUMBER_RE.findall(text) if not _YEAR_RE.fullmatch(n)))


def extract_dates(text: str) -> List[str]:
    return list(dict.fromkeys(_DATE_RE.findall(text)))


def classify_entity(entity: str) -> str:
    """Heuristic person/organization/location/other classification, used
    only for the optional semantic_mismatch diagnostic — not NER-grade."""
    if any(entity.endswith(suf) or f" {suf}" in entity for suf in _ORG_SUFFIXES):
        return "organization"
    if any(entity.endswith(suf) or f" {suf}" in entity for suf in _LOCATION_SUFFIXES):
        return "location"
    if len(entity.split()) == 2:
        return "person"
    return "other"


# ─────────────────────────────────────────────
# CLAIM DATACLASS
# ─────────────────────────────────────────────

@dataclass
class Claim:
    text: str
    kind: str                          # "primary" | "secondary"
    qualifier_type: Optional[str] = None   # "temporal" | "locative" | None
    method: str = "single_claim"       # how this claim was produced (auditability)


# ─────────────────────────────────────────────
# CONJUNCTION SPLITTING (independent clauses)
# ─────────────────────────────────────────────

_STRONG_SPLIT_RE = re.compile(
    r"(?:(?<=[.!?])\s+|\s+and also\s+|,\s+and also\s+|\s+as well as\s+|,\s+and\s+)",
    re.IGNORECASE,
)
# A bare " and " with no comma is ambiguous — it might join two independent
# clauses ("X did A and served as B") or be part of a compound proper noun
# ("Rock and Roll Hall of Fame", "Johnson and Johnson"). Only split on it
# when NOT flanked by a capitalized word on both sides, which is the
# signature of the compound-name case.
_BARE_AND_RE = re.compile(r"\b(\w+)\s+and\s+(\w+)\b")
_LEADING_CONJ_RE = re.compile(r"^(?:and|also|but|while|with|as well as)\s+", re.IGNORECASE)
MIN_CLAUSE_WORDS = 3


def _split_bare_and(text: str) -> List[str]:
    match = _BARE_AND_RE.search(text)
    if not match or (match.group(1)[:1].isupper() and match.group(2)[:1].isupper()):
        return [text]
    and_start = text.index(" and ", match.start())
    return [text[:and_start], text[and_start + 5:]]


def _split_conjunctions(answer: str) -> List[str]:
    parts = [p.strip(" ,.") for p in _STRONG_SPLIT_RE.split(answer) if p.strip(" ,.")]
    expanded: List[str] = []
    for part in parts:
        expanded.extend(_split_bare_and(part))
    parts = [p.strip(" ,.") for p in expanded if p.strip(" ,.")]
    parts = [_LEADING_CONJ_RE.sub("", p).strip() for p in parts]
    parts = [p for p in parts if len(p.split()) >= MIN_CLAUSE_WORDS]
    return parts if parts else [answer.strip()]


# ─────────────────────────────────────────────
# TRAILING-QUALIFIER SPLITTER REGISTRY
# Each splitter: (clause) -> Optional[(main, secondary_claim, qualifier_type)]
# Registering a new splitter = adding one function here; extract_claims()
# never needs to change.
# ─────────────────────────────────────────────

_SPLITTER_REGISTRY: List[Tuple[str, Callable[[str], Optional[Tuple[str, str, str]]]]] = []


def register_splitter(name: str):
    def _decorator(fn):
        _SPLITTER_REGISTRY.append((name, fn))
        return fn
    return _decorator


_TEMPORAL_TRAILING_RE = re.compile(
    r"^(?P<main>.*\S)\s+(?P<qualifier>(?:in|on|during|since|from|by)\s+"
    r"(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}|\d{3,4}))\s*[.]?\s*$",
    re.IGNORECASE,
)
_LOCATIVE_TRAILING_RE = re.compile(
    r"^(?P<main>.*\S)\s+(?P<qualifier>(?:in|at|near)\s+"
    r"[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,3})\s*[.]?\s*$"
)


def _anchor_entity_and_verb(main: str) -> Optional[Tuple[str, str]]:
    """Finds the last entity span in `main` (the qualifier's natural anchor)
    and the verb phrase between the first and last entity spans, if two
    distinct entities are present. Returns None when there's no second
    entity to anchor the secondary claim to — in that case the caller
    should NOT split (the qualifier stays attached to the primary claim)."""
    entities = extract_entities(main)
    if len(entities) < 2:
        return None
    first, last = entities[0], entities[-1]
    if first == last:
        return None
    start = main.find(first) + len(first)
    end = main.rfind(last)
    if end <= start:
        return None
    verb_phrase = main[start:end].strip(" ,")
    if not verb_phrase:
        return None
    return last, verb_phrase


@register_splitter("temporal_qualifier")
def _split_temporal(clause: str) -> Optional[Tuple[str, str, str]]:
    if len(clause.split()) < 5:
        return None
    m = _TEMPORAL_TRAILING_RE.match(clause)
    if not m:
        return None
    main, qualifier = m.group("main"), m.group("qualifier")
    anchor = _anchor_entity_and_verb(main)
    if anchor is None:
        return None
    entity, verb_phrase = anchor
    secondary = f"{entity} {verb_phrase} {qualifier}".strip()
    return main, secondary, "temporal"


@register_splitter("locative_qualifier")
def _split_locative(clause: str) -> Optional[Tuple[str, str, str]]:
    if len(clause.split()) < 5:
        return None
    m = _LOCATIVE_TRAILING_RE.match(clause)
    if not m:
        return None
    main, qualifier = m.group("main"), m.group("qualifier")
    anchor = _anchor_entity_and_verb(main)
    if anchor is None:
        return None
    entity, verb_phrase = anchor
    secondary = f"{entity} {verb_phrase} {qualifier}".strip()
    return main, secondary, "locative"


def _apply_qualifier_splitters(clause: str) -> List[Tuple[str, Optional[str], str]]:
    """Returns [(text, qualifier_type, method), ...] for one clause: either
    [main, secondary] if a registered splitter matched, or [clause] as-is."""
    for name, splitter in _SPLITTER_REGISTRY:
        result = splitter(clause)
        if result is not None:
            main, secondary, qualifier_type = result
            return [(main, None, "single_claim"), (secondary, qualifier_type, name)]
    return [(clause, None, "single_claim")]


# ─────────────────────────────────────────────
# YES/NO REFORMULATION
# A bare "Yes"/"No" has no propositional content of its own — an NLI model
# can't check it against evidence. Reformulate it into the declarative
# statement the question is actually asking about, so there's something to
# verify. Requires a "Were/Was/Is/Are/Did/Does/Do/Have/Has/Can/Could/Would/
# Will <subject> <predicate>?" question shape (the standard yes/no form);
# outside that shape, the caller falls back to using the answer as-is.
# ─────────────────────────────────────────────

_YESNO_RE = re.compile(r"^\s*(yes|no)\s*[.!]?\s*$", re.IGNORECASE)
_LEADING_AUX_RE = re.compile(
    r"^(Were|Was|Is|Are|Did|Does|Do|Have|Has|Can|Could|Would|Will)\s+(.+?)\s*\?*\s*$",
    re.IGNORECASE,
)
_CONNECTOR_GAP_RE = re.compile(r"^\s*(?:,|and)?\s*$", re.IGNORECASE)


def _find_subject_boundary(remainder: str) -> int:
    """Returns how many leading characters of `remainder` form its subject,
    using entity-span positions rather than cue words: "Scott Derrickson
    and Ed Wood of the same nationality" extends across the "and" to
    include both conjoined names, while "Paris the capital of France" stops
    right after "Paris" since "the" isn't a connector. Returns 0 when the
    remainder doesn't start with a recognizable entity at all."""
    entities = list(_ENTITY_RE.finditer(remainder))
    if not entities or entities[0].start() != 0:
        return 0
    end = entities[0].end()
    for ent in entities[1:]:
        gap = remainder[end:ent.start()]
        if _CONNECTOR_GAP_RE.match(gap):
            end = ent.end()
        else:
            break
    return end


def _reformulate_yes_no(question: str, answer: str) -> Optional[str]:
    if not question or not _YESNO_RE.match(answer):
        return None
    aux_match = _LEADING_AUX_RE.match(question.strip())
    if not aux_match:
        return None
    aux, remainder = aux_match.group(1), aux_match.group(2)

    article_match = re.match(r"^(the|a|an)\s+", remainder, re.IGNORECASE)
    prefix = article_match.group(0) if article_match else ""
    body = remainder[len(prefix):]

    split_at = _find_subject_boundary(body)
    if split_at == 0 or split_at >= len(body):
        return None

    subject, predicate = (prefix + body[:split_at]).strip(), body[split_at:].strip()
    is_yes = answer.strip().lower().startswith("y")
    negation = "" if is_yes else "not "
    return f"{subject} {aux.lower()} {negation}{predicate}".strip()


# ─────────────────────────────────────────────
# LLM DECOMPOSITION (minority fallback)
# ─────────────────────────────────────────────

LLM_FALLBACK_WORD_THRESHOLD = 22


def _llm_decompose(answer: str) -> List[str]:
    """Asks the LLM to list independent atomic claims, one per line. Used
    only when deterministic splitting leaves a single long clause intact —
    the same minority-fallback pattern already validated in this project's
    Stage 2 label-generation cascade."""
    prompt = (
        f"Split the following answer into independent factual claims, one per line. "
        f"Each line must be a short, self-contained statement. Do not add information "
        f"that isn't in the original answer. Do not number the lines.\n\n"
        f"Answer: {answer}\n\nClaims:"
    )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 120},
        )
        lines = [
            re.sub(r"^[\-\*\d.\)]+\s*", "", line).strip()
            for line in response["message"]["content"].splitlines()
        ]
        return [line for line in lines if len(line.split()) >= MIN_CLAUSE_WORDS]
    except Exception:
        return []


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def extract_claims(answer: str, question: str = "") -> List[Claim]:
    """Decomposes a generated answer into an ordered list of Claims. The
    first claim is treated as the primary answer to the question; every
    other claim (additional conjunctive clauses, or qualifiers peeled off by
    a registered splitter) is "secondary". This ordering assumption matches
    typical HotpotQA answer phrasing (the main assertion first, elaboration
    after) and is a documented simplification for compound comparison-style
    answers where two clauses are equally central.

    Always returns at least one claim — if every decomposition step is a
    no-op, the whole answer becomes a single primary claim.
    """
    answer = answer.strip()
    if not answer:
        return [Claim(text=answer, kind="primary", method="empty")]

    reformulated = _reformulate_yes_no(question, answer)
    if reformulated is not None:
        return [Claim(text=reformulated, kind="primary", method="yes_no_reformulation")]

    clauses = _split_conjunctions(answer)

    claims: List[Claim] = []
    for clause_idx, clause in enumerate(clauses):
        for text, qualifier_type, method in _apply_qualifier_splitters(clause):
            kind = "primary" if (clause_idx == 0 and not claims) else "secondary"
            claims.append(Claim(text=text, kind=kind, qualifier_type=qualifier_type, method=method))

    if len(claims) == 1 and len(claims[0].text.split()) > LLM_FALLBACK_WORD_THRESHOLD:
        decomposed = _llm_decompose(claims[0].text)
        if len(decomposed) > 1:
            claims = [
                Claim(text=text, kind=("primary" if i == 0 else "secondary"), method="llm_decomposition")
                for i, text in enumerate(decomposed)
            ]

    return claims if claims else [Claim(text=answer, kind="primary", method="fallback")]

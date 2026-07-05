"""
Stage 2 V3: Evidence-Grounded Self-Verification (fine-tuned DeBERTa-v3-large)
================================================================================
Project: HARA — Hallucination-Aware Retrieval Agent

Single-file Stage 2 verifier. Pipeline for one (question, generated_answer,
retrieved_passages) triple:

    1. CLAIM EXTRACTION    -> decompose the answer into atomic claims
    2. EVIDENCE MATCHING   -> rank the already-retrieved passages per claim
                              (no new retrieval — narrow RAG over Stage 1's
                              own passage set, per claim)
    3. NLI SCORING         -> MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli,
                              fine-tuned on this project's own HotpotQA-
                              derived (evidence, claim, label) triples (see
                              Stage_2_Verifier_Train.py), falls back to the
                              zero-shot checkpoint if no fine-tuned checkpoint
                              exists yet on disk
    4. AGGREGATION         -> per-claim diagnosis + question-intent/coverage/
                              type checks combined into one VerificationReport

Public entry point (unchanged call-site shape from the prior V2 design, so
Stage 3/4/5/6 need no changes):

    from Stage_2_Verifier import verify, load_verifier

    verifier_model, verifier_tokenizer = load_verifier(VERIFIER_PATH)
    report = verify(question, generated_answer, reranked_passages, verifier_model)
    report["overall_status"]        # SUPPORTED | PARTIAL | UNSUPPORTED
    report["recommended_action"]    # KEEP | EXPAND | REWRITE | DECOMPOSE | COMPARE | ABSTAIN
    report["failure_reason"]        # explainable reason, or None

Adaptive feedback (point 5 of the HARA architecture — "the generator learns
from the validator"): every non-SUPPORTED verdict is optionally appended to
FEEDBACK_LOG_PATH (opt-in via HARA_ADAPTIVE_FEEDBACK=1, off by default so
existing Stage 3/4/5/6 behavior — including batch evaluation runs — is
unaffected unless explicitly enabled). Stage_2_AdaptiveFeedback.py reads
that log and distills it into a corrective system prompt applied to the
local Ollama model tag, so Stage 1's generator adapts on the next run
without any change to Stage_1_RAG_Pipeline.py or any other stage's file.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import ollama

# Stage_1_RAG_Pipeline (imported here, and by every Stage 3/4/5/6 caller
# before this module) pulls in faiss, which initializes its own native
# OpenMP thread pool. If torch's multi-threaded CPU backend initializes
# afterward in the same process, loading or running the NLI model segfaults
# deterministically — reproduced and confirmed via bisection. faiss must
# therefore be imported BEFORE torch in this module, and torch forced to
# single-threaded CPU execution immediately after; the NLI model runs on
# MPS/CUDA when available (see DEVICE below), where this has no meaningful
# performance cost — only CPU-side tensor ops are affected.
from Stage_1_RAG_Pipeline import OLLAMA_MODEL, RERANKER_MODEL

import torch
torch.set_num_threads(1)

import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("stage2_v3")

__all__ = [
    "verify", "load_verifier", "load_nli_verifier", "NLIVerifier", "NLIResult",
    "verify_legacy", "build_verify_context", "VERIFIER_PATH",
    "normalize_verification", "legacy_scores",
]


# ════════════════════════════════════════════════════════════════════════
# SECTION 1 — CLAIM EXTRACTION
# Splits a generated answer into independent, checkable atomic claims:
# deterministic conjunction/qualifier splitting first, LLM decomposition
# only as a fallback for long unstructured answers.
# ════════════════════════════════════════════════════════════════════════

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

# A naive split on ". " breaks mid-sentence at abbreviations ("Dr. Robotnik",
# "K. A. Applegate") — merge any split that landed right after a known
# abbreviation or a single-letter initial rather than a real sentence
# boundary. Shared by claim splitting and evidence sentence-refinement.
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "jr", "sr", "st", "vs", "prof", "rev", "gen",
    "sen", "rep", "gov", "lt", "col", "capt", "sgt", "mt", "no", "vol", "etc",
}


def _ends_with_abbreviation(sentence: str) -> bool:
    words = sentence.split()
    if not words:
        return False
    last = words[-1].rstrip(".")
    if last.lower() in _ABBREVIATIONS:
        return True
    return len(last) == 1 and last.isupper()  # a lone initial, e.g. "K." in "K. A. Applegate"


def _split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for part in _SENTENCE_SPLIT_RE.split(text):
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return sentences


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
    for sentence in _split_sentences(text):
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


# Demonyms/nationality adjectives ("South Korean", "American") match the
# capitalized-span entity regex but aren't a genuine second comparison
# subject — confirmed to cause a real bug: a claim like "YG Entertainment
# formed the South Korean boy group WINNER" was treated as naming 3 distinct
# entities, triggering a search for a passage "about" South Korean-ness,
# which pulled in an unrelated South Korean group's passage and made the
# whole premise read as contradicting the real claim.
_DEMONYMS = {
    "south korean", "north korean", "korean", "american", "british", "english",
    "scottish", "irish", "french", "german", "italian", "spanish", "russian",
    "chinese", "japanese", "indian", "australian", "canadian", "mexican",
    "brazilian", "dutch", "swedish", "norwegian", "danish", "egyptian",
    "turkish", "greek", "polish", "portuguese", "vietnamese", "thai",
}


def is_comparable_entity(entity: str) -> bool:
    """True for entities worth fetching their own dedicated evidence
    passage for (a second comparison subject) — excludes bare demonyms/
    nationality adjectives, which are not a genuine second subject."""
    return entity.lower() not in _DEMONYMS


@dataclass
class Claim:
    text: str
    kind: str                          # "primary" | "secondary"
    qualifier_type: Optional[str] = None   # "temporal" | "locative" | None
    method: str = "single_claim"       # how this claim was produced (auditability)


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


# Trailing-qualifier splitter registry: each splitter peels a date/place
# phrase off the end of a clause and re-anchors it to the last entity
# mentioned, producing a secondary claim. Registering a new splitter = one
# more function here; extract_claims() itself never needs to change.
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


# Yes/No reformulation: a bare "Yes"/"No" has no propositional content of
# its own — an NLI model can't check it against evidence. Reformulate it
# into the declarative statement the question is actually asking about.
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


LLM_FALLBACK_WORD_THRESHOLD = 22


def _llm_decompose(answer: str) -> List[str]:
    """Asks the LLM to list independent atomic claims, one per line — used
    only when deterministic splitting leaves a single long clause intact
    (minority fallback)."""
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


def extract_claims(answer: str, question: str = "") -> List[Claim]:
    """Decomposes a generated answer into an ordered list of Claims. The
    first claim is the primary answer to the question; every other claim
    (additional conjunctive clauses, or qualifiers peeled off by a
    registered splitter) is "secondary". Always returns at least one claim."""
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


# ════════════════════════════════════════════════════════════════════════
# SECTION 2 — EVIDENCE MATCHING
# For each atomic claim, ranks the passages Stage 1/3/4 already retrieved
# and reranked — never issues new retrieval. Narrow RAG: scoring stays
# within this project's own already-retrieved vector-search results.
# ════════════════════════════════════════════════════════════════════════

MIN_RELEVANCE_SCORE = 0.0   # ms-marco cross-encoder logits: >0 ~ topically relevant
DEFAULT_TOP_K = 2
MIN_SENTENCES_FOR_REFINEMENT = 3  # passages this short are left whole

_matcher: Optional[CrossEncoder] = None


@dataclass
class EvidenceMatch:
    title: str
    text: str
    relevance_score: float


def _get_matcher() -> CrossEncoder:
    global _matcher
    if _matcher is None:
        _matcher = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _matcher


def match_evidence(
    claim_text: str,
    passages: List[Dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    matcher: Optional[CrossEncoder] = None,
) -> List[EvidenceMatch]:
    """Ranks `passages` for a single claim. Prefer match_evidence_batch()
    when scoring multiple claims for the same question."""
    if not passages or not claim_text.strip():
        return []
    model = matcher or _get_matcher()
    pairs = [(claim_text, p["text"][:512]) for p in passages]
    scores = model.predict(pairs)
    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [
        EvidenceMatch(title=p["title"], text=p["text"], relevance_score=float(s))
        for p, s in ranked[:top_k]
    ]


def match_evidence_batch(
    claim_texts: List[str],
    passages: List[Dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    matcher: Optional[CrossEncoder] = None,
) -> List[List[EvidenceMatch]]:
    """Ranks `passages` against every claim in one batched cross-encoder call."""
    if not passages or not claim_texts:
        return [[] for _ in claim_texts]
    model = matcher or _get_matcher()
    pairs = [(c, p["text"][:512]) for c in claim_texts for p in passages]
    scores = model.predict(pairs)
    n = len(passages)
    results: List[List[EvidenceMatch]] = []
    for i in range(len(claim_texts)):
        chunk = scores[i * n:(i + 1) * n]
        ranked = sorted(zip(passages, chunk), key=lambda x: x[1], reverse=True)
        results.append([
            EvidenceMatch(title=p["title"], text=p["text"], relevance_score=float(s))
            for p, s in ranked[:top_k]
        ])
    return results


def best_sentence(claim_text: str, passage_text: str, matcher: Optional[CrossEncoder] = None) -> str:
    """Refines a whole passage down to its single best-matching sentence for
    a given claim. A terse claim checked against a full multi-sentence
    passage can flip an NLI model from ENTAILED to CONTRADICTED once an
    unrelated later sentence is in the same premise — confirmed empirically.
    Short passages are returned unchanged."""
    sentences = [s.strip() for s in _split_sentences(passage_text) if s.strip()]
    if len(sentences) < MIN_SENTENCES_FOR_REFINEMENT:
        return passage_text
    model = matcher or _get_matcher()
    pairs = [(claim_text, s) for s in sentences]
    scores = model.predict(pairs)
    return max(zip(sentences, scores), key=lambda x: x[1])[0]


def build_premise(
    claim_text: str,
    entities: List[str],
    passages: List[Dict[str, Any]],
    top_k_evidence: int = DEFAULT_TOP_K,
    matcher: Optional[CrossEncoder] = None,
) -> Tuple[str, List[EvidenceMatch]]:
    """Constructs the NLI premise for one claim, returning (premise_text,
    evidence_matches). A claim naming >=2 distinct entities (a comparison)
    gets the best passage PER entity concatenated, since a single premise
    usually only covers one side and a single-premise NLI model can't
    perform the missing cross-passage comparison on its own."""
    model = matcher or _get_matcher()
    ranked = match_evidence(claim_text, passages, top_k=max(top_k_evidence, len(entities) or 1), matcher=model)
    if not ranked:
        return "", []

    if len(entities) < 2:
        premise = best_sentence(claim_text, ranked[0].text, matcher=model)
        return premise, ranked

    segments: List[str] = []
    seen_titles = set()
    for entity in entities:
        entity_matches = match_evidence(entity, passages, top_k=1, matcher=model)
        if not entity_matches or entity_matches[0].title in seen_titles:
            continue
        seen_titles.add(entity_matches[0].title)
        segments.append(best_sentence(claim_text, entity_matches[0].text, matcher=model))

    if not segments:
        premise = best_sentence(claim_text, ranked[0].text, matcher=model)
        return premise, ranked

    return " ".join(segments), ranked


# ════════════════════════════════════════════════════════════════════════
# SECTION 3 — NLI VERIFIER (fine-tuned DeBERTa-v3-large-mnli-fever-anli)
# Loads a fine-tuned checkpoint if Stage_2_Verifier_Train.py has produced
# one at FINE_TUNED_MODEL_DIR; otherwise falls back to the public zero-shot
# checkpoint so the pipeline still works before training completes.
# ════════════════════════════════════════════════════════════════════════

ZERO_SHOT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
FINE_TUNED_MODEL_DIR = "models/hara_deberta_v3_large_verifier"
MAX_LENGTH = 256
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

_NLI_LABEL_MAP = {
    "entailment": "ENTAILED",
    "entailed": "ENTAILED",
    "neutral": "NEUTRAL",
    "contradiction": "CONTRADICTED",
    "contradicted": "CONTRADICTED",
}


@dataclass
class NLIResult:
    verdict: str            # ENTAILED | NEUTRAL | CONTRADICTED
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float

    @property
    def confidence(self) -> float:
        return max(self.entailment_prob, self.neutral_prob, self.contradiction_prob)


def resolve_model_path() -> str:
    """Fine-tuned checkpoint if Stage_2_Verifier_Train.py has produced one,
    else the zero-shot base checkpoint (downloaded from the Hub)."""
    if os.path.isdir(FINE_TUNED_MODEL_DIR) and os.listdir(FINE_TUNED_MODEL_DIR):
        return FINE_TUNED_MODEL_DIR
    return ZERO_SHOT_NLI_MODEL


class NLIVerifier:
    """Loads the (fine-tuned if available, else zero-shot) NLI checkpoint
    once and scores (evidence, claim) pairs."""

    def __init__(self, model_name: str = None, device: str = DEVICE):
        self.model_name = model_name or resolve_model_path()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(device)
        self.model.eval()
        self._label_order = self._resolve_label_order()

    def _resolve_label_order(self) -> List[str]:
        id2label = self.model.config.id2label
        order = []
        for i in range(len(id2label)):
            raw = id2label[i].lower()
            order.append(_NLI_LABEL_MAP.get(raw, raw.upper()))
        return order

    def score_batch(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """pairs: list of (premise/evidence, hypothesis/claim). A pair with
        an empty premise short-circuits to a zero-confidence NEUTRAL without
        spending a model call on it."""
        results: List[Optional[NLIResult]] = [None] * len(pairs)
        scoreable_idx = [i for i, (premise, _) in enumerate(pairs) if premise.strip()]

        if scoreable_idx:
            premises = [pairs[i][0] for i in scoreable_idx]
            hypotheses = [pairs[i][1] for i in scoreable_idx]
            inputs = self.tokenizer(
                premises, hypotheses, return_tensors="pt",
                padding=True, truncation=True, max_length=MAX_LENGTH,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu()
            for pos, i in enumerate(scoreable_idx):
                scores = {label: float(probs[pos][j]) for j, label in enumerate(self._label_order)}
                verdict = max(scores, key=scores.get)
                results[i] = NLIResult(
                    verdict=verdict,
                    entailment_prob=scores.get("ENTAILED", 0.0),
                    neutral_prob=scores.get("NEUTRAL", 0.0),
                    contradiction_prob=scores.get("CONTRADICTED", 0.0),
                )

        for i in range(len(pairs)):
            if results[i] is None:
                results[i] = NLIResult(
                    verdict="NEUTRAL", entailment_prob=0.0,
                    neutral_prob=1.0, contradiction_prob=0.0,
                )
        return results

    def score(self, premise: str, hypothesis: str) -> NLIResult:
        return self.score_batch([(premise, hypothesis)])[0]


def load_nli_verifier(model_name: str = None) -> NLIVerifier:
    return NLIVerifier(model_name=model_name)


# ════════════════════════════════════════════════════════════════════════
# SECTION 4 — QUESTION INTENT
# Infers what a question is actually asking for, purely via regex/
# heuristics — used to catch a well-entailed claim that's still the wrong
# type or doesn't address the question.
# ════════════════════════════════════════════════════════════════════════

_COMPARATIVE_WORDS = (
    "older", "younger", "first", "last", "earlier", "later", "before", "after",
    "more", "less", "higher", "lower", "longer", "shorter", "bigger", "smaller",
    "taller", "same", "both",
)
_INTENT_LEADING_AUX_RE = re.compile(
    r"^(were|was|is|are|did|does|do|have|has|can|could|would|will)\b", re.IGNORECASE
)
_OR_COMPARISON_RE = re.compile(r"\b[A-Z][\w.']*(?:\s+[A-Z][\w.']*)*\s+or\s+[A-Z][\w.']*")

_WH_TYPE_RULES = (
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
    (r"\bwhen\b|\bwhat\s+date\b|\bwhat\s+time\s*frame\b|\bwhat\s+time\s+period\b", "DATE"),
)

_LEADING_WH_RE = re.compile(
    r"^\s*(what|which|who|whom|whose|where|when|how)\b", re.IGNORECASE
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
    if _INTENT_LEADING_AUX_RE.match(q) and any(w in q.lower() for w in _COMPARATIVE_WORDS):
        return True
    return False


def _detect_type(question: str, comparison: bool) -> str:
    if comparison:
        return "COMPARISON"
    q_lower = question.lower()
    stripped = question.strip()

    matches = []
    for pattern, etype in _WH_TYPE_RULES:
        m = re.search(pattern, q_lower)
        if m:
            matches.append((m.start(), etype))

    if matches:
        if _LEADING_WH_RE.match(stripped):
            matches.sort(key=lambda item: item[0])
        else:
            # Opening content is very likely a relative clause embedding an
            # unrelated wh-word; HotpotQA phrasings overwhelmingly put the
            # real question word last, right before the "?".
            matches.sort(key=lambda item: -item[0])
        return matches[0][1]
    if _INTENT_LEADING_AUX_RE.match(question.strip()):
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


# ════════════════════════════════════════════════════════════════════════
# SECTION 5 — ANSWER TYPE VALIDATION
# Scores whether a candidate answer's surface form matches the type the
# question expects — independent of whether it's evidence-supported.
# ════════════════════════════════════════════════════════════════════════

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
            # ("Who formed WINNER?" -> "YG Entertainment") — a near-match.
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


# ════════════════════════════════════════════════════════════════════════
# SECTION 6 — QUESTION COVERAGE & COMPLETENESS
# score_question_coverage(): does the answer's CONTENT address what was
# asked, regardless of whether it's true?
# score_completeness(): does the answer contain ALL required parts for
# this question's shape (comparison needs a resolved side, LIST needs
# multiple items)?
# ════════════════════════════════════════════════════════════════════════

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
        # An unresolved hedge ("both", "unclear") covers the topic without
        # answering the question.
        if _HEDGE_BOTH_RE.search(answer):
            return min(base, 0.3)
        return max(base, 0.6) if answer else 0.0

    return base


def score_completeness(intent: QuestionIntent, answer: str) -> float:
    """0.0-1.0: does the answer contain enough parts for this question's
    shape? Single-fact questions are complete with just one correct fact."""
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


# ════════════════════════════════════════════════════════════════════════
# SECTION 7 — AGGREGATION
# Combines FIVE independent dimensions into one VerificationReport:
#   1. Evidence Support     (40%) — per-claim NLI entailment
#   2. Question Coverage    (25%) — does the content address what was asked
#   3. Answer Completeness  (20%) — does it contain all required parts
#   4. Answer Type Match    (10%) — does the surface form match the type
#   5. Contradiction         (5%) — penalty if any claim is contradicted
# Question-aware checks (2-5) act as corrective overrides on top of the
# per-claim NLI diagnosis (1) — they can escalate severity but never redeem
# a genuine contradiction.
# ════════════════════════════════════════════════════════════════════════

AMBIGUITY_RELEVANCE_DELTA = 1.0

DEFAULT_WEIGHTS = {
    "evidence_support": 0.40,
    "question_coverage": 0.25,
    "answer_completeness": 0.20,
    "answer_type": 0.10,
    "contradiction": 0.05,
}

ANSWER_TYPE_ESCALATION_THRESHOLD = 0.30
COVERAGE_ESCALATION_THRESHOLD = 0.30
COMPLETENESS_PARTIAL_THRESHOLD = 0.70

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

    evidence_support_score: float = 0.0
    question_coverage_score: float = 0.0
    answer_completeness_score: float = 0.0
    answer_type_score: float = 0.0
    contradiction_score: float = 1.0
    question_intent: Optional[Dict[str, Any]] = None
    weights_used: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    entity, number, or date — rather than one generic reason."""
    years = extract_years(claim.text)
    nums = extract_numbers(claim.text)
    entities = extract_entities(claim.text)

    has_relevant_evidence = best_match is not None and best_match.relevance_score >= MIN_RELEVANCE_SCORE
    if not has_relevant_evidence:
        return "NO_RELEVANT_PASSAGE"

    if years and not any(y in all_evidence_text for y in years):
        return "MISSING_REQUIRED_DATE"
    if nums and not any(n in all_evidence_text for n in nums):
        return "MISSING_REQUIRED_NUMBER"
    if entities and not any(e.lower() in all_evidence_text.lower() for e in entities):
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
    explainable failure reason (Evidence Support dimension only)."""
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


def _base_status_and_reason(claim_verifications: List[ClaimVerification]):
    """Evidence Support dimension's own severity-based verdict. Any claim
    CONTRADICTED is treated as severe regardless of whether it's the
    primary or a secondary/elaboration claim (contradiction -> ABSTAIN is
    stronger than a mere downgrade)."""
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

    # Corrective overrides: a well-entailed claim that's the wrong type or
    # doesn't address the question is still a bad answer — can only
    # escalate severity, never redeem an already-detected contradiction.
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


# ════════════════════════════════════════════════════════════════════════
# SECTION 8 — ADAPTIVE FEEDBACK LOGGING (opt-in)
# Every non-SUPPORTED verdict is optionally appended here. Stage 1/3/4/5/6
# are never imported by or aware of this — it's a side effect of verify()
# alone. Off by default (HARA_ADAPTIVE_FEEDBACK unset) so existing
# behavior, including batch evaluation runs, is unaffected unless a user
# explicitly opts in. See Stage_2_AdaptiveFeedback.py for the consumer.
# ════════════════════════════════════════════════════════════════════════

FEEDBACK_LOG_PATH = "verifier_feedback.jsonl"


def _log_feedback(question: str, answer: str, report: Dict[str, Any]) -> None:
    if os.environ.get("HARA_ADAPTIVE_FEEDBACK") != "1":
        return
    if report.get("overall_status") == "SUPPORTED":
        return
    try:
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "question": question,
                "answer": answer,
                "overall_status": report.get("overall_status"),
                "failure_reason": report.get("failure_reason"),
                "recommended_action": report.get("recommended_action"),
                "question_intent": report.get("question_intent"),
            }) + "\n")
    except OSError:
        pass  # feedback logging must never break the verify() call it rides on


# ════════════════════════════════════════════════════════════════════════
# SECTION 9 — PUBLIC ORCHESTRATOR
# Unchanged call-site shape from the prior V2 design — Stage 3/4/5/6 import
# these names and this return shape without modification.
# ════════════════════════════════════════════════════════════════════════

VERIFIER_PATH = FINE_TUNED_MODEL_DIR  # os.path.exists(VERIFIER_PATH) gates Stage 3/4/5/6 loading


def verify(
    question: str,
    generated_answer: str,
    passages: List[Dict[str, Any]],
    nli_verifier: NLIVerifier,
    top_k_evidence: int = DEFAULT_TOP_K,
    weights: Dict[str, float] = None,
) -> Dict[str, Any]:
    """Evidence-grounded, question-aware self-verification for one
    (question, answer) pair. `passages` should be the passages actually
    used for generation (Stage 1/3/4's reranked top-k). Returns a
    JSON-serializable dict — see VerificationReport for the exact field list.
    """
    claims = extract_claims(generated_answer, question=question)
    claim_texts = [c.text for c in claims]

    premises, evidence_matches = [], []
    for claim_text in claim_texts:
        entities = [e for e in extract_entities(claim_text) if is_comparable_entity(e)]
        premise, matches = build_premise(claim_text, entities, passages, top_k_evidence=top_k_evidence)
        premises.append(premise)
        evidence_matches.append(matches)

    nli_pairs = list(zip(premises, claim_texts))
    nli_results = nli_verifier.score_batch(nli_pairs)

    all_evidence_text = "\n".join(p["text"] for p in passages)
    claim_verifications = build_claim_verifications(
        claims, evidence_matches, nli_results, all_evidence_text, question=question,
    )

    report = aggregate(claim_verifications, question=question, answer=generated_answer, weights=weights)
    report_dict = report.to_dict()
    _log_feedback(question, generated_answer, report_dict)
    return report_dict


def load_verifier(model_path: str = None):
    """Old call sites do `verifier_model, verifier_tokenizer = load_verifier(
    VERIFIER_PATH)`, then gate verification on `verifier_tokenizer is not
    None` (Stage 3/4/5 all do this). Returns the same NLIVerifier instance
    in both slots — verify_legacy() below only ever uses the first — so
    that gate stays truthy and those call sites need no other change.
    `model_path`, if given, overrides the resolved fine-tuned/zero-shot path."""
    nli_verifier = load_nli_verifier(model_name=model_path if model_path and os.path.isdir(model_path) else None)
    return nli_verifier, nli_verifier


_LEGACY_PASSAGE_SEP = "\n---PASSAGE---\n"


def build_verify_context(passages: List[Dict[str, Any]], answer: str, top_n: int = 5) -> str:
    """Same reordering behavior as the old fine-tuned-classifier verifier
    (answer-containing passages first), returned as a single string.
    Passages are joined with a recoverable separator: verify_legacy() splits
    on it to restore individual passage boundaries — concatenating unrelated
    passages into one blob measurably hurts the NLI model."""
    a_lower = answer.lower().strip()
    containing = [p for p in passages if a_lower in p["text"].lower()]
    others = [p for p in passages if a_lower not in p["text"].lower()]
    ordered = (containing + others)[:top_n]
    return _LEGACY_PASSAGE_SEP.join(p["text"] for p in ordered)


def legacy_scores(label: str, support_score: float) -> Dict[str, float]:
    """Synthesizes a SUPPORTED/PARTIAL/UNSUPPORTED probability triple from
    support_score for old call sites that read verification["scores"]."""
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
    """Drop-in replacement for the old positional verify(context, answer,
    model, tokenizer, question=...) signature. `model` must be an
    NLIVerifier (from load_verifier() above); `tokenizer` is accepted only
    for positional compatibility and ignored."""
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
    """Normalizes a verification dict to the new report shape, whether it
    came directly from verify() (already this shape) or from verify_legacy()
    — which may carry a "verification_report" nested inside it, and whose
    top-level label/confidence can reflect a demotion applied *after*
    verify_legacy() returned. Takes label/confidence from the OUTER
    (possibly-demoted) fields, not the stale nested report."""
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

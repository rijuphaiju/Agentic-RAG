"""
Stage 2 V2: Evidence Matcher
=============================
Project: HARA — Hallucination-Aware Retrieval Agent

For each atomic claim, ranks the passages Stage 1/3/4 already retrieved and
reranked — it never issues a new retrieval call. A claim is short (a single
proposition), so scoring it against the already-small reranked passage set
(Stage 1's TOP_K, typically 5) with a cross-encoder is cheap; this module
exists specifically so the NLI verifier gets the single best-matching
passage as its premise per claim, rather than every claim being scored
against the same generic top-1 passage the whole answer used.

Reuses the same cross-encoder checkpoint Stage 1's reranker already uses
(RERANKER_MODEL), but keeps its own lazily-loaded instance so this module
has no import-time dependency on Stage 1's internal state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder

from Stage_1_RAG_Pipeline import RERANKER_MODEL

MIN_RELEVANCE_SCORE = 0.0   # ms-marco cross-encoder logits: >0 ~ topically relevant
DEFAULT_TOP_K = 2

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
MIN_SENTENCES_FOR_REFINEMENT = 3  # passages this short are left whole

# A naive split on ". " breaks mid-sentence at abbreviations ("Dr. Robotnik",
# "K. A. Applegate") — confirmed to produce a genuinely garbled, subject-less
# fragment ("Robotnik from "Sonic the Hedgehog", and Pete.") that an NLI
# model then correctly refuses to entail, since it isn't a real sentence.
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
    """Sentence-splits `text`, re-merging any split that landed right after
    a known abbreviation or a single-letter initial rather than a real
    sentence boundary."""
    sentences: List[str] = []
    for part in _SENTENCE_SPLIT_RE.split(text):
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return sentences

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
    when scoring multiple claims for the same question — it makes one
    batched cross-encoder call instead of one per claim."""
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
    """Ranks `passages` against every claim in one batched cross-encoder
    call (len(claim_texts) x len(passages) pairs — cheap since `passages`
    is already the small reranked set, not a full corpus).
    """
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
    a given claim. A terse claim ("Animorphs") checked against a full
    multi-sentence passage can flip an NLI model from ENTAILED to
    CONTRADICTED once an unrelated later sentence is in the same premise —
    confirmed empirically (a passage opening "Animorphs is a science
    fantasy series..." followed by unrelated thematic sentences scored
    CONTRADICTED as a whole passage, ENTAILED as just its first sentence).
    Short passages are returned unchanged — there's nothing to gain by
    splitting a passage that's already close to one sentence.
    """
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
    """
    Constructs the NLI premise for one claim, returning (premise_text,
    evidence_matches) — evidence_matches is the ranked EvidenceMatch list
    used for best_evidence/evidence_score reporting.

    A claim naming >=2 distinct entities (e.g. a comparison — "Giuseppe
    Verdi and Ambroise Thomas are both Opera composers") needs facts about
    BOTH entities in the same premise: the single best-matching passage
    usually only discusses one of them, and a single-premise NLI model
    can't perform the missing cross-passage comparison — confirmed
    empirically (NEUTRAL when given only the Verdi passage, ENTAILED once
    the Ambroise Thomas passage was concatenated in). For a single-entity
    (or entity-free) claim, this reduces to the original "best passage,
    refined to its best sentence" behavior.
    """
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

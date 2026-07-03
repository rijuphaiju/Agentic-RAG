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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

from Stage_1_RAG_Pipeline import RERANKER_MODEL

MIN_RELEVANCE_SCORE = 0.0   # ms-marco cross-encoder logits: >0 ~ topically relevant
DEFAULT_TOP_K = 2

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

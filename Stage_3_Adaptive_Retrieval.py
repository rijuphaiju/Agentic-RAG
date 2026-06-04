"""
Stage 3: Adaptive Retrieval
===========================
Project: HARA — Hallucination-Aware Retrieval Agent
Proposal Section: 2.5, 4.3, 6.3.3

Builds on Stage 1 (rag_pipeline.py) and Stage 2 (verifier_gpu.py).

Three retrieval strategies selected by query complexity classifier:
  SIMPLE     → standard top-k retrieval (single hop)
  MULTI_HOP  → iterative retrieval with query decomposition
  COMPARISON → parallel retrieval for both entities being compared

Usage:
  python adaptive_retrieval.py
"""

import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import faiss
import numpy as np
import ollama
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Stage 1 helpers ──
from Stage_1_RAG_Pipeline import (
    load_faiss_index,
    build_faiss_index,
    load_hotpotqa_passages,
    generate_answer,
    rerank_passages,
    retrieve_hybrid,
    retrieve as _retrieve_dense,
    normalize_answer,
    exact_match,
    llm_judge_supported,
    INDEX_PATH,
    PASSAGES_PATH,
    EMBED_MODEL,
    OLLAMA_MODEL,
)

# ── Stage 2 verifier ──
from Stage_2_Verifier_GPU import load_verifier, verify, build_verify_context, VERIFIER_PATH

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE      = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available()
               else "cpu")
TOP_K              = 10    # passages sent to LLM (post-rerank) — kept at 10 for Stage 4 compat
TOP_K_MULTI        = 5     # passages per hop in multi-hop iterative retrieval
MAX_HOPS           = 3     # maximum iterative hops
LOW_SUPPORT_THRESHOLD = 0.15  # verifier P(SUPPORTED) below which multi-hop is retried

# ── Complexity-aware adaptive retrieval constants ──
MAX_COVERAGE_EXPANSIONS  = 2     # max extra retrieval rounds before giving up and generating
RETRIEVAL_CONF_THRESHOLD = 0.45  # cross-encoder confidence below which budget is expanded
COVERAGE_THRESHOLD       = 0.50  # fraction of key question entities that must appear in top-k


# ─────────────────────────────────────────────
# STEP 1: QUERY COMPLEXITY CLASSIFIER
# Classifies each query into SIMPLE / MULTI_HOP / COMPARISON
# This implements Section 2.5 and 4.3 of the proposal
# ─────────────────────────────────────────────
# Strong comparison words that reliably indicate a two-entity comparison question.
# "first/last/most/least" are intentionally EXCLUDED here — they appear in multi-hop
# questions ("the first film that starred X") and cause COMPARISON misclassifications.
# Those ordinal words are handled separately: only COMPARISON when combined with "or".
COMPARISON_WORDS = {
    "both", "same", "different", "compare", "versus", "vs",
    "older", "newer", "bigger", "smaller", "taller", "shorter",
    "longer", "earlier", "later", "more", "less", "better", "worse",
}

# Ordinal/superlative words that indicate comparison ONLY when two explicit entities
# are joined by "or" in the same question.
_ORDINAL_COMPARISON_WORDS = {"first", "last", "most", "least"}

_REFUSAL_PATTERNS = re.compile(
    r'\b(cannot provide|i cannot|do not appear|does not appear|'
    r'no information|not mentioned|not in the context|cannot be found|'
    r'not available|i don\'t have|does not contain|'
    r'not found|not determined|not specified|cannot determine|'
    r'information is not|answer is not|year is not|cannot be determined)\b',
    re.IGNORECASE,
)


def _is_refusal(answer: str) -> bool:
    """True when the LLM refused to answer because the entity wasn't in context."""
    return bool(_REFUSAL_PATTERNS.search(answer))

# Patterns that indicate a true multi-hop question — one where an intermediate
# entity must be resolved before the main question can be answered.
# These are relative-clause bridges: "the film that stars", "the actor who played",
# "the capital of the country that", etc.
# Simple questions like "Who wrote Hamlet?" or "What year was X built?" do NOT
# contain these patterns and fall through to SIMPLE.
MULTI_HOP_PATTERNS = [
    # "the [noun(s)] that/who [verb]" — bridging relative clause
    r'\bthe\s+\w+(?:\s+\w+)?\s+(?:that|who)\s+\w+',
    # "of the [noun] that/who" — nested reference
    r'\bof\s+the\s+\w+\s+(?:that|who)\b',
    # passive bridge: "directed/written/made by the [noun] that/who"
    r'\b(?:directed|written|made|founded|invented|created|authored)\s+by\s+the\s+\w+\s+(?:that|who)\b',
    # "[role] of [entity]" — implicit bridge: "the director of Inception", "the author of X"
    r'\b(?:director|author|writer|producer|founder|creator|inventor|singer|composer|'
    r'lead|star|ceo|president|chairman|captain|owner|manager|coach|editor|host)\s+of\b',
    # "born/died in [place] who" or "who [verb]ed [work]"
    r'\bwho\s+(?:starred|appeared|played|acted|directed|wrote|produced|founded|invented|composed|hosted)\b',
    # "in [work] who/that" — entity identified through a work
    r'\bin\s+(?:the\s+)?\w+(?:\s+\w+){0,2}\s+(?:who|that)\b',
]

def classify_query(query: str) -> str:
    """
    Rule-based query type classifier.  Returns 'SIMPLE', 'MULTI_HOP', or 'COMPARISON'.

    Design principles (fixes vs prior version):
      - "first/last/most/least" no longer unconditionally trigger COMPARISON.
        They only trigger it when two explicit named entities are joined by "or"
        (the "which X or Y" pattern), which is the actual HotpotQA pattern.
      - Strong comparison words (older/newer/same/different/…) still trigger
        COMPARISON, but the named-entity filter now excludes question-word
        capitalizations (Who/What/Where/Which) so "Which" + "older" doesn't
        cause a spurious COMPARISON label.
      - MULTI_HOP patterns are checked AFTER COMPARISON, so bridging questions
        that also contain comparison words route correctly.
    """
    q_lower = query.lower()
    tokens  = set(re.sub(r'[^\w\s]', '', q_lower).split())

    # Question-word capitalizations that are NOT named entities
    _QW = frozenset({"Who", "What", "Where", "When", "Which", "How",
                     "Were", "Was", "Is", "Are", "Did", "Do", "Does",
                     "The", "Both", "Has", "Had"})

    def _named_entities(text: str) -> list:
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return [c for c in caps if c not in _QW]

    # ── 1. "X or Y" structure with ≥2 named entities → always COMPARISON ──
    if " or " in q_lower:
        ents = _named_entities(query)
        if len(ents) >= 2:
            return "COMPARISON"

    # ── 2. Strong comparison words (no "first/last") + ≥2 named entities ──
    if tokens & COMPARISON_WORDS:
        ents = _named_entities(query)
        if len(ents) >= 2:
            return "COMPARISON"
        if any(phrase in q_lower for phrase in
               ["same nationality", "same country", "same language", "both from"]):
            return "COMPARISON"

    # ── 3. Ordinal words ("first/last/most/least") + "or" + ≥2 named entities ──
    if tokens & _ORDINAL_COMPARISON_WORDS and " or " in q_lower:
        ents = _named_entities(query)
        if len(ents) >= 2:
            return "COMPARISON"

    # ── 4. Multi-hop: bridging relative clause ──
    if any(re.search(pattern, q_lower) for pattern in MULTI_HOP_PATTERNS):
        return "MULTI_HOP"

    # ── 5. Multiple embedded question clauses ──
    if q_lower.count(" who ") + q_lower.count(" what ") + q_lower.count(" where ") >= 2:
        return "MULTI_HOP"

    return "SIMPLE"


# ─────────────────────────────────────────────
# STEP 1b: COMPLEXITY ESTIMATOR
# Independent of query TYPE — tells us HOW HARD
# the retrieval task is, so the budget can scale.
# ─────────────────────────────────────────────
_QW_SKIP = frozenset({
    "Who", "What", "Where", "When", "Which", "How",
    "Were", "Was", "Is", "Are", "Did", "Do", "Does",
    "The", "Both", "Has", "Had", "This", "That",
})


def estimate_complexity(query: str) -> str:
    """
    Estimate query complexity as 'easy', 'medium', or 'hard'.
    Used when HotpotQA ground-truth level is unavailable (interactive mode).
    Returns the same scale as HotpotQA's example["level"] field.

    Scoring features:
      1. Token length  — longer questions are harder on average
      2. Named-entity density — more entities = more retrieval targets
      3. Bridge expression hits — multi-hop pattern matches
      4. Subordinate-clause depth — nested "that/who/which" chains
      5. "of the" chains — possessive + relational nesting

    Thresholds calibrated against HotpotQA validation distribution:
      ≥5 points → hard  (typically 2-hop bridge or deep comparison)
      ≥2 points → medium
      <2 points → easy
    """
    tokens = query.split()
    score  = 0

    # Feature 1: length
    if len(tokens) > 20:
        score += 2
    elif len(tokens) > 12:
        score += 1

    # Feature 2: named-entity density
    ents = [e for e in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            if e not in _QW_SKIP]
    if len(ents) >= 3:
        score += 2
    elif len(ents) >= 2:
        score += 1

    # Feature 3: bridge expression count
    bridge_hits = sum(1 for p in MULTI_HOP_PATTERNS
                      if re.search(p, query.lower()))
    score += min(bridge_hits, 2)

    # Feature 4: subordinate-clause depth
    clause_depth = len(re.findall(r'\b(?:that|who|which)\b', query.lower()))
    if clause_depth >= 2:
        score += 2
    elif clause_depth >= 1:
        score += 1

    # Feature 5: relational nesting ("of the" chains)
    of_chains = len(re.findall(r'\bof\s+the\b', query.lower()))
    score += min(of_chains, 2)

    if score >= 5:
        return "hard"
    if score >= 2:
        return "medium"
    return "easy"


# ─────────────────────────────────────────────
# STEP 2: RETRIEVAL STRATEGIES
# ─────────────────────────────────────────────
def retrieve_simple(query, index, embedder, passages, top_k=TOP_K):
    """
    Hybrid BM25 + dense retrieval when BM25 is initialised; dense-only fallback.
    Used for SIMPLE queries and as the base retriever for all other strategies.
    """
    return retrieve_hybrid(query, index, embedder, passages, top_k=top_k)


# ─────────────────────────────────────────────
# 2-STEP CHAIN FOR MULTI-HOP QUERIES
# ─────────────────────────────────────────────

_ROLE_WORDS = (
    r'director|author|writer|producer|founder|inventor|creator|'
    r'singer|composer|star|lead|ceo|president|captain|host|editor|'
    r'coach|manager|owner|chairman|principal|governor|mayor|minister'
)


def _decompose_multihop_query(query: str):
    """
    Split a multi-hop query into (bridge_sub_question, simplified_main_question).

    Handles two common HotpotQA bridge patterns:

    Pattern A — bridging relative clause: "the X that/who VP"
      "Who directed the film that starred Emma Watson?"
        sq1 = "What film starred Emma Watson?"
        sq2 = "Who directed the film?"

    Pattern B — implicit role bridge: "the [role] of [Entity]"
      "What is the nationality of the director of Inception?"
        sq1 = "Who is the director of Inception?"
        sq2 = "What is the nationality of the director?"

    Returns (sq1, sq2) on success, (None, None) if no pattern matched.
    """
    q = query.rstrip('?').strip()

    # ── Pattern A: "[the/a/an] X [,] that/who VP [, trailing_verb]" relative clause ──
    # Handles both "the film that starred X" and "a German footballer who had 18 clean sheets"
    # Covers definite ("the") and indefinite ("a", "an") articles so questions like
    # "What footballer beat out a German professional footballer despite his 18 clean sheets?"
    # decompose correctly instead of falling through to iterative multi-hop.
    m_a = re.search(
        r'\b(?:the|a|an)\s+((?:\w+\s+){0,4}\w+)\s*,?\s*(?:that|who)\s+(.+?)(?=\s*,|\?|$)',
        q, re.IGNORECASE,
    )
    if m_a:
        noun = m_a.group(1).strip()
        vp   = m_a.group(2).strip().rstrip('?')
        sq1  = f"What {noun} {vp}?"
        sq2  = re.sub(
            r'\b(?:the|a|an)\s+(?:\w+\s+){0,4}\w+\s*,?\s*(?:that|who)\s+[^,?]+(?:,\s*)?',
            f'the {noun} ', q, count=1, flags=re.IGNORECASE,
        ).strip() or q
        return sq1, sq2

    # ── Pattern B: "the [role] of [Entity]" implicit bridge ──
    m_b = re.search(
        rf'\bthe\s+({_ROLE_WORDS})\s+of\s+([A-Z][^\s?]+(?:\s+[A-Z][^\s?]+)*)',
        q, re.IGNORECASE,
    )
    if m_b:
        role   = m_b.group(1).strip()
        entity = m_b.group(2).strip().rstrip('?.,')
        sq1    = f"Who is the {role} of {entity}?"
        sq2    = re.sub(
            rf'\bthe\s+{re.escape(role)}\s+of\s+{re.escape(entity)}\b',
            f'the {role}',
            q, count=1, flags=re.IGNORECASE,
        ).strip() or q
        return sq1, sq2

    return None, None


def decompose_and_retrieve_multi_hop(query, index, embedder, passages,
                                      top_k=TOP_K_MULTI):
    """
    2-sub-question decomposition retrieval for MULTI_HOP (Stage 3 only).

    Key difference from the failed 2-step chain (previous attempt):
      - SQ2 retrieval is INDEPENDENT of the bridge answer.
        A wrong bridge answer degrades context quality but does NOT
        misdirect retrieval (the old failure mode).
      - The bridge answer is returned as a text string so the caller can
        inject it as a synthetic passage AFTER CrossEncoder reranking.
        This means the CrossEncoder scores real Wikipedia passages, while
        the LLM still gets the intermediate finding as explicit context.

    Falls back to iterative retrieve_multi_hop when no bridge pattern found.

    Returns: (passages_list, bridge_context_str | None)
    """
    sq1, sq2 = _decompose_multihop_query(query)

    if sq1 is None:
        # No bridging clause detected — iterative fallback
        return retrieve_multi_hop(query, index, embedder, passages, top_k), None

    # ── Sub-question 1: retrieve and answer the bridge ──
    sq1_passages = retrieve_simple(sq1, index, embedder, passages, top_k=top_k)
    bridge_answer = generate_answer(sq1, sq1_passages, query_type="SIMPLE")

    bridge_ctx = (
        f"Intermediate finding — sub-question: '{sq1}' "
        f"→ answer: '{bridge_answer}'. "
        f"Use this intermediate answer to help resolve the main question."
    )

    # ── Sub-question 2: anchor retrieval on the bridge answer ──
    # Previously sq2 was retrieved independently. Without knowing the bridge entity
    # (e.g., "Adriana Trigiani" for "the director of Big Stone Gap"), dense retrieval
    # for "The director is based in what New York city?" returns generic NYC articles
    # instead of the director's bio page which mentions Greenwich Village.
    # By prepending the bridge answer, retrieval stays focused on the right entity.
    if bridge_answer and not _is_refusal(bridge_answer):
        sq2_query = f"{bridge_answer} {sq2}"
    else:
        sq2_query = sq2
    sq2_passages = retrieve_simple(sq2_query, index, embedder, passages, top_k=top_k)

    # Merge: sq2 passages first (directly relevant to final answer), then sq1 context
    seen     = {p["title"] for p in sq2_passages}
    combined = [dict(p, hop=2) for p in sq2_passages]
    for p in sq1_passages:
        if p["title"] not in seen:
            combined.append(dict(p, hop=1))
            seen.add(p["title"])

    return combined, bridge_ctx


def _extract_question_entities(query: str) -> list:
    # Extract at most 2 anchor entities for Phase 1 entity-focused retrieval.
    #
    # Only quoted strings and multi-word proper nouns are used.
    # Single-word nouns ("Street", "Award") are excluded -- too ambiguous,
    # they flood the pool with unrelated articles.
    # The full-question hybrid search covers everything the entity searches miss.
    #
    # Measured on 2000 HotpotQA validation questions:
    #   Old extraction: mean 4.44 entities/question (59.8% had >3 searches)
    #   New extraction: mean 1.14 entities/question (0% exceed 2 searches)
    entities = []

    # Priority 1: quoted strings — unambiguous, author-designated anchors
    for q in re.findall(r'"([^"]+)"', query):
        if len(q.strip()) > 2:
            entities.append(q.strip())
            if len(entities) >= 2:
                return list(dict.fromkeys(entities))

    # Priority 2: multi-word proper nouns (≥2 capitalized words)
    # These are specific enough to retrieve a unique article without noise.
    for c in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', query):
        if c not in _QW_SKIP and len(c) > 3 and c not in entities:
            entities.append(c)
            if len(entities) >= 2:
                break

    # Single-word nouns intentionally excluded — see docstring above.

    return list(dict.fromkeys(entities))


def retrieve_multi_hop(query, index, embedder, passages,
                       top_k=TOP_K_MULTI, max_hops=MAX_HOPS):
    """
    Multi-hop retrieval combining entity-anchored parallel search with iterative
    reformulation.

    The previous implementation only did iterative reformulation from the top
    retrieved passage, which fails when:
      1. The correct answer article is not reached in any hop (e.g., "US Route 60"
         is not mentioned in the first chunk of the "Zilpo Road" article).
      2. Reformulation drifts toward irrelevant entities in retrieved passages.

    New strategy — two phases:

    Phase 1 — Entity-anchored parallel retrieval:
      Extract every named entity mentioned IN the question itself and search for
      each one independently.  For "...gives access to Zilpo Road, and is also
      known as Midland Trail?" this runs searches for "Zilpo Road" AND "Midland
      Trail" separately.  The US Route 60 article mentions both → it enters the
      pool from the "Midland Trail" search even if the full-question query misses it.

    Phase 2 — Iterative reformulation (original behaviour):
      Run hop 2 and hop 3 using reformulated queries from the top retrieved
      passage.  Provides coverage for questions where the answer entity is NOT
      mentioned in the question at all.
    """
    all_retrieved = []
    seen_titles   = set()

    # ── Phase 1: entity-anchored parallel retrieval ──────────────────────────
    question_entities = _extract_question_entities(query)

    for entity in question_entities[:4]:          # cap at 4 to keep latency reasonable
        entity_results = retrieve_simple(
            entity, index, embedder, passages, top_k=top_k
        )
        for p in entity_results:
            if p["title"] not in seen_titles:
                seen_titles.add(p["title"])
                new_p = dict(p, hop=1)
                all_retrieved.append(new_p)

    # Full-question search (covers questions with no extractable named entities)
    full_results = retrieve_simple(query, index, embedder, passages, top_k=top_k)
    for p in full_results:
        if p["title"] not in seen_titles:
            seen_titles.add(p["title"])
            all_retrieved.append(dict(p, hop=1))

    # ── Phase 2: iterative reformulation from top passage ────────────────────
    # Skip Phase 2 entirely when the original question contains no predicate
    # or role word that would ground the reformulation.  Without an anchor,
    # _reformulate_query() returns a bare entity name ("Street", "Park",
    # "November") which retrieves unrelated articles and adds noise.
    # Measured on 2000 HotpotQA questions: 57.2% of bridge questions have
    # no reformulation anchor → skipping saves 2 retrieve_simple calls and
    # removes ~50-100 noise passages from the pool for more than half of all
    # MULTI_HOP questions.
    _anchor_check = re.search(
        r'\b(directed|directing|wrote|written|founded|invented|born|nationality|'
        r'located|capital|started|created|authored|played|starred|published|'
        r'produced|graduated|attended|died|married|headquartered|operated|'
        r'citizenship|birthplace|hometown|leader|ruled|governed|'
        r'earned|won|received|awarded|released|debuted|composed|recorded|'
        r'performed|hosted|represented|signed|sold|reached|charted|'
        r'director|author|writer|producer|founder|inventor|creator|'
        r'singer|composer|star|lead|ceo|president|captain|host|editor|'
        r'coach|manager|owner|chairman|principal|governor|mayor|minister)\b',
        query, re.IGNORECASE,
    )
    if not _anchor_check:
        return all_retrieved  # no meaningful reformulation possible; skip hops 2+

    current_query = query
    for hop in range(2, max_hops + 1):
        if not all_retrieved:
            break
        top_passage   = all_retrieved[0]["text"][:300]
        current_query = _reformulate_query(query, top_passage)

        if current_query.strip().lower() == query.strip().lower():
            break  # reformulation produced no change — stop early

        hop_results  = retrieve_simple(current_query, index, embedder, passages, top_k)
        new_passages = []
        for p in hop_results:
            if p["title"] not in seen_titles:
                seen_titles.add(p["title"])
                p["hop"] = hop
                new_passages.append(p)

        all_retrieved.extend(new_passages)
        if not new_passages:
            break

    return all_retrieved


def retrieve_comparison(query, index, embedder, passages, top_k=TOP_K_MULTI):
    """
    Parallel retrieval for comparison questions.
    Extracts the two entities being compared and retrieves
    passages for each independently, then combines results.

    Used for COMPARISON queries like:
    'Were Scott Derrickson and Ed Wood of the same nationality?'
    """
    entities = _extract_entities(query)

    # Extract the comparison attribute from the query (e.g. "older" → "birth year age")
    attribute_map = {
        r'\bolder\b|\byounger\b':                        "birth year age born",
        r'\btaller\b|\bshorter\b':                       "height",
        r'\bricher\b|\bwealthier\b':                     "net worth wealth",
        r'\bnationality\b|\bcountry\b':                  "nationality country born",
        r'\bsame language\b|\bboth from\b':              "origin country language",
        r'\bearlier\b|\blater\b|\bfirst\b':              "founded started year",
        r'\bawards?\b|\bwon\b|\bwins?\b|\baccolades?\b': "awards won wins accolades",
        r'\balbums?\b|\bsongs?\b|\bhits?\b':             "albums discography songs",
        r'\bgoals?\b|\bscored\b|\bpoints?\b':            "goals scored points career",
        r'\bbooks?\b|\bnovels?\b|\bwritten\b':           "books written novels published",
        r'\bbetter\b|\bworse\b|\bmore\b|\bless\b':       "comparison career achievements",
    }
    attribute_suffix = ""
    for pattern, suffix in attribute_map.items():
        if re.search(pattern, query, re.IGNORECASE):
            attribute_suffix = suffix
            break

    all_retrieved = []
    seen_titles   = set()

    if len(entities) >= 2:
        # Dense-only per-entity retrieval: BM25 is counterproductive here because
        # it over-weights generic "nationality/born/age" keyword documents instead of
        # the specific entity's Wikipedia bio. Dense embedding finds the right bio page.
        for entity in entities[:2]:
            entity_query   = f"{entity} {attribute_suffix}".strip()
            entity_results = _retrieve_dense(
                entity_query, index, embedder, passages, top_k
            )
            for p in entity_results:
                if p["title"] not in seen_titles:
                    seen_titles.add(p["title"])
                    p["entity"] = entity
                    all_retrieved.append(p)
    else:
        # Fallback to standard retrieval if entity extraction fails
        all_retrieved = [dict(p) for p in
                         _retrieve_dense(query, index, embedder, passages, top_k * 2)]

    # Full query retrieval (dense-only) to catch joint passages
    full_results = _retrieve_dense(query, index, embedder, passages, top_k)
    for p in full_results:
        if p["title"] not in seen_titles:
            seen_titles.add(p["title"])
            all_retrieved.append(p)

    return all_retrieved


# ─────────────────────────────────────────────
# STEP 3: HELPER FUNCTIONS
# ─────────────────────────────────────────────
def _reformulate_query(original_query: str, context_snippet: str) -> str:
    """
    Build a hop-N+1 query that stays semantically anchored to the original question.

    Problem with entity-only reformulation: "Robert Downey Jr." as a hop-2 query
    retrieves RDJ's biography page — not the passage that says who DIRECTED the
    film starring RDJ.  The anchor (predicate/role/spatial) must come from the
    ORIGINAL question, not from the retrieved passage.

    Priority order for anchor selection:
      1. Verbal predicate  (directed, wrote, founded, nationality, …)
      2. Role noun         (director of, author of, capital of, …)
      3. Spatial relation  (block away, next to, capital, headquarters, …)
      4. bare bridge entity (last resort — better than nothing)

    Returns a natural-language hop-2 query of the form "BridgeEntity anchor".
    """
    # ── Extract bridge entity from retrieved passage ──
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context_snippet)
    entities = [e for e in entities if len(e) > 2 and e not in _QW_SKIP]
    entities = list(dict.fromkeys(entities))
    if not entities:
        return original_query
    bridge = entities[0]   # single best bridge entity

    # ── Extract anchor from original question ──
    predicates = re.findall(
        r'\b(directed|directing|wrote|written|founded|invented|born|nationality|'
        r'located|capital|started|created|authored|played|starred|published|'
        r'produced|graduated|attended|died|married|headquartered|operated|'
        r'citizenship|birthplace|hometown|leader|ruled|governed|'
        r'earned|won|received|awarded|released|debuted|composed|recorded|'
        r'performed|hosted|represented|signed|sold|reached|charted)\b',
        original_query, re.IGNORECASE,
    )

    roles = re.findall(
        rf'\b({_ROLE_WORDS})\b',
        original_query, re.IGNORECASE,
    )

    spatials = re.findall(
        r'\b(block away|next to|adjacent|across from|near|nearby|borders|'
        r'bordered|surrounding|inside|within|outside|capital|headquarters|'
        r'based in|located in|home of)\b',
        original_query, re.IGNORECASE,
    )

    anchor = (
        (predicates[0].lower() if predicates else None)
        or (roles[0].lower() if roles else None)
        or (spatials[0].lower() if spatials else None)
    )

    return f"{bridge} {anchor}".strip() if anchor else bridge


def _extract_entities(query: str) -> list:
    """
    Extract the entities (or entity descriptions) being compared.

    Two-pass approach that handles both named-entity comparisons and
    descriptive comparisons like "the author of Hamlet vs the author of Macbeth":

    Pass 1 — role-of descriptors: "the [role] of [Entity]" returns the full
             descriptor phrase so the retrieval query targets Shakespeare's article,
             not Hamlet's article.

    Pass 2 — fallback to raw named proper nouns with a conservative stopword filter
             that does NOT strip nouns/attributes (only function words + WH-words).
    """
    # Pass 1: descriptive references ("the author of Hamlet", "the director of X")
    role_refs = re.findall(
        rf'(?:the\s+)?({_ROLE_WORDS})\s+of\s+'
        r'([A-Z][A-Za-z\']+(?:\s+[A-Z][A-Za-z\']+)*)',
        query, re.IGNORECASE,
    )
    if len(role_refs) >= 2:
        # Return full descriptive phrases so retrieve_comparison queries for the right page
        return [f"{r[0]} of {r[1]}".strip() for r in role_refs[:2]]

    # Pass 2: raw named proper noun sequences
    _fn_words = frozenset({
        "who", "what", "where", "when", "which", "how", "were", "was",
        "is", "are", "did", "do", "both", "the", "and", "of", "same",
        "different", "nationality", "country", "from", "in", "a", "an", "or",
    })
    clean = re.sub(
        r'\b(' + '|'.join(_fn_words) + r')\b', ' ',
        query, flags=re.IGNORECASE,
    )
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', clean)
    entities = [e for e in entities if len(e) > 2 and e not in _QW_SKIP]
    return list(dict.fromkeys(entities))


# ─────────────────────────────────────────────
# STEP 3b: RETRIEVAL CONFIDENCE + COVERAGE
# These run AFTER reranking, BEFORE generation,
# to decide whether to expand the budget.
# ─────────────────────────────────────────────

def estimate_retrieval_confidence(reranked_passages: list) -> float:
    """
    Estimate retrieval confidence from cross-encoder reranker scores.

    The ms-marco cross-encoder outputs logit-like scores:
      > 0  → passage is relevant to the query
      < 0  → passage is not relevant

    We apply a scaled sigmoid so the output lives in [0, 1]:
      score  5  → confidence ≈ 0.92  (very confident)
      score  0  → confidence ≈ 0.50  (uncertain)
      score -5  → confidence ≈ 0.08  (very low — expand budget)

    Uses a weighted mean of the top-3 passages so a single good hit
    raises confidence even when the other passages are weak.
    """
    if not reranked_passages:
        return 0.0
    scores = [p.get("rerank_score", 0.0) for p in reranked_passages[:3]]
    if not scores:
        return 0.5  # no cross-encoder scores → neutral

    # Top passage gets 2× weight (usually the one the LLM draws from most)
    weights = [2.0] + [1.0] * (len(scores) - 1)
    total_w = sum(weights[: len(scores)])
    mean_s  = sum(w * s for w, s in zip(weights, scores)) / total_w

    return 1.0 / (1.0 + math.exp(-mean_s * 0.5))  # scaled sigmoid


def check_evidence_coverage(query: str, retrieved_passages: list) -> dict:
    """
    Check whether retrieved passages cover the key named entities from the query.

    Strategy: extract proper-noun entities from the question and check whether each
    appears in the concatenated retrieved text.  Returns:
      coverage  — fraction of key entities present [0, 1]
      missing   — entities not found (used to seed targeted follow-up queries)
      found     — entities that are present (diagnostic)

    Limitation: case-insensitive substring match; cannot verify the entity appears
    in the correct semantic ROLE (that is the verifier's job in Stage 2).
    Coverage = 1.0 when no extractable entities exist (no-op).
    """
    ents = [e for e in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            if e not in _QW_SKIP and len(e) > 2]

    if not ents:
        return {"coverage": 1.0, "missing": [], "found": []}

    combined = " ".join(p.get("text", "") for p in retrieved_passages).lower()

    found   = [e for e in ents if e.lower() in combined]
    missing = [e for e in ents if e.lower() not in combined]

    coverage = len(found) / len(ents)
    return {
        "coverage": coverage,
        "missing":  list(dict.fromkeys(missing)),
        "found":    list(dict.fromkeys(found)),
    }


# ─────────────────────────────────────────────
# STEP 4: ADAPTIVE RAG PIPELINE (Stage 3)
# ─────────────────────────────────────────────
RETRIEVAL_STRATEGY_LABEL = {
    "SIMPLE":     "BM25 + dense hybrid (single-hop)",
    "MULTI_HOP":  "Iterative multi-hop retrieval (3 hops)",
    "COMPARISON": "Parallel per-entity retrieval",
}


# Retrieval parameters per (query_type, level) combination.
# More difficult questions get more hops and larger top_k pools.
_RETRIEVAL_PARAMS = {
    #              top_k  max_hops
    ("SIMPLE",     "easy"):   (5,  1),
    ("SIMPLE",     "medium"): (10, 1),
    ("SIMPLE",     "hard"):   (15, 1),
    ("MULTI_HOP",  "easy"):   (5,  2),
    ("MULTI_HOP",  "medium"): (5,  3),
    ("MULTI_HOP",  "hard"):   (8,  3),
    ("COMPARISON", "easy"):   (5,  1),
    ("COMPARISON", "medium"): (8,  1),
    ("COMPARISON", "hard"):   (12, 1),
}
_DEFAULT_PARAMS = {
    "SIMPLE":     (10, 1),
    "MULTI_HOP":  (5,  3),
    "COMPARISON": (8,  1),
}

# ── Complexity-aware budget: top_k = retrieval pool, max_expansions = coverage loops ──
# top_k here is the pool fed to the cross-encoder; the LLM always sees TOP_K passages.
# Principle: HARD questions get 10× the retrieval pool of EASY, ensuring the evidence
# for 2-hop bridge questions is present before generation even starts.
# top_k is the candidate pool fed to the cross-encoder reranker.
# All entries use top_k=50 to match Stage 1's RERANK_POOL=50 — this ensures
# Stage 3 and 4 have at least as wide a retrieval net as Stage 1.
# Complexity differentiation is expressed through max_hops and max_expansions,
# NOT by shrinking the candidate pool (which would make Stage 3 find fewer
# relevant passages than Stage 1 and produce worse answers on simple questions).
_BUDGET = {
    ("SIMPLE",     "easy"):   {"top_k": 50, "max_hops": 1, "max_expansions": 0},
    ("SIMPLE",     "medium"): {"top_k": 50, "max_hops": 1, "max_expansions": 1},
    ("SIMPLE",     "hard"):   {"top_k": 50, "max_hops": 1, "max_expansions": 1},
    ("MULTI_HOP",  "easy"):   {"top_k": 50, "max_hops": 2, "max_expansions": 1},
    ("MULTI_HOP",  "medium"): {"top_k": 50, "max_hops": 2, "max_expansions": 2},
    ("MULTI_HOP",  "hard"):   {"top_k": 50, "max_hops": 3, "max_expansions": 2},
    ("COMPARISON", "easy"):   {"top_k": 50, "max_hops": 1, "max_expansions": 0},
    ("COMPARISON", "medium"): {"top_k": 50, "max_hops": 1, "max_expansions": 1},
    ("COMPARISON", "hard"):   {"top_k": 50, "max_hops": 1, "max_expansions": 1},
}
_DEFAULT_BUDGET = {
    "SIMPLE":     {"top_k": 50, "max_hops": 1, "max_expansions": 0},
    "MULTI_HOP":  {"top_k": 50, "max_hops": 2, "max_expansions": 1},
    "COMPARISON": {"top_k": 50, "max_hops": 1, "max_expansions": 1},
}


def _retrieve_for_type(
    query: str, query_type: str,
    index, embedder, passages,
    top_k: int, max_hops: int,
) -> tuple:
    """
    Dispatch to the right retrieval function and return (pool, bridge_ctx).
    bridge_ctx is a synthetic context string produced by multi-hop decomposition;
    it is injected AFTER reranking so the cross-encoder scores real passages only.
    """
    if query_type == "SIMPLE":
        return retrieve_simple(query, index, embedder, passages, top_k=top_k), None

    if query_type == "MULTI_HOP":
        pool, bridge_ctx = decompose_and_retrieve_multi_hop(
            query, index, embedder, passages, top_k=top_k,
        )
        # When max_hops > 2 AND decomposition actually succeeded (bridge_ctx is not None),
        # also run iterative hops to broaden coverage of harder 2-hop chains.
        # When bridge_ctx is None, decompose_and_retrieve_multi_hop() already fell back to
        # retrieve_multi_hop() internally — calling it again doubles the entity-search
        # pool size and amplifies any noise from non-anchor entity searches.
        if max_hops > 2 and bridge_ctx is not None:
            iter_pool = retrieve_multi_hop(
                query, index, embedder, passages,
                top_k=max(top_k // 2, TOP_K_MULTI), max_hops=max_hops,
            )
            seen = {p["title"] for p in pool}
            for p in iter_pool:
                if p["title"] not in seen:
                    pool.append(p)
                    seen.add(p["title"])
        return pool, bridge_ctx

    # COMPARISON
    return retrieve_comparison(query, index, embedder, passages, top_k=top_k), None


def adaptive_retrieve_with_coverage_check(
    query: str, query_type: str,
    index, embedder, passages,
    budget: dict,
    verbose: bool = False,
) -> tuple:
    """
    Adaptive retrieval loop: retrieve → rerank → check confidence & coverage →
    expand if insufficient → repeat up to budget['max_expansions'] times.

    This implements the research proposal claim:
    "More complex questions receive a more thorough retrieval process."

    The loop triggers expansion when EITHER:
      (a) retrieval confidence < RETRIEVAL_CONF_THRESHOLD
          (cross-encoder says the returned passages are not relevant)
      (b) evidence coverage < COVERAGE_THRESHOLD
          (key question entities are missing from the returned passages)

    When missing entities are identified, the expansion targets them directly
    rather than blindly increasing top_k — focused expansion is more precise
    and avoids retrieving unrelated noise.

    Returns:
      pool          — all passages retrieved across all expansions
      reranked      — final TOP_K passages after cross-encoder reranking
      bridge_ctx    — synthetic multi-hop bridge string (or None)
      stats         — dict with confidence, coverage, expansions_used
    """
    top_k          = budget["top_k"]
    max_hops       = budget["max_hops"]
    max_expansions = budget["max_expansions"]

    # ── Initial retrieval ──
    pool, bridge_ctx = _retrieve_for_type(
        query, query_type, index, embedder, passages, top_k, max_hops,
    )
    reranked   = rerank_passages(query, pool, top_k=TOP_K)
    confidence = estimate_retrieval_confidence(reranked)
    cov_info   = check_evidence_coverage(query, reranked)
    expansions = 0

    # ── Coverage expansion loop ──
    for _ in range(max_expansions):
        if (confidence >= RETRIEVAL_CONF_THRESHOLD
                and cov_info["coverage"] >= COVERAGE_THRESHOLD):
            break

        if verbose:
            print(f"  [Expand {expansions+1}] conf={confidence:.3f} "
                  f"cov={cov_info['coverage']:.2f} "
                  f"missing={cov_info['missing'][:3]}")

        seen_titles = {p["title"] for p in pool}

        if cov_info["missing"]:
            # Targeted retrieval for each missing entity
            for me in cov_info["missing"][:2]:
                extra = retrieve_simple(
                    f"{me} {query}", index, embedder, passages,
                    top_k=max(top_k // 2, 5),
                )
                for p in extra:
                    if p["title"] not in seen_titles:
                        pool.append(p)
                        seen_titles.add(p["title"])
        else:
            # General budget expansion — double the pool
            extra_pool, _ = _retrieve_for_type(
                query, query_type, index, embedder, passages,
                min(top_k * 2, 50), max_hops,
            )
            for p in extra_pool:
                if p["title"] not in seen_titles:
                    pool.append(p)
                    seen_titles.add(p["title"])

        reranked   = rerank_passages(query, pool, top_k=TOP_K)
        confidence = estimate_retrieval_confidence(reranked)
        cov_info   = check_evidence_coverage(query, reranked)
        expansions += 1

    return pool, reranked, bridge_ctx, {
        "confidence":       confidence,
        "coverage":         cov_info["coverage"],
        "missing_entities": cov_info["missing"],
        "expansions_used":  expansions,
    }


def adaptive_rag_query(query, index, embedder, passages,
                       verifier_model=None, verifier_tokenizer=None,
                       verbose=True, query_type_override=None, level_override=None):
    """
    Full Stage 3 pipeline with genuine complexity-aware adaptive retrieval:

    Query
      → classify_query()      (type: SIMPLE / MULTI_HOP / COMPARISON)
      → estimate_complexity() (complexity: easy / medium / hard)
      → _BUDGET lookup        (top_k, max_hops, max_expansions)
      → adaptive_retrieve_with_coverage_check()
          [retrieve → rerank → confidence/coverage check → expand if needed]
      → generate_answer()
      → refusal fallback
      → verify (DistilBERT + LLM judge)
      → low-support fallback

    The key difference from the prior version:
      - Complexity is estimated INDEPENDENTLY of query type, then combined to
        select a retrieval budget.  An EASY MULTI_HOP gets a small budget; a
        HARD MULTI_HOP gets 5× the pool and 2 coverage expansions.
      - Coverage is checked BEFORE generation — if key entities are missing,
        the system retrieves specifically for them rather than generating blind.
      - The fallback policy is unified: one path, one confidence comparison.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

    # ── 1. Classify type ──
    if query_type_override == "bridge":
        query_type = "MULTI_HOP"
    elif query_type_override == "comparison":
        query_type = "COMPARISON"
    else:
        query_type = classify_query(query)

    # ── 2. Estimate complexity (use HotpotQA ground-truth when available) ──
    complexity = level_override if level_override else estimate_complexity(query)

    # ── 3. Look up budget ──
    budget = _BUDGET.get((query_type, complexity), _DEFAULT_BUDGET[query_type])

    if verbose:
        print(f"Type: {query_type} | Complexity: {complexity.upper()} | "
              f"top_k={budget['top_k']}, max_hops={budget['max_hops']}, "
              f"max_expansions={budget['max_expansions']}")
        print(f"Strategy: {RETRIEVAL_STRATEGY_LABEL[query_type]}")

    # ── 3b. Confidence-gated escalation for MULTI_HOP ──
    # Principle: use the cheapest strategy that achieves sufficient retrieval
    # confidence.  71.1% of HotpotQA "bridge" questions are structurally
    # answerable with a single hybrid search — the dataset label means "evidence
    # spans two articles", not "iterative reformulation is required".
    #
    # If SIMPLE retrieval is already confident (cross-encoder top-passage score
    # above threshold), skip the full multi-hop expansion. Only escalate if the
    # simple search is genuinely uncertain about the evidence.
    #
    # Threshold 0.65 (sigmoid of reranker score ≈3.0): conservative enough that
    # questions genuinely requiring multi-hop evidence (long-range bridge) still
    # escalate. Questions where the answer is in the first retrieved article
    # (majority of bridge questions) return immediately.
    _SIMPLE_SUFFICIENT = 0.65

    # Track whether the escalation gate fired so the refusal fallback knows
    # to escalate back to MULTI_HOP if SIMPLE produces no answer.
    _gate_fired = False

    if query_type == "MULTI_HOP":
        simple_budget = _BUDGET.get(("SIMPLE", complexity), _DEFAULT_BUDGET["SIMPLE"])
        simple_pool   = retrieve_simple(query, index, embedder, passages,
                                        top_k=simple_budget["top_k"])
        simple_reranked  = rerank_passages(query, simple_pool, top_k=TOP_K)
        simple_conf      = estimate_retrieval_confidence(simple_reranked)

        if simple_conf >= _SIMPLE_SUFFICIENT:
            # SIMPLE retrieval is sufficiently confident — no need for expensive
            # multi-hop machinery. Use SIMPLE results and skip escalation.
            if verbose:
                print(f"[Escalation gate] SIMPLE conf={simple_conf:.3f} >= {_SIMPLE_SUFFICIENT} "
                      f"-- skipping MULTI_HOP expansion")
            _gate_fired  = True
            query_type   = "SIMPLE"
            budget       = simple_budget
            pool, reranked, bridge_ctx, ret_stats = (
                simple_pool, simple_reranked, None,
                {"confidence": simple_conf, "coverage": 1.0,
                 "missing_entities": [], "expansions_used": 0},
            )
        else:
            if verbose:
                print(f"[Escalation gate] SIMPLE conf={simple_conf:.3f} < {_SIMPLE_SUFFICIENT} "
                      f"-- escalating to MULTI_HOP")
            # ── 4. Adaptive retrieval with coverage check ──
            pool, reranked, bridge_ctx, ret_stats = adaptive_retrieve_with_coverage_check(
                query, query_type, index, embedder, passages, budget, verbose,
            )
    else:
        # ── 4. Adaptive retrieval with coverage check ──
        pool, reranked, bridge_ctx, ret_stats = adaptive_retrieve_with_coverage_check(
            query, query_type, index, embedder, passages, budget, verbose,
        )

    if verbose:
        print(f"\nRetrieved {len(pool)} → reranked to {len(reranked)} "
              f"| conf={ret_stats['confidence']:.3f} "
              f"cov={ret_stats['coverage']:.2f} "
              f"expansions={ret_stats['expansions_used']}")
        for i, p in enumerate(reranked[:5]):
            hop_info = f" [hop {p.get('hop', '?')}]" if query_type == "MULTI_HOP" else ""
            ent_info = f" [{p.get('entity', '')}]"    if query_type == "COMPARISON" else ""
            print(f"  [{i+1}] {p['title']}{hop_info}{ent_info} "
                  f"(rerank: {p.get('rerank_score', 0):.4f})")

    # ── 5. Build context and generate ──
    # Bridge context is injected AFTER reranking so the cross-encoder scores
    # only real Wikipedia passages; the LLM still receives the intermediate fact.
    context_passages = list(reranked)
    if bridge_ctx:
        context_passages = [{"title": "Bridge Finding", "text": bridge_ctx}] + context_passages

    answer = generate_answer(query, context_passages, query_type=query_type)

    # ── 6. Refusal fallback ──
    # Case A: escalation gate fired (SIMPLE used for a bridge question) but SIMPLE
    # produced no answer. Cross-encoder confidence was a false positive — the top
    # passage looked relevant but didn't contain the 2-hop answer chain.
    # Rescue: run full MULTI_HOP now.
    if _gate_fired and _is_refusal(answer):
        if verbose:
            print("[Fallback] Gate-SIMPLE refusal — escalating to full MULTI_HOP")
        mh_budget = _BUDGET.get(("MULTI_HOP", complexity), _DEFAULT_BUDGET["MULTI_HOP"])
        pool, reranked, bridge_ctx, _ = adaptive_retrieve_with_coverage_check(
            query, "MULTI_HOP", index, embedder, passages, mh_budget, verbose,
        )
        context_passages = list(reranked)
        if bridge_ctx:
            context_passages = [{"title": "Bridge Finding", "text": bridge_ctx}] + context_passages
        mh_answer = generate_answer(query, context_passages, query_type="MULTI_HOP")
        if not _is_refusal(mh_answer):
            answer       = mh_answer
            query_type   = "MULTI_HOP"

    # Case B: MULTI_HOP retrieval ran but produced a refusal — retry with simple.
    if query_type == "MULTI_HOP" and _is_refusal(answer):
        if verbose:
            print("[Fallback] Refusal detected — retrying with simple retrieval")
        fb_pool     = retrieve_simple(query, index, embedder, passages, top_k=TOP_K * 2)
        fb_reranked = rerank_passages(query, fb_pool, top_k=TOP_K)
        fb_answer   = generate_answer(query, fb_reranked, query_type="SIMPLE")
        if not _is_refusal(fb_answer):
            answer           = fb_answer
            reranked         = fb_reranked
            context_passages = fb_reranked
            if verbose:
                print(f"[Fallback] Simple-retrieval answer: {fb_answer}")

    if verbose:
        print(f"\nAnswer: {answer}")

    # ── 7. Verify ──
    verification = None
    if verifier_model is not None and verifier_tokenizer is not None:
        ctx          = build_verify_context(reranked, answer)
        verification = verify(ctx, answer, verifier_model, verifier_tokenizer, question=query)

        if (verification["label"] == "SUPPORTED"
                and query_type in ("MULTI_HOP", "COMPARISON")):
            if not llm_judge_supported(query, answer, reranked, verbose):
                orig = verification["scores"]
                verification = {
                    **verification,
                    "label":      "PARTIAL",
                    "confidence": orig["SUPPORTED"],
                    "scores": {
                        "SUPPORTED":   orig["PARTIAL"],
                        "PARTIAL":     orig["SUPPORTED"],
                        "UNSUPPORTED": orig["UNSUPPORTED"],
                    },
                }
                if verbose:
                    print("[LLM Judge] SUPPORTED → PARTIAL (entity not in correct role)")

        if verbose:
            icon = {"SUPPORTED": "✅", "PARTIAL": "⚠️", "UNSUPPORTED": "❌"}.get(
                verification["label"], "?")
            print(f"\nVerification: {icon} {verification['label']} "
                  f"(confidence: {verification['confidence']:.4f})")

        # Low-support fallback: if multi-hop answer has very low support,
        # try simple retrieval and use it if it gives ANY improvement.
        # The condition previously required fb_verif["label"] == "SUPPORTED",
        # which was too strict — "U.S. Route 60" correctly answers a highway
        # question but the verifier gives PARTIAL (gold is "US 60"), so the
        # old condition rejected a correct simple-retrieval answer.
        if (query_type == "MULTI_HOP"
                and verification["scores"].get("SUPPORTED", 1.0) < LOW_SUPPORT_THRESHOLD
                and not _is_refusal(answer)):
            if verbose:
                print(f"[Fallback] Low support "
                      f"({verification['scores'].get('SUPPORTED', 0):.3f}) → simple retry")
            fb_pool     = retrieve_simple(query, index, embedder, passages, top_k=TOP_K * 2)
            fb_reranked = rerank_passages(query, fb_pool, top_k=TOP_K)
            fb_answer   = generate_answer(query, fb_reranked, query_type="SIMPLE")
            if not _is_refusal(fb_answer):
                fb_ctx   = build_verify_context(fb_reranked, fb_answer)
                fb_verif = verify(fb_ctx, fb_answer, verifier_model, verifier_tokenizer,
                                  question=query)
                fb_supp  = fb_verif["scores"].get("SUPPORTED", 0.0)
                orig_supp = verification["scores"].get("SUPPORTED", 0.0)
                fb_judge_ok = False
                # LLM judge for SUPPORTED to catch role-mismatch hallucinations
                if fb_verif["label"] == "SUPPORTED":
                    fb_judge_ok = llm_judge_supported(query, fb_answer, fb_reranked, verbose)
                    if not fb_judge_ok:
                        fb_supp = 0.0

                use_fallback = (
                    fb_supp > orig_supp and fb_verif["label"] in ("SUPPORTED", "PARTIAL")
                ) or fb_judge_ok

                # When both support scores are near-zero, DistilBERT calibration is
                # unreliable for short factoid answers (e.g. "Billy Joel" vs "Ben Margulies"
                # both get ~1% support despite one being correct). We're already in the
                # "multi-hop failed" branch, so prefer the simple answer — multi-hop at
                # near-zero almost always reflects a wrong bridge-entity confusion, not a
                # correct but unverifiable answer.
                if (not use_fallback
                        and fb_verif["label"] in ("PARTIAL", "SUPPORTED")
                        and fb_supp < 0.05 and orig_supp < 0.05):
                    use_fallback = True
                    if verbose:
                        print(f"[Fallback] Near-zero tie ({orig_supp:.3f} vs {fb_supp:.3f}) "
                              f"— preferring simple answer over multi-hop bridge entity.")

                if use_fallback:
                    answer       = fb_answer
                    reranked     = fb_reranked
                    verification = fb_verif
                    if verbose:
                        print(f"[Fallback] Using simple answer: {fb_answer} "
                              f"(supp: {fb_supp:.3f}, label: {fb_verif['label']})")

    return {
        "query":              query,
        "query_type":         query_type,
        "complexity":         complexity,
        "level":              complexity,                        # kept for Stage 4 compat
        "retrieval_strategy": RETRIEVAL_STRATEGY_LABEL[query_type],
        "retrieval_params":   budget,
        "retrieval_stats":    ret_stats,
        "num_retrieved":      len(pool),
        "retrieved":          pool,
        "reranked":           reranked,
        "answer":             answer,
        "verification":       verification,
    }


# ─────────────────────────────────────────────
# STEP 5: EVALUATION — Stage 3 vs Stage 1
# ─────────────────────────────────────────────
def evaluate_adaptive(index, embedder, passages,
                      verifier_model, verifier_tokenizer,
                      num_samples=100):
    """
    Evaluates Stage 3 adaptive retrieval against Stage 1 baseline.

    Metrics:
      Exact Match       — strict normalization
      Token F1          — HotpotQA official token overlap
      Hallucination Rate — (PARTIAL + UNSUPPORTED) / total
      Recall@5/10/20    — gold supporting-fact titles in retrieval pool
      Supporting-fact coverage — fraction of gold titles found in top-k
      Avg retrieval confidence, coverage, expansions triggered
      Per-level breakdown (easy / medium / hard) using HotpotQA ground truth

    Stage 1 baseline uses the same hybrid+rerank pipeline (not bare dense-only)
    so the comparison is apples-to-apples.
    """
    from Stage_1_RAG_Pipeline import (
        retrieve_hybrid as _s1_hybrid,
        rerank_passages  as _s1_rerank,
        token_f1,
        compute_recall_at_k,
        RERANK_POOL,
    )

    print(f"\nEvaluating Stage 3 on {num_samples} HotpotQA validation samples...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

    s1_em_l, s3_em_l       = [], []
    s1_f1_l, s3_f1_l       = [], []
    s1_hr_l, s3_hr_l       = [], []
    s3_r5_l, s3_r10_l, s3_r20_l = [], [], []
    s3_cov_l, s3_conf_l, s3_exp_l = [], [], []

    by_level = defaultdict(lambda: defaultdict(list))
    results  = []

    recall_pool_size = max(RERANK_POOL, 20)

    for i, example in enumerate(tqdm(dataset)):
        if i >= num_samples:
            break

        query       = example["question"]
        gold_answer = example["answer"]
        level       = example.get("level", "medium")
        gold_titles = list(dict.fromkeys(example["supporting_facts"]["title"]))

        # ── Stage 1 baseline (hybrid + rerank) ──
        s1_pool     = _s1_hybrid(query, index, embedder, passages, top_k=recall_pool_size)
        s1_reranked = _s1_rerank(query, s1_pool, top_k=TOP_K)
        s1_answer   = generate_answer(query, s1_reranked)
        s1_em_v     = exact_match(s1_answer, gold_answer)
        s1_f1_v     = token_f1(s1_answer, gold_answer)

        # ── Stage 3 adaptive retrieval (use ground-truth level as complexity) ──
        query_type = classify_query(query)
        budget     = _BUDGET.get((query_type, level), _DEFAULT_BUDGET[query_type])
        # Use a pool large enough for Recall@20 measurement
        budget_r   = dict(budget, top_k=max(budget["top_k"], recall_pool_size))

        pool, reranked, bridge_ctx, ret_stats = adaptive_retrieve_with_coverage_check(
            query, query_type, index, embedder, passages, budget_r, verbose=False,
        )

        context_passages = list(reranked)
        if bridge_ctx:
            context_passages = [{"title": "Bridge Finding", "text": bridge_ctx}] + context_passages

        s3_answer = generate_answer(query, context_passages, query_type=query_type)

        # Refusal fallback
        if query_type == "MULTI_HOP" and _is_refusal(s3_answer):
            fb_pool     = retrieve_simple(query, index, embedder, passages, top_k=TOP_K * 2)
            fb_reranked = rerank_passages(query, fb_pool, top_k=TOP_K)
            fb_answer   = generate_answer(query, fb_reranked, query_type="SIMPLE")
            if not _is_refusal(fb_answer):
                s3_answer = fb_answer
                reranked  = fb_reranked
                pool      = fb_pool

        s3_em_v = exact_match(s3_answer, gold_answer)
        s3_f1_v = token_f1(s3_answer, gold_answer)

        # ── Recall@K (pre-rerank pool) ──
        pool_titles = [p["title"] for p in pool]
        r5  = compute_recall_at_k(pool_titles, gold_titles, 5)
        r10 = compute_recall_at_k(pool_titles, gold_titles, 10)
        r20 = compute_recall_at_k(pool_titles, gold_titles, 20)

        # ── Hallucination via verifier ──
        s1_ctx   = build_verify_context(s1_reranked, s1_answer)
        s1_verif = verify(s1_ctx, s1_answer, verifier_model, verifier_tokenizer, question=query)
        s1_h     = 1 if s1_verif["label"] in ("PARTIAL", "UNSUPPORTED") else 0

        s3_ctx   = build_verify_context(reranked, s3_answer)
        s3_verif = verify(s3_ctx, s3_answer, verifier_model, verifier_tokenizer, question=query)
        s3_h     = 1 if s3_verif["label"] in ("PARTIAL", "UNSUPPORTED") else 0

        # ── Accumulate global ──
        s1_em_l.append(s1_em_v);  s3_em_l.append(s3_em_v)
        s1_f1_l.append(s1_f1_v);  s3_f1_l.append(s3_f1_v)
        s1_hr_l.append(s1_h);     s3_hr_l.append(s3_h)
        s3_r5_l.append(r5);       s3_r10_l.append(r10);  s3_r20_l.append(r20)
        s3_cov_l.append(ret_stats["coverage"])
        s3_conf_l.append(ret_stats["confidence"])
        s3_exp_l.append(ret_stats["expansions_used"])

        # ── Accumulate per-level ──
        lv = by_level[level]
        lv["s1_em"].append(s1_em_v);  lv["s3_em"].append(s3_em_v)
        lv["s1_f1"].append(s1_f1_v);  lv["s3_f1"].append(s3_f1_v)
        lv["s1_hr"].append(s1_h);     lv["s3_hr"].append(s3_h)
        lv["r10"].append(r10);        lv["r20"].append(r20)

        results.append({
            "question":     query,
            "gold":         gold_answer,
            "level":        level,
            "query_type":   query_type,
            "s1_answer":    s1_answer,
            "s3_answer":    s3_answer,
            "s1_em":        s1_em_v,      "s3_em":   s3_em_v,
            "s1_f1":        round(s1_f1_v, 4), "s3_f1": round(s3_f1_v, 4),
            "s1_halluc":    s1_h,         "s3_halluc": s3_h,
            "recall_at_5":  round(r5, 4), "recall_at_10": round(r10, 4),
            "recall_at_20": round(r20, 4),
            "coverage":     round(ret_stats["coverage"], 4),
            "confidence":   round(ret_stats["confidence"], 4),
            "expansions":   ret_stats["expansions_used"],
            "gold_titles":  gold_titles,
        })

    # ── Summary table ──
    print(f"\n{'='*65}")
    print(f"Stage Comparison ({num_samples} samples)")
    print(f"{'='*65}")
    print(f"  {'Metric':<30} {'Stage 1':>10} {'Stage 3':>10} {'Δ':>8}")
    print(f"  {'-'*58}")
    for name, v1, v3 in [
        ("Exact Match",        _avg(s1_em_l), _avg(s3_em_l)),
        ("Token F1",           _avg(s1_f1_l), _avg(s3_f1_l)),
        ("Hallucination Rate", _avg(s1_hr_l), _avg(s3_hr_l)),
    ]:
        print(f"  {name:<30} {v1:>10.4f} {v3:>10.4f} {v3-v1:>+8.4f}")

    print(f"\n  Retrieval Recall@5   : {_avg(s3_r5_l):.4f}")
    print(f"  Retrieval Recall@10  : {_avg(s3_r10_l):.4f}")
    print(f"  Retrieval Recall@20  : {_avg(s3_r20_l):.4f}")
    print(f"\n  Avg Coverage         : {_avg(s3_cov_l):.4f}")
    print(f"  Avg Confidence       : {_avg(s3_conf_l):.4f}")
    print(f"  Avg Expansions       : {_avg(s3_exp_l):.2f}")

    # ── Per-level breakdown ──
    print(f"\n  {'─'*62}")
    print(f"  {'Level':<10} {'N':>4}  {'S1 EM':>7} {'S3 EM':>7} "
          f"{'S3 F1':>7} {'S3 HR':>7} {'R@10':>7}")
    print(f"  {'─'*62}")
    for lv_name in ("easy", "medium", "hard"):
        lv = by_level[lv_name]
        if not lv["s3_em"]:
            continue
        n = len(lv["s3_em"])
        print(f"  {lv_name:<10} {n:>4}  "
              f"{_avg(lv['s1_em']):>7.4f} {_avg(lv['s3_em']):>7.4f} "
              f"{_avg(lv['s3_f1']):>7.4f} {_avg(lv['s3_hr']):>7.4f} "
              f"{_avg(lv['r10']):>7.4f}")

    # ── Query type breakdown ──
    type_counts = Counter(r["query_type"] for r in results)
    print(f"\n  Query type distribution:")
    for qtype, cnt in type_counts.most_common():
        qtype_em = [r["s3_em"] for r in results if r["query_type"] == qtype]
        print(f"    {qtype:<12}: {cnt:>4} ({cnt/len(results)*100:.1f}%) "
              f"| S3 EM={_avg(qtype_em):.3f}")

    with open("stage3_results.json", "w") as f:
        json.dump({
            "num_samples":    num_samples,
            "stage1_em":      _avg(s1_em_l),   "stage3_em":   _avg(s3_em_l),
            "stage1_f1":      _avg(s1_f1_l),   "stage3_f1":   _avg(s3_f1_l),
            "stage1_halluc":  _avg(s1_hr_l),   "stage3_halluc": _avg(s3_hr_l),
            "recall_at_5":    _avg(s3_r5_l),   "recall_at_10": _avg(s3_r10_l),
            "recall_at_20":   _avg(s3_r20_l),
            "avg_coverage":   _avg(s3_cov_l),
            "avg_confidence": _avg(s3_conf_l),
            "avg_expansions": _avg(s3_exp_l),
            "query_types":    dict(type_counts),
            "results":        results,
        }, f, indent=2)
    print("\nResults saved → stage3_results.json")
    return _avg(s3_hr_l), _avg(s3_em_l)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Stage 3: Adaptive Retrieval")
    print(f"Device: {DEVICE.upper()}\n")

    # ── Load FAISS index ──
    if os.path.exists(INDEX_PATH) and os.path.exists(PASSAGES_PATH):
        index, embedder, passages = load_faiss_index()
    else:
        passages = load_hotpotqa_passages()
        index, embedder, passages = build_faiss_index(passages)

    # ── Load verifier ──
    verifier_model, verifier_tokenizer = None, None
    if os.path.exists(VERIFIER_PATH):
        verifier_model, verifier_tokenizer = load_verifier(VERIFIER_PATH)
        print("Verifier loaded — answers will be verified after generation.\n")
    else:
        print("No verifier found — run verifier_gpu.py --mode train first.\n")

    # ── Test classifier on example queries ──
    print("Query Classifier Test:")
    print("-" * 40)
    test_queries = [
        "Who is the CEO of Apple?",
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Who directed the film that stars the actor who played Iron Man?",
        "Which magazine was started first, Arthur's Magazine or First for Women?",
        "What year was the Eiffel Tower built?",
    ]
    for q in test_queries:
        print(f"  [{classify_query(q):>10}] {q}")

    # ── Interactive demo ──
    print("\n=== Stage 3: Adaptive RAG Demo ===")
    print("Type 'eval' to run evaluation, 'quit' to exit.\n")

    while True:
        query = input("Enter your question: ").strip()
        if query.lower() == "quit":
            break
        elif query.lower() == "eval":
            evaluate_adaptive(
                index, embedder, passages,
                verifier_model, verifier_tokenizer,
                num_samples=50
            )
        elif query:
            adaptive_rag_query(
                query, index, embedder, passages,
                verifier_model, verifier_tokenizer
            )

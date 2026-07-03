"""
Stage 3: Adaptive Retrieval Planner
====================================
Project: HARA — Hallucination-Aware Retrieval Agent
Proposal Section: 2.5, 4.3, 6.3.3

Builds on Stage 1 (Stage_1_RAG_Pipeline.py) and Stage 2 (Stage_2_Verifier_GPU.py).

Architecture: official HotpotQA distractor protocol, matching the redesigned
Stage 1/2/5. There is no global FAISS index and no global BM25 — every request
builds a temporary per-question corpus via Stage 1's build_example_corpus(),
retrieves adaptively from just that corpus, and discards it afterward.

Stage 3's contribution is NOT "retrieve more" — it is that different questions
run genuinely different retrieval ALGORITHMS, not just different parameter
values plugged into one generic retrieval function. The Adaptive Retrieval
Planner selects one of nine named retrieval pipelines based on
(question type × difficulty), where difficulty controls which processing
STAGES run (plain hybrid retrieval vs. coverage-gated targeted expansion vs.
full bridge/entity discovery), not merely top_k/max_hops/max_expansions
values fed into a single generic function.

Question types (HotpotQA ground truth when available, else classify_query()):
  SIMPLE     → standard hybrid retrieval, optionally coverage-expanded
  MULTI_HOP  → bridge-aware retrieval, escalating from plain hybrid (easy) to
               cheap bridge-entity detection (medium) to full sub-question
               decomposition (hard)
  COMPARISON → parallel per-entity retrieval, escalating to targeted
               attribute-specific re-querying for the harder tier

After generation and Stage 2 verification, Stage 3 may perform AT MOST ONE
verifier-guided retrieval refinement (escalate to a harder pipeline, retrieve
again, regenerate, re-verify, return) — this is deliberately capped at a
single retry so Stage 3 stays clearly distinct from Stage 4's genuinely
iterative agentic loop.

Usage:
  python Stage_3_Adaptive_Retrieval.py
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict

import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Stage 1 helpers ──
from Stage_1_RAG_Pipeline import (
    build_example_corpus,
    generate_answer,
    rerank_passages,
    retrieve_hybrid,
    retrieve as _retrieve_dense,
    exact_match,
    llm_judge_supported,
    compute_recall_at_k,
    EMBED_MODEL,
    RERANK_POOL,
)

# ── Stage 2 verifier (V2: evidence-grounded self-verification) ──
# Native interface — verify(question, answer, passages, nli_verifier) —
# returns the full structured report (overall_status, overall_confidence,
# support_score, failure_reason, recommended_action, claims, question_intent,
# etc.) directly, no legacy verify_legacy()/build_verify_context() shim.
from Stage_2_Verifier import load_verifier, verify, VERIFIER_PATH

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE      = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available()
               else "cpu")
TOP_K              = 10    # passages sent to LLM (post-rerank) — kept at 10 for Stage 4 compat
TOP_K_MULTI        = 5     # passages per hop / per entity in multi-hop & comparison retrieval
MAX_HOPS           = 3     # maximum iterative hops inside retrieve_multi_hop()
LOW_SUPPORT_THRESHOLD = 0.15  # verifier P(SUPPORTED) below which one refinement retry fires

# ── Coverage / confidence thresholds shared by every pipeline tier ──
RETRIEVAL_CONF_THRESHOLD = 0.45  # cross-encoder confidence below which a tier expands
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
    Only used for questions outside the loaded HotpotQA benchmark — HotpotQA's own
    ground-truth `type` field is always preferred when available (see
    adaptive_rag_query's query_type_override).
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
# the retrieval task is, so the right pipeline tier is chosen.
# ─────────────────────────────────────────────
_QW_SKIP = frozenset({
    "Who", "What", "Where", "When", "Which", "How",
    "Were", "Was", "Is", "Are", "Did", "Do", "Does",
    "The", "Both", "Has", "Had", "This", "That",
})


def estimate_complexity(query: str) -> str:
    """
    Estimate query complexity as 'easy', 'medium', or 'hard'.
    Only used for questions outside the loaded HotpotQA benchmark — HotpotQA's
    own ground-truth `level` field is always preferred when available (see
    adaptive_rag_query's level_override).

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
# STEP 2: RETRIEVAL STRATEGIES (building blocks — reused by every pipeline tier)
# ─────────────────────────────────────────────
def retrieve_simple(query, index, embedder, passages, top_k=TOP_K, bm25=None):
    """
    Hybrid BM25 + dense retrieval scoped to this question's own temporary
    corpus. `bm25` is the per-question BM25Okapi instance build_example_corpus()
    returned for this request — passed through explicitly so there is no
    hidden fallback to a (now nonexistent) global BM25 object.
    """
    return retrieve_hybrid(query, index, embedder, passages, top_k=top_k, bm25=bm25)


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
                                      top_k=TOP_K_MULTI, bm25=None):
    """
    2-sub-question decomposition retrieval — the "Bridge Entity Discovery +
    Sub-question Generation + Hop 1 + Hop 2" mechanism used by the BRIDGE-hard
    pipeline tier.

    Key difference from a naive 2-step chain:
      - SQ2 retrieval is INDEPENDENT of the bridge answer.
        A wrong bridge answer degrades context quality but does NOT
        misdirect retrieval (the old failure mode).
      - The bridge answer is returned as a text string so the caller can
        inject it as a synthetic passage AFTER CrossEncoder reranking.
        This means the CrossEncoder scores real Wikipedia passages, while
        the LLM still gets the intermediate finding as explicit context.

    Falls back to retrieve_multi_hop() (entity-anchored + iterative) when no
    bridge pattern is found in the question — this is the "alternate bridge
    discovery mechanism" for questions decompose_and_retrieve_multi_hop can't
    parse structurally.

    Returns: (passages_list, bridge_context_str | None)
    """
    sq1, sq2 = _decompose_multihop_query(query)

    if sq1 is None:
        # No bridging clause detected — fall back to entity-anchored + iterative retrieval
        return retrieve_multi_hop(query, index, embedder, passages, top_k, bm25=bm25), None

    # ── Sub-question 1 (Hop 1): retrieve and answer the bridge ──
    sq1_passages = retrieve_simple(sq1, index, embedder, passages, top_k=top_k, bm25=bm25)
    bridge_answer = generate_answer(sq1, sq1_passages, query_type="SIMPLE")

    bridge_ctx = (
        f"Intermediate finding — sub-question: '{sq1}' "
        f"→ answer: '{bridge_answer}'. "
        f"Use this intermediate answer to help resolve the main question."
    )

    # ── Sub-question 2 (Hop 2): anchor retrieval on the bridge answer ──
    # Previously sq2 was retrieved independently. Without knowing the bridge entity
    # (e.g., "Adriana Trigiani" for "the director of Big Stone Gap"), dense retrieval
    # for "The director is based in what New York city?" returns generic NYC articles
    # instead of the director's bio page which mentions Greenwich Village.
    # By prepending the bridge answer, retrieval stays focused on the right entity.
    if bridge_answer and not _is_refusal(bridge_answer):
        sq2_query = f"{bridge_answer} {sq2}"
    else:
        sq2_query = sq2
    sq2_passages = retrieve_simple(sq2_query, index, embedder, passages, top_k=top_k, bm25=bm25)

    # Merge: sq2 passages first (directly relevant to final answer), then sq1 context
    seen     = {p["title"] for p in sq2_passages}
    combined = [dict(p, hop=2) for p in sq2_passages]
    for p in sq1_passages:
        if p["title"] not in seen:
            combined.append(dict(p, hop=1))
            seen.add(p["title"])

    return combined, bridge_ctx


def _extract_question_entities(query: str) -> list:
    # Extract at most 2 anchor entities for entity-anchored parallel retrieval.
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
                       top_k=TOP_K_MULTI, max_hops=MAX_HOPS, bm25=None):
    """
    Multi-hop retrieval combining entity-anchored parallel search with iterative
    reformulation. Used as the fallback bridge-discovery mechanism when
    decompose_and_retrieve_multi_hop() can't parse a structural bridge pattern
    out of the question.

    Two phases:

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
            entity, index, embedder, passages, top_k=top_k, bm25=bm25,
        )
        for p in entity_results:
            if p["title"] not in seen_titles:
                seen_titles.add(p["title"])
                new_p = dict(p, hop=1)
                all_retrieved.append(new_p)

    # Full-question search (covers questions with no extractable named entities)
    full_results = retrieve_simple(query, index, embedder, passages, top_k=top_k, bm25=bm25)
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

        hop_results  = retrieve_simple(current_query, index, embedder, passages, top_k, bm25=bm25)
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


# Comparison-attribute keyword map — module level so both retrieve_comparison()
# and the COMPARISON-hard pipeline's targeted attribute expansion can reuse it
# without duplicating the mapping.
_COMPARISON_ATTRIBUTE_MAP = {
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


def retrieve_comparison(query, index, embedder, passages, top_k=TOP_K_MULTI):
    """
    Parallel retrieval for comparison questions.
    Extracts the two entities being compared and retrieves
    passages for each independently, then combines results.

    Deliberately dense-only, not hybrid: BM25 over-weights generic
    "nationality/born/age" keyword documents instead of the specific entity's
    Wikipedia bio page, which dense embedding finds correctly. This is an
    intentional, preserved design choice — not something the BM25-threading
    fix should touch.

    Used for COMPARISON queries like:
    'Were Scott Derrickson and Ed Wood of the same nationality?'
    """
    entities = _extract_entities(query)

    attribute_suffix = ""
    for pattern, suffix in _COMPARISON_ATTRIBUTE_MAP.items():
        if re.search(pattern, query, re.IGNORECASE):
            attribute_suffix = suffix
            break

    all_retrieved = []
    seen_titles   = set()

    if len(entities) >= 2:
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
    Build a reformulated query that stays semantically anchored to the
    original question but targets whatever entity the current best passage
    surfaced. Used both by retrieve_multi_hop()'s Phase 2 iterative hops and
    by the SIMPLE-hard pipeline's "Retrieve Again" step.

    Problem with entity-only reformulation: "Robert Downey Jr." as a follow-up
    query retrieves RDJ's biography page — not the passage that says who
    DIRECTED the film starring RDJ. The anchor (predicate/role/spatial) must
    come from the ORIGINAL question, not from the retrieved passage.

    Priority order for anchor selection:
      1. Verbal predicate  (directed, wrote, founded, nationality, …)
      2. Role noun         (director of, author of, capital of, …)
      3. Spatial relation  (block away, next to, capital, headquarters, …)
      4. bare bridge entity (last resort — better than nothing)

    Returns a natural-language follow-up query of the form "BridgeEntity anchor".
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
# Run AFTER reranking, BEFORE generation, to decide whether a pipeline
# tier's coverage-gated expansion step should fire.
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
      score -5  → confidence ≈ 0.08  (very low — expand)

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
      missing   — entities not found (used to seed targeted follow-up queries —
                  interpreted as "missing bridge entity" for MULTI_HOP pipelines,
                  "missing comparison entity" for COMPARISON pipelines, and
                  "missing fact" for SIMPLE pipelines)
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
# LEGACY COMPATIBILITY LAYER — used only by Stage_4_Agentic_Loop.py
# ─────────────────────────────────────────────
# Stage_4_Agentic_Loop.py imports _BUDGET, _DEFAULT_BUDGET,
# adaptive_retrieve_with_coverage_check, _retrieve_for_type, _RETRIEVAL_PARAMS,
# and _DEFAULT_PARAMS directly and calls several of them with real control
# flow (not just dead imports — confirmed by inspection). Stage 4 is
# explicitly out of scope for this redesign ("Stage 4 will remain the only
# fully agentic iterative retrieval stage"), so these are preserved verbatim,
# unchanged from the pre-redesign implementation, purely for Stage 4's benefit.
# Stage 3's own adaptive_rag_query() below does NOT use any of this — it uses
# the new pipeline-planner system (_PIPELINES/_select_pipeline) instead.
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
    """[LEGACY — Stage 4 only.] Dispatch to the right retrieval function and
    return (pool, bridge_ctx)."""
    if query_type == "SIMPLE":
        return retrieve_simple(query, index, embedder, passages, top_k=top_k), None

    if query_type == "MULTI_HOP":
        pool, bridge_ctx = decompose_and_retrieve_multi_hop(
            query, index, embedder, passages, top_k=top_k,
        )
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
    """[LEGACY — Stage 4 only.] Pre-redesign adaptive retrieval loop: retrieve
    → rerank → check confidence & coverage → expand if insufficient → repeat
    up to budget['max_expansions'] times."""
    top_k          = budget["top_k"]
    max_hops       = budget["max_hops"]
    max_expansions = budget["max_expansions"]

    pool, bridge_ctx = _retrieve_for_type(
        query, query_type, index, embedder, passages, top_k, max_hops,
    )
    reranked   = rerank_passages(query, pool, top_k=TOP_K)
    confidence = estimate_retrieval_confidence(reranked)
    cov_info   = check_evidence_coverage(query, reranked)
    expansions = 0

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


# ─────────────────────────────────────────────
# STEP 4: ADAPTIVE RETRIEVAL PLANNER — PIPELINE DEFINITIONS
# ─────────────────────────────────────────────
# Every pipeline function shares the signature:
#   (query, index, embedder, passages, bm25) -> (pool, reranked, bridge_ctx, stats)
# stats = {"coverage": float, "confidence": float, "expanded": bool}
#
# Difficulty controls WHICH STAGES RUN, not just numeric parameters:
#   easy   pipelines never call check_evidence_coverage() at all
#   medium pipelines check coverage and do ONE targeted expansion step
#   hard   pipelines check coverage and do the full type-specific discovery
#          mechanism (bridge decomposition / attribute-targeted comparison
#          retrieval / targeted-expansion-plus-reformulated-retry)
# ─────────────────────────────────────────────

def _rerank_and_stats(query, pool):
    """Shared post-retrieval step: rerank, then compute confidence + coverage
    on the reranked result — reused by every pipeline tier."""
    reranked   = rerank_passages(query, pool, top_k=TOP_K)
    confidence = estimate_retrieval_confidence(reranked)
    cov_info   = check_evidence_coverage(query, reranked)
    return reranked, confidence, cov_info


def _merge_pools(pool_a: list, pool_b: list) -> list:
    """Title-deduplicated merge of two passage pools, preserving pool_a's order first."""
    seen   = {p["title"] for p in pool_a}
    merged = list(pool_a)
    for p in pool_b:
        if p["title"] not in seen:
            merged.append(p)
            seen.add(p["title"])
    return merged


def _targeted_expand(query, pool, missing_entities, index, embedder, passages, bm25,
                      top_k=TOP_K_MULTI):
    """
    Generic targeted expansion, reused by SIMPLE and MULTI_HOP-medium pipelines:
    re-query specifically for each missing entity rather than blindly widening
    the pool, since a per-question corpus is already small and a targeted
    re-query is more likely to change the reranked ordering than a blanket
    re-run of the same query would.
    """
    if missing_entities:
        for me in missing_entities[:2]:
            extra = retrieve_simple(f"{me} {query}", index, embedder, passages,
                                     top_k=top_k, bm25=bm25)
            pool = _merge_pools(pool, extra)
    else:
        extra = retrieve_simple(query, index, embedder, passages,
                                 top_k=top_k * 2, bm25=bm25)
        pool = _merge_pools(pool, extra)
    return pool


def _expand_missing_comparison_entity(query, pool, missing_entities, index, embedder, passages):
    """
    COMPARISON-specific expansion: re-query specifically for whichever compared
    entity's evidence is missing. Dense-only, matching retrieve_comparison()'s
    own design rationale (BM25 over-weights generic attribute keywords instead
    of finding the entity's bio page).
    """
    for me in missing_entities[:2]:
        extra = _retrieve_dense(me, index, embedder, passages, TOP_K_MULTI)
        pool  = _merge_pools(pool, extra)
    return pool


def _targeted_attribute_expand(query, pool, missing_entities, index, embedder, passages):
    """
    COMPARISON-hard only: re-query the missing entity together with the
    specific comparison attribute this question is asking about (age,
    nationality, awards, …), reusing retrieve_comparison()'s own attribute map
    instead of duplicating it.
    """
    attribute_suffix = ""
    for pattern, suffix in _COMPARISON_ATTRIBUTE_MAP.items():
        if re.search(pattern, query, re.IGNORECASE):
            attribute_suffix = suffix
            break
    if not attribute_suffix:
        return pool
    for me in missing_entities[:2]:
        extra = _retrieve_dense(f"{me} {attribute_suffix}", index, embedder, passages, TOP_K_MULTI)
        pool  = _merge_pools(pool, extra)
    return pool


def _needs_expansion(confidence: float, cov_info: dict) -> bool:
    return confidence < RETRIEVAL_CONF_THRESHOLD or cov_info["coverage"] < COVERAGE_THRESHOLD


# ── SIMPLE pipelines ──────────────────────────────────────────────────────
def _pipeline_simple_easy(query, index, embedder, passages, bm25):
    """Hybrid Retrieval → Cross Encoder → Generate. No coverage check at all."""
    pool = retrieve_simple(query, index, embedder, passages, top_k=RERANK_POOL, bm25=bm25)
    reranked, confidence, cov = _rerank_and_stats(query, pool)
    return pool, reranked, None, {
        "coverage": cov["coverage"], "coverage_before": cov["coverage"],
        "confidence": confidence, "expanded": False,
    }


def _pipeline_simple_medium(query, index, embedder, passages, bm25):
    """Hybrid Retrieval → Coverage Check → Targeted Expansion → Cross Encoder → Generate."""
    pool = retrieve_simple(query, index, embedder, passages, top_k=RERANK_POOL, bm25=bm25)
    reranked, confidence, cov = _rerank_and_stats(query, pool)
    coverage_before = cov["coverage"]
    expanded = False
    if _needs_expansion(confidence, cov):
        pool = _targeted_expand(query, pool, cov["missing"], index, embedder, passages, bm25)
        reranked, confidence, cov = _rerank_and_stats(query, pool)
        expanded = True
    return pool, reranked, None, {
        "coverage": cov["coverage"], "coverage_before": coverage_before,
        "confidence": confidence, "expanded": expanded,
    }


def _pipeline_simple_hard(query, index, embedder, passages, bm25):
    """Hybrid Retrieval → Coverage Check → Targeted Expansion → Retrieve Again → Merge → Cross Encoder → Generate.

    "Retrieve Again" reformulates from the current best passage (reusing
    _reformulate_query, the same mechanism retrieve_multi_hop's Phase 2 uses)
    rather than re-running the identical query — the per-question corpus is
    small and largely already covered by one hybrid pass, so a differently
    worded query is what can actually change which passages rank highest,
    not a second identical search.
    """
    pool = retrieve_simple(query, index, embedder, passages, top_k=RERANK_POOL, bm25=bm25)
    reranked, confidence, cov = _rerank_and_stats(query, pool)
    coverage_before = cov["coverage"]
    expanded = False
    if _needs_expansion(confidence, cov):
        pool = _targeted_expand(query, pool, cov["missing"], index, embedder, passages, bm25)
        if pool:
            reformulated = _reformulate_query(query, pool[0]["text"][:300])
            if reformulated.strip().lower() != query.strip().lower():
                extra = retrieve_simple(reformulated, index, embedder, passages,
                                         top_k=RERANK_POOL, bm25=bm25)
                pool = _merge_pools(pool, extra)
        reranked, confidence, cov = _rerank_and_stats(query, pool)
        expanded = True
    return pool, reranked, None, {
        "coverage": cov["coverage"], "coverage_before": coverage_before,
        "confidence": confidence, "expanded": expanded,
    }


# ── MULTI_HOP (bridge) pipelines ──────────────────────────────────────────
def _pipeline_bridge_easy(query, index, embedder, passages, bm25):
    """Hybrid Retrieval → Cross Encoder → Generate. Deliberately NO bridge
    decomposition — most HotpotQA bridge questions this easy are already
    answerable from a single hybrid search, so paying for an extra LLM call
    to decompose the question would be pure overhead."""
    pool = retrieve_simple(query, index, embedder, passages, top_k=RERANK_POOL, bm25=bm25)
    reranked, confidence, cov = _rerank_and_stats(query, pool)
    return pool, reranked, None, {
        "coverage": cov["coverage"], "coverage_before": cov["coverage"],
        "confidence": confidence, "expanded": False,
    }


def _pipeline_bridge_medium(query, index, embedder, passages, bm25):
    """Hybrid Retrieval → Coverage Check → Bridge Entity Detection → Retrieve
    Missing Bridge → Merge → Cross Encoder → Generate.

    "Bridge Entity Detection" is a cheap, LLM-free pattern match
    (_decompose_multihop_query) — if it finds a sub-question, we retrieve for
    it directly. Unlike BRIDGE-hard, we do NOT call generate_answer() to
    produce an intermediate bridge answer here — that extra LLM cost is
    reserved for the hard tier only.
    """
    pool = retrieve_simple(query, index, embedder, passages, top_k=RERANK_POOL, bm25=bm25)
    reranked, confidence, cov = _rerank_and_stats(query, pool)
    coverage_before = cov["coverage"]
    expanded = False
    if _needs_expansion(confidence, cov):
        sq1, _sq2 = _decompose_multihop_query(query)
        if sq1:
            bridge_pool = retrieve_simple(sq1, index, embedder, passages, top_k=TOP_K_MULTI, bm25=bm25)
            pool = _merge_pools(pool, bridge_pool)
        else:
            pool = _targeted_expand(query, pool, cov["missing"], index, embedder, passages, bm25)
        reranked, confidence, cov = _rerank_and_stats(query, pool)
        expanded = True
    return pool, reranked, None, {
        "coverage": cov["coverage"], "coverage_before": coverage_before,
        "confidence": confidence, "expanded": expanded,
    }


def _pipeline_bridge_hard(query, index, embedder, passages, bm25):
    """Hybrid Retrieval → Coverage Check → Bridge Entity Discovery →
    Sub-question Generation → Retrieve Hop 1 → Retrieve Hop 2 → Merge →
    Cross Encoder → Generate.

    Reuses decompose_and_retrieve_multi_hop() in full — bridge entity
    discovery + sub-question generation + hop 1 + hop 2 retrieval are exactly
    what that function already does. If it can't parse a structural bridge
    pattern, it internally falls back to retrieve_multi_hop() (entity-anchored
    + iterative reformulation) as the alternate discovery mechanism — no new
    code needed for that fallback.
    """
    pool = retrieve_simple(query, index, embedder, passages, top_k=RERANK_POOL, bm25=bm25)
    reranked, confidence, cov = _rerank_and_stats(query, pool)
    coverage_before = cov["coverage"]
    bridge_ctx = None
    expanded = False
    if _needs_expansion(confidence, cov):
        decomposed_pool, bridge_ctx = decompose_and_retrieve_multi_hop(
            query, index, embedder, passages, top_k=TOP_K_MULTI, bm25=bm25,
        )
        pool = _merge_pools(pool, decomposed_pool)
        reranked, confidence, cov = _rerank_and_stats(query, pool)
        expanded = True
    return pool, reranked, bridge_ctx, {
        "coverage": cov["coverage"], "coverage_before": coverage_before,
        "confidence": confidence, "expanded": expanded,
    }


# ── COMPARISON pipelines ──────────────────────────────────────────────────
def _pipeline_comparison_easy(query, index, embedder, passages, bm25):
    """Retrieve Entity A / Retrieve Entity B → Merge → Cross Encoder → Generate.
    retrieve_comparison() already does exactly this internally."""
    pool = retrieve_comparison(query, index, embedder, passages, top_k=TOP_K_MULTI)
    reranked, confidence, cov = _rerank_and_stats(query, pool)
    return pool, reranked, None, {
        "coverage": cov["coverage"], "coverage_before": cov["coverage"],
        "confidence": confidence, "expanded": False,
    }


def _pipeline_comparison_medium(query, index, embedder, passages, bm25):
    """Retrieve Entity A/B → Coverage Check → Expand Missing Entity → Merge →
    Cross Encoder → Generate."""
    pool = retrieve_comparison(query, index, embedder, passages, top_k=TOP_K_MULTI)
    reranked, confidence, cov = _rerank_and_stats(query, pool)
    coverage_before = cov["coverage"]
    expanded = False
    if _needs_expansion(confidence, cov):
        pool = _expand_missing_comparison_entity(query, pool, cov["missing"], index, embedder, passages)
        reranked, confidence, cov = _rerank_and_stats(query, pool)
        expanded = True
    return pool, reranked, None, {
        "coverage": cov["coverage"], "coverage_before": coverage_before,
        "confidence": confidence, "expanded": expanded,
    }


def _pipeline_comparison_hard(query, index, embedder, passages, bm25):
    """Retrieve Entity A/B → Coverage Check → Targeted Attribute Retrieval →
    Expand Missing Entity → Merge → Cross Encoder → Generate.

    "Targeted Attribute Retrieval" is the one genuinely new piece of logic in
    this redesign: re-query the under-covered entity together with the
    specific attribute the question is asking about (age, nationality,
    awards, …) instead of a generic re-query — reusing
    retrieve_comparison()'s own attribute keyword map.
    """
    pool = retrieve_comparison(query, index, embedder, passages, top_k=TOP_K_MULTI)
    reranked, confidence, cov = _rerank_and_stats(query, pool)
    coverage_before = cov["coverage"]
    expanded = False
    if _needs_expansion(confidence, cov):
        pool = _targeted_attribute_expand(query, pool, cov["missing"], index, embedder, passages)
        pool = _expand_missing_comparison_entity(query, pool, cov["missing"], index, embedder, passages)
        reranked, confidence, cov = _rerank_and_stats(query, pool)
        expanded = True
    return pool, reranked, None, {
        "coverage": cov["coverage"], "coverage_before": coverage_before,
        "confidence": confidence, "expanded": expanded,
    }


def _pipeline_wide_fallback(query, index, embedder, passages, bm25):
    """Last-resort pipeline used only by the verifier-guided refinement retry
    when the original answer already used the hardest tier for its type —
    a wide plain hybrid re-query, matching the style of the old low-support
    fallback."""
    pool = retrieve_simple(query, index, embedder, passages, top_k=RERANK_POOL * 2, bm25=bm25)
    reranked, confidence, cov = _rerank_and_stats(query, pool)
    return pool, reranked, None, {
        "coverage": cov["coverage"], "coverage_before": cov["coverage"],
        "confidence": confidence, "expanded": True,
    }


_PIPELINES = {
    ("SIMPLE",     "easy"):   _pipeline_simple_easy,
    ("SIMPLE",     "medium"): _pipeline_simple_medium,
    ("SIMPLE",     "hard"):   _pipeline_simple_hard,
    ("MULTI_HOP",  "easy"):   _pipeline_bridge_easy,
    ("MULTI_HOP",  "medium"): _pipeline_bridge_medium,
    ("MULTI_HOP",  "hard"):   _pipeline_bridge_hard,
    ("COMPARISON", "easy"):   _pipeline_comparison_easy,
    ("COMPARISON", "medium"): _pipeline_comparison_medium,
    ("COMPARISON", "hard"):   _pipeline_comparison_hard,
}
_DEFAULT_PIPELINE = {
    "SIMPLE":     _pipeline_simple_medium,
    "MULTI_HOP":  _pipeline_bridge_medium,
    "COMPARISON": _pipeline_comparison_medium,
}


def _select_pipeline(query_type: str, level: str):
    """The Adaptive Retrieval Planner: selects an entire retrieval pipeline
    (a callable composing specific stages) based on (type, difficulty) —
    not a set of parameters fed into one generic function."""
    return _PIPELINES.get((query_type, level), _DEFAULT_PIPELINE.get(query_type, _pipeline_simple_medium))


def _escalate_pipeline(query_type: str, level: str):
    """Used only by the single verifier-guided refinement retry: escalate to
    the hardest pipeline tier for this type, or a wide generic fallback if
    already at the hardest tier."""
    if level != "hard":
        return _PIPELINES.get((query_type, "hard"), _DEFAULT_PIPELINE.get(query_type, _pipeline_simple_medium))
    return _pipeline_wide_fallback


def _pipeline_label(pipeline_fn) -> str:
    return pipeline_fn.__name__.replace("_pipeline_", "").upper()


def _verify_with_role_check(query, answer, reranked, query_type,
                             verifier_model, verifier_tokenizer, verbose):
    """
    Runs Stage 2 verification, then for MULTI_HOP/COMPARISON demotes a
    SUPPORTED status to PARTIAL if the LLM judge finds the answer's entity is
    not in the correct semantic role (the NLI verifier checks entailment
    against evidence but cannot verify the entity fills the right ROLE in a
    multi-entity relationship).

    Shared by both the initial verification and the one-time refinement
    retry's verification, so a refined answer is held to exactly the same
    scrutiny as the original — otherwise refinement could let a role-mismatch
    hallucination through a check the original answer was subject to.
    """
    verification = verify(query, answer, reranked, verifier_model)

    if verification["overall_status"] == "SUPPORTED" and query_type in ("MULTI_HOP", "COMPARISON"):
        if not llm_judge_supported(query, answer, reranked, verbose):
            verification = {
                **verification,
                "overall_status": "PARTIAL",
                "overall_confidence": verification["support_score"],
                "failure_reason": verification["failure_reason"] or "WRONG_ENTITY",
                "recommended_action": "COMPARE" if query_type == "COMPARISON" else "REWRITE",
            }
            if verbose:
                print("[LLM Judge] SUPPORTED → PARTIAL (entity not in correct role)")

    return verification


def adaptive_rag_query(query, index, embedder, passages,
                       verifier_model=None, verifier_tokenizer=None,
                       verbose=True, query_type_override=None, level_override=None,
                       bm25=None):
    """
    Full Stage 3 pipeline — Adaptive Retrieval Planner.

      Question
        → question analysis (type: SIMPLE/MULTI_HOP/COMPARISON, level: easy/medium/hard)
          — HotpotQA ground truth (query_type_override/level_override) is used
            whenever available; classify_query()/estimate_complexity() are only
            a fallback for questions outside the loaded benchmark.
        → planner selects ONE of nine named retrieval pipelines (type × level)
        → retrieve → rerank → generate
        → Stage 2 verify (+ LLM judge role-mismatch check for MULTI_HOP/COMPARISON)
        → AT MOST ONE verifier-guided retrieval refinement if the answer was
          refused, UNSUPPORTED, or low-confidence — escalate to a harder
          pipeline, retrieve again, regenerate, re-verify, keep whichever is
          better. Never chained further than this single retry — that boundary
          is what keeps Stage 3 distinct from Stage 4's genuinely iterative
          agentic loop.

    `bm25` is the per-question BM25Okapi instance build_example_corpus()
    returned for this request; every retrieval path below threads it through
    explicitly rather than falling back to any global BM25 state.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

    # ── 1. Question analysis: type ──
    if query_type_override == "bridge":
        query_type = "MULTI_HOP"
    elif query_type_override == "comparison":
        query_type = "COMPARISON"
    elif query_type_override in ("SIMPLE", "MULTI_HOP", "COMPARISON"):
        query_type = query_type_override
    else:
        query_type = classify_query(query)

    # ── 2. Question analysis: difficulty ──
    level = level_override if level_override else estimate_complexity(query)

    # ── 3. Adaptive Retrieval Planner: select the pipeline ──
    pipeline_fn = _select_pipeline(query_type, level)

    if verbose:
        print(f"Type: {query_type} | Level: {level.upper()} | Pipeline: {_pipeline_label(pipeline_fn)}")

    # ── 4. Retrieve, rerank, coverage-check / expand (inside the pipeline) ──
    pool, reranked, bridge_ctx, stats = pipeline_fn(query, index, embedder, passages, bm25)

    if verbose:
        print(f"\nRetrieved {len(pool)} → reranked to {len(reranked)} "
              f"| conf={stats['confidence']:.3f} cov={stats['coverage']:.2f} "
              f"expanded={stats['expanded']}")
        for i, p in enumerate(reranked[:5]):
            print(f"  [{i+1}] {p['title']} (rerank: {p.get('rerank_score', 0):.4f})")

    # ── 5. Generate ──
    context_passages = list(reranked)
    if bridge_ctx:
        context_passages = [{"title": "Bridge Finding", "text": bridge_ctx}] + context_passages
    answer = generate_answer(query, context_passages, query_type=query_type)

    if verbose:
        print(f"\nAnswer: {answer}")

    # ── 6. Verify ──
    verification     = None
    refinement_used   = False
    if verifier_model is not None and verifier_tokenizer is not None:
        verification = _verify_with_role_check(
            query, answer, reranked, query_type,
            verifier_model, verifier_tokenizer, verbose,
        )

        if verbose:
            icon = {"SUPPORTED": "✅", "PARTIAL": "⚠️", "UNSUPPORTED": "❌"}.get(
                verification["overall_status"], "?")
            print(f"\nVerification: {icon} {verification['overall_status']} "
                  f"(confidence: {verification['overall_confidence']:.4f}, "
                  f"reason: {verification.get('failure_reason')})")

        # ── 7. At most ONE verifier-guided retrieval refinement ──
        needs_refinement = (
            _is_refusal(answer)
            or verification["overall_status"] == "UNSUPPORTED"
            or verification["support_score"] < LOW_SUPPORT_THRESHOLD
        )

        if needs_refinement:
            if verbose:
                print(f"[Refinement] Escalating retrieval pipeline (one retry only)")
            escalated_fn = _escalate_pipeline(query_type, level)
            r_pool, r_reranked, r_bridge_ctx, r_stats = escalated_fn(query, index, embedder, passages, bm25)

            r_context = list(r_reranked)
            if r_bridge_ctx:
                r_context = [{"title": "Bridge Finding", "text": r_bridge_ctx}] + r_context
            r_answer = generate_answer(query, r_context, query_type=query_type)

            if not _is_refusal(r_answer):
                r_verif  = _verify_with_role_check(
                    query, r_answer, r_reranked, query_type,
                    verifier_model, verifier_tokenizer, verbose,
                )
                r_supp    = r_verif["support_score"]
                orig_supp = verification["support_score"]

                improved = (
                    r_supp > orig_supp
                    or (verification["overall_status"] == "UNSUPPORTED" and r_verif["overall_status"] != "UNSUPPORTED")
                )
                if improved:
                    answer          = r_answer
                    pool            = r_pool
                    reranked        = r_reranked
                    verification    = r_verif
                    stats           = r_stats
                    pipeline_fn     = escalated_fn
                    refinement_used = True
                    if verbose:
                        print(f"[Refinement] Using refined answer: {r_answer} "
                              f"(supp: {r_supp:.3f}, status: {r_verif['overall_status']})")

    return {
        "query":              query,
        "query_type":         query_type,
        "complexity":         level,
        "level":              level,                        # kept for Stage 4 compat
        "retrieval_strategy": _pipeline_label(pipeline_fn),
        "retrieval_params":   stats,
        "retrieval_stats":    stats,
        "num_retrieved":      len(pool),
        "retrieved":          pool,
        "reranked":           reranked,
        "answer":             answer,
        "verification":       verification,
        "refinement_used":    refinement_used,
    }


# ─────────────────────────────────────────────
# STEP 5: EVALUATION — Stage 3 vs Stage 1
# ─────────────────────────────────────────────
def evaluate_adaptive(embedder, verifier_model, verifier_tokenizer, num_samples=100):
    """
    Evaluates Stage 3's adaptive retrieval planner against the Stage 1
    baseline, under the official distractor protocol: EVERY validation
    question builds its own temporary corpus via build_example_corpus(),
    used identically by both the Stage 1 baseline and Stage 3 (so the
    comparison is apples-to-apples on the exact same evidence pool), then
    discarded before the next question.

    Evaluates the REAL production pipeline — calls adaptive_rag_query()
    directly rather than a lower-level helper, so these numbers reflect
    exactly what Stage 5's /chat endpoint does. Uses HotpotQA's own
    ground-truth type/level (never estimated) for every question, per the
    project's evaluation methodology.

    Metrics:
      Exact Match, Token F1, Hallucination Rate (PARTIAL+UNSUPPORTED / total)
      Recall@5/10/20 (gold supporting-fact titles in the retrieval pool)
      Avg coverage / confidence / expansion + refinement rate
      Per-level breakdown (easy / medium / hard)
    """
    from Stage_1_RAG_Pipeline import (
        build_example_corpus as _build_corpus,
        token_f1,
    )

    print(f"\nEvaluating Stage 3 on {num_samples} HotpotQA validation samples...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

    s1_em_l, s3_em_l       = [], []
    s1_f1_l, s3_f1_l       = [], []
    s1_hr_l, s3_hr_l       = [], []
    s3_r5_l, s3_r10_l, s3_r20_l = [], [], []
    s3_cov_l, s3_conf_l, s3_refine_l = [], [], []

    by_level = defaultdict(lambda: defaultdict(list))
    results  = []
    skipped  = 0

    for i, example in enumerate(tqdm(dataset)):
        if i >= num_samples:
            break

        query       = example["question"]
        gold_answer = example["answer"]
        level       = example.get("level", "medium")
        qtype       = example.get("type", "bridge")
        gold_titles = list(dict.fromkeys(example["supporting_facts"]["title"]))

        # ── Fresh per-question corpus, shared by both Stage 1 baseline and Stage 3 ──
        ex_index, ex_passages, ex_bm25 = _build_corpus(example, embedder)
        if ex_index is None:
            skipped += 1
            continue

        recall_pool = min(max(RERANK_POOL, 20), len(ex_passages))

        # ── Stage 1 baseline (hybrid + rerank), same corpus Stage 3 will use ──
        s1_pool     = retrieve_hybrid(query, ex_index, embedder, ex_passages,
                                       top_k=recall_pool, bm25=ex_bm25)
        s1_reranked = rerank_passages(query, s1_pool, top_k=TOP_K)
        s1_answer   = generate_answer(query, s1_reranked)
        s1_em_v     = exact_match(s1_answer, gold_answer)
        s1_f1_v     = token_f1(s1_answer, gold_answer)

        # ── Stage 3 — real production pipeline, ground-truth type/level ──
        s3_result = adaptive_rag_query(
            query, ex_index, embedder, ex_passages,
            verifier_model, verifier_tokenizer, verbose=False,
            query_type_override=qtype, level_override=level,
            bm25=ex_bm25,
        )
        s3_answer  = s3_result["answer"]
        s3_pool    = s3_result["retrieved"]
        s3_em_v    = exact_match(s3_answer, gold_answer)
        s3_f1_v    = token_f1(s3_answer, gold_answer)

        # ── Recall@K (pre-rerank pool) ──
        pool_titles = [p["title"] for p in s3_pool]
        r5  = compute_recall_at_k(pool_titles, gold_titles, 5)
        r10 = compute_recall_at_k(pool_titles, gold_titles, 10)
        r20 = compute_recall_at_k(pool_titles, gold_titles, 20)

        # ── Hallucination via verifier ──
        s1_verif = verify(query, s1_answer, s1_reranked, verifier_model)
        s1_h     = 1 if s1_verif["overall_status"] in ("PARTIAL", "UNSUPPORTED") else 0

        s3_verif = s3_result["verification"] or {"overall_status": "UNSUPPORTED"}
        s3_h     = 1 if s3_verif["overall_status"] in ("PARTIAL", "UNSUPPORTED") else 0

        s1_em_l.append(s1_em_v);  s3_em_l.append(s3_em_v)
        s1_f1_l.append(s1_f1_v);  s3_f1_l.append(s3_f1_v)
        s1_hr_l.append(s1_h);     s3_hr_l.append(s3_h)
        s3_r5_l.append(r5);       s3_r10_l.append(r10);  s3_r20_l.append(r20)
        s3_cov_l.append(s3_result["retrieval_stats"]["coverage"])
        s3_conf_l.append(s3_result["retrieval_stats"]["confidence"])
        s3_refine_l.append(1 if s3_result["refinement_used"] else 0)

        lv = by_level[level]
        lv["s1_em"].append(s1_em_v);  lv["s3_em"].append(s3_em_v)
        lv["s1_f1"].append(s1_f1_v);  lv["s3_f1"].append(s3_f1_v)
        lv["s1_hr"].append(s1_h);     lv["s3_hr"].append(s3_h)
        lv["r10"].append(r10);        lv["r20"].append(r20)

        results.append({
            "question":     query,
            "gold":         gold_answer,
            "level":        level,
            "query_type":   s3_result["query_type"],
            "pipeline":     s3_result["retrieval_strategy"],
            "s1_answer":    s1_answer,
            "s3_answer":    s3_answer,
            "s1_em":        s1_em_v,      "s3_em":   s3_em_v,
            "s1_f1":        round(s1_f1_v, 4), "s3_f1": round(s3_f1_v, 4),
            "s1_halluc":    s1_h,         "s3_halluc": s3_h,
            "recall_at_5":  round(r5, 4), "recall_at_10": round(r10, 4),
            "recall_at_20": round(r20, 4),
            "coverage":     round(s3_result["retrieval_stats"]["coverage"], 4),
            "confidence":   round(s3_result["retrieval_stats"]["confidence"], 4),
            "refinement_used": s3_result["refinement_used"],
            "gold_titles":  gold_titles,
        })

    print(f"\n{'='*65}")
    print(f"Stage Comparison ({len(results)} samples{f', {skipped} skipped' if skipped else ''})")
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
    print(f"  Refinement Rate      : {_avg(s3_refine_l):.2%}")

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

    pipeline_counts = Counter(r["pipeline"] for r in results)
    print(f"\n  Pipeline distribution:")
    for pname, cnt in pipeline_counts.most_common():
        p_em = [r["s3_em"] for r in results if r["pipeline"] == pname]
        print(f"    {pname:<20}: {cnt:>4} ({cnt/len(results)*100:.1f}%) "
              f"| S3 EM={_avg(p_em):.3f}")

    with open("stage3_results.json", "w") as f:
        json.dump({
            "num_samples":    len(results),
            "stage1_em":      _avg(s1_em_l),   "stage3_em":   _avg(s3_em_l),
            "stage1_f1":      _avg(s1_f1_l),   "stage3_f1":   _avg(s3_f1_l),
            "stage1_halluc":  _avg(s1_hr_l),   "stage3_halluc": _avg(s3_hr_l),
            "recall_at_5":    _avg(s3_r5_l),   "recall_at_10": _avg(s3_r10_l),
            "recall_at_20":   _avg(s3_r20_l),
            "avg_coverage":   _avg(s3_cov_l),
            "avg_confidence": _avg(s3_conf_l),
            "refinement_rate": _avg(s3_refine_l),
            "pipelines":      dict(pipeline_counts),
            "results":        results,
        }, f, indent=2)
    print("\nResults saved → stage3_results.json")
    return _avg(s3_hr_l), _avg(s3_em_l)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    print("Stage 3: Adaptive Retrieval Planner")
    print(f"Device: {DEVICE.upper()}\n")

    # Official distractor protocol: no global FAISS/BM25 is ever built here.
    # Each selected question gets its own ephemeral corpus from
    # build_example_corpus(), exactly matching Stage 1's own demo pattern.
    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Loading HotpotQA train + validation splits (distractor)...")
    train_dataset = load_dataset("hotpot_qa", "distractor", split="train")
    val_dataset   = load_dataset("hotpot_qa", "distractor", split="validation")
    combined_dataset = concatenate_datasets([train_dataset, val_dataset])
    train_size = len(train_dataset)

    verifier_model, verifier_tokenizer = None, None
    if VERIFIER_PATH:
        try:
            verifier_model, verifier_tokenizer = load_verifier(VERIFIER_PATH)
            print("Verifier loaded — answers will be verified after generation.\n")
        except FileNotFoundError:
            print("No verifier found — run Stage_2_Verifier_GPU.py --mode train first.\n")

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

    print(f"\n=== Stage 3: Adaptive Retrieval Demo ===")
    print(f"{len(combined_dataset)} questions loaded "
          f"({train_size} train + {len(val_dataset)} validation).")
    print("Retrieval is scoped per-question — no global corpus is ever built.")
    print(f"Enter an example index (0-{len(combined_dataset)-1}), "
          f"'eval' to run batch evaluation, or 'quit' to exit.\n")

    while True:
        cmd = input("Example index / 'eval' / 'quit': ").strip()
        if cmd.lower() == "quit":
            break
        elif cmd.lower() == "eval":
            evaluate_adaptive(embedder, verifier_model, verifier_tokenizer, num_samples=50)
        elif cmd.isdigit():
            idx = int(cmd)
            if not (0 <= idx < len(combined_dataset)):
                print(f"Index out of range (0-{len(combined_dataset)-1}).")
                continue

            origin_split = "train" if idx < train_size else "validation"
            example = combined_dataset[idx]
            print(f"\n[{origin_split}] Gold answer: {example['answer']}")

            ex_index, ex_passages, ex_bm25 = build_example_corpus(example, embedder)
            if ex_index is None:
                print("This example has an empty context — skipping.")
                continue

            adaptive_rag_query(
                example["question"], ex_index, embedder, ex_passages,
                verifier_model, verifier_tokenizer, verbose=True,
                query_type_override=example.get("type"),
                level_override=example.get("level"),
                bm25=ex_bm25,
            )
        else:
            print("Enter an example index, 'eval', or 'quit'.")

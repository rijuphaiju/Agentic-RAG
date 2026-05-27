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

import os
import pickle
import re
import sys

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
    INDEX_PATH,
    PASSAGES_PATH,
    EMBED_MODEL,
    OLLAMA_MODEL,
)

# ── Stage 2 verifier ──
from Stage_2_Verifier_GPU import load_verifier, verify, VERIFIER_PATH

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K       = 10       # passages per retrieval call
TOP_K_MULTI = 5        # passages per hop in multi-hop retrieval
MAX_HOPS    = 3        # maximum hops for multi-hop retrieval


# ─────────────────────────────────────────────
# STEP 1: QUERY COMPLEXITY CLASSIFIER
# Classifies each query into SIMPLE / MULTI_HOP / COMPARISON
# This implements Section 2.5 and 4.3 of the proposal
# ─────────────────────────────────────────────
COMPARISON_WORDS = {
    "both", "same", "different", "compare", "versus", "vs",
    "older", "newer", "bigger", "smaller", "taller", "shorter",
    "longer", "earlier", "later", "more", "less", "better", "worse",
    # NOTE: "which" and "either"/"neither" removed — they appear in MULTI_HOP
    # questions like "Which magazine was started first" and caused wrong routing.
}

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
]

def classify_query(query):
    """
    Rule-based query complexity classifier.
    Returns: 'SIMPLE', 'MULTI_HOP', or 'COMPARISON'

    This is the adaptive routing mechanism described in proposal Section 2.5.
    During the experimental phase this can be replaced with a trained classifier.
    """
    q_lower = query.lower()
    tokens  = set(q_lower.split())

    # COMPARISON: explicit comparison words + two named entities
    if tokens & COMPARISON_WORDS:
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
        if len(capitalized) >= 2:
            return "COMPARISON"
        if any(phrase in q_lower for phrase in ["same nationality", "same country",
                                                 "same language", "both from"]):
            return "COMPARISON"

    # COMPARISON: "which X or Y" pattern — two entities joined by "or"
    # e.g. "Which magazine was started first, Arthur's Magazine or First for Women?"
    # "which" alone isn't enough — require the "or" and two capitalized chunks.
    if "which" in tokens and " or " in q_lower:
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', query)
        if len(capitalized) >= 2:
            return "COMPARISON"

    # MULTI_HOP: bridging relative clause requiring intermediate entity lookup
    if any(re.search(pattern, q_lower) for pattern in MULTI_HOP_PATTERNS):
        return "MULTI_HOP"

    # Multiple question-like clauses in one sentence
    if q_lower.count(" who ") + q_lower.count(" what ") + q_lower.count(" where ") >= 2:
        return "MULTI_HOP"

    return "SIMPLE"


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

def _decompose_multihop_query(query):
    """
    Split a multi-hop query into (bridge_sub_question, simplified_main_question).
    Detects "the <noun> that/who <VP>" bridging relative clauses.

    Returns (sq1, sq2) on success, (None, None) if no pattern matched.

    Example:
      "Who directed the film that starred Emma Watson?"
        → sq1 = "What film starred Emma Watson?"
        → sq2 = "Who directed the film?"
    """
    m = re.search(
        r'\bthe\s+((?:\w+\s+){0,2}\w+)\s+(?:that|who)\s+(.+?)(?=\?|\s*$)',
        query.rstrip('?'), re.IGNORECASE,
    )
    if not m:
        return None, None

    noun        = m.group(1).strip()
    verb_phrase = m.group(2).strip().rstrip('?')
    sq1 = f"What {noun} {verb_phrase}?"

    # Simplified main: replace the relative clause with just "the <noun>"
    sq2 = re.sub(
        r'\bthe\s+(?:\w+\s+){0,2}\w+\s+(?:that|who)\s+.+?(?=\s+\w|\?|$)',
        f'the {noun}',
        query, count=1, flags=re.IGNORECASE,
    ).strip()
    if not sq2 or sq2 == query:
        sq2 = query  # fallback: use original if substitution failed

    return sq1, sq2


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

    # ── Sub-question 2: retrieve independently using the simplified question ──
    sq2_passages = retrieve_simple(sq2, index, embedder, passages, top_k=top_k)

    # Merge: sq2 passages first (directly relevant to final answer), then sq1 context
    seen     = {p["title"] for p in sq2_passages}
    combined = [dict(p, hop=2) for p in sq2_passages]
    for p in sq1_passages:
        if p["title"] not in seen:
            combined.append(dict(p, hop=1))
            seen.add(p["title"])

    return combined, bridge_ctx


def retrieve_multi_hop(query, index, embedder, passages,
                       top_k=TOP_K_MULTI, max_hops=MAX_HOPS):
    """
    Iterative multi-hop retrieval (used by Stage 4 agentic loop and as
    the fallback inside decompose_and_retrieve_multi_hop).
    Each hop reformulates the query from the top retrieved passage.
    """
    all_retrieved = []
    seen_titles   = set()
    current_query = query

    for hop in range(1, max_hops + 1):
        hop_results = retrieve_simple(current_query, index, embedder, passages, top_k)

        new_passages = []
        for p in hop_results:
            if p["title"] not in seen_titles:
                seen_titles.add(p["title"])
                p["hop"] = hop
                new_passages.append(p)

        all_retrieved.extend(new_passages)

        if not new_passages:
            break

        top_passage   = new_passages[0]["text"][:300]
        current_query = _reformulate_query(query, top_passage)

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
        r'\bolder\b|\byounger\b':            "birth year age born",
        r'\btaller\b|\bshorter\b':           "height",
        r'\bricher\b|\bwealthier\b':         "net worth wealth",
        r'\bnationality\b|\bcountry\b':      "nationality country born",
        r'\bsame language\b|\bboth from\b':  "origin country language",
        r'\bbetter\b|\bworse\b':             "comparison",
        r'\bearlier\b|\blater\b|\bfirst\b':  "founded started year",
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
def _reformulate_query(original_query, context_snippet):
    """
    Query reformulation for multi-hop retrieval.

    Strategy: extract the bridging entity (named entity from hop N's top passage)
    and combine it with the original question's predicate so that hop N+1 stays
    anchored to what the question actually asks about.

    Old behaviour returned ONLY the capitalized terms, completely losing the
    original question intent. "Robert Downey Jr." as a hop-2 query retrieves
    biography pages — not the director of the film starring RDJ.
    """
    # Extract bridging named entities from the retrieved passage
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', context_snippet)
    entities = [e for e in entities if len(e) > 2]
    entities = list(dict.fromkeys(entities))[:3]

    if not entities:
        return original_query

    bridge = " ".join(entities[:2])

    # Preserve the question's core predicate so the hop-2 query is directional
    predicates = re.findall(
        r'\b(directed|directing|wrote|written|founded|invented|born|nationality|'
        r'located|capital|started|created|authored|played|starred|published|produced)\b',
        original_query, re.IGNORECASE,
    )
    predicate = predicates[0].lower() if predicates else ""

    return f"{bridge} {predicate}".strip() if predicate else bridge


def _extract_entities(query):
    """
    Extract named entities from a comparison query.
    Uses simple capitalization heuristic.
    """
    # Remove common question/comparison words (including wh-words so
    # "Who" is never returned as an entity)
    clean = re.sub(
        r'\b(who|what|where|when|which|how|were|was|is|are|did|do|'
        r'both|the|and|of|same|different|nationality|country|from|in|'
        r'a|an|older|younger|taller|shorter|bigger|smaller|richer|'
        r'better|worse|more|less|earlier|later|longer|newer|or)\b',
        ' ', query, flags=re.IGNORECASE
    )
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', clean)
    # Filter out very short or common words
    entities = [e for e in entities if len(e) > 2]
    return list(dict.fromkeys(entities))  # deduplicate preserving order


# ─────────────────────────────────────────────
# STEP 4: ADAPTIVE RAG PIPELINE (Stage 3)
# ─────────────────────────────────────────────
def adaptive_rag_query(query, index, embedder, passages,
                       verifier_model=None, verifier_tokenizer=None):
    """
    Full Stage 3 pipeline:
    Query → Classify → Adaptive Retrieve → Generate → Verify → Return

    If verifier is provided, shows verification result alongside answer.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # ── 1. Classify query complexity ──
    query_type = classify_query(query)
    print(f"Query type: {query_type}")

    # ── 2. Adaptive retrieval ──
    if query_type == "SIMPLE":
        retrieved = retrieve_simple(query, index, embedder, passages)
    elif query_type == "MULTI_HOP":
        retrieved = retrieve_multi_hop(query, index, embedder, passages)
    else:  # COMPARISON
        retrieved = retrieve_comparison(query, index, embedder, passages)

    print(f"\nRetrieved {len(retrieved)} passages:")
    for i, p in enumerate(retrieved[:5]):  # show top 5
        hop_info = f" [hop {p.get('hop', 1)}]" if query_type == "MULTI_HOP" else ""
        ent_info = f" [{p.get('entity', '')}]" if query_type == "COMPARISON" else ""
        print(f"  [{i+1}] {p['title']}{hop_info}{ent_info} (score: {p.get('score', 0):.4f})")

    # ── 3. Rerank then generate ──
    print("\nReranking passages...")
    pool     = retrieved[:TOP_K * 2] if len(retrieved) > TOP_K else retrieved
    reranked = rerank_passages(query, pool, top_k=TOP_K)
    print("\nGenerating answer...")
    answer = generate_answer(query, reranked, query_type=query_type)
    print(f"\nAnswer: {answer}")

    # ── 4. Verify if verifier is available ──
    verification = None
    if verifier_model is not None and verifier_tokenizer is not None:
        context = " ".join([p["text"] for p in retrieved[:5]])
        verification = verify(context, answer, verifier_model, verifier_tokenizer)
        print(f"\nVerification: {verification['icon']} {verification['label']} "
              f"(confidence: {verification['confidence']:.4f})")
        print(f"  Scores: {verification['scores']}")

    return {
        "query":        query,
        "query_type":   query_type,
        "retrieved":    retrieved,
        "answer":       answer,
        "verification": verification,
    }


# ─────────────────────────────────────────────
# STEP 5: EVALUATION — Stage 3 vs Stage 1
# ─────────────────────────────────────────────
def evaluate_adaptive(index, embedder, passages,
                      verifier_model, verifier_tokenizer,
                      num_samples=100):
    """
    Evaluates Stage 3 adaptive retrieval against Stage 1 baseline.
    Reports Exact Match and Hallucination Rate for both stages.
    Implements the comparative evaluation framework from proposal Table 6.2.
    """
    import json
    from Stage_1_RAG_Pipeline import retrieve as retrieve_stage1

    print(f"\nEvaluating Stage 3 on {num_samples} HotpotQA validation samples...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    stage1_em, stage3_em         = [], []
    stage1_halluc, stage3_halluc = [], []
    results = []

    for i, example in enumerate(tqdm(dataset)):
        if i >= num_samples:
            break

        query       = example["question"]
        gold_answer = example["answer"]

        # ── Stage 1: basic retrieval ──
        s1_retrieved = retrieve_stage1(query, index, embedder, passages)
        s1_answer    = generate_answer(query, s1_retrieved)
        s1_em        = exact_match(s1_answer, gold_answer)

        # ── Stage 3: adaptive retrieval ──
        query_type = classify_query(query)
        if query_type == "SIMPLE":
            s3_retrieved = retrieve_simple(query, index, embedder, passages)
        elif query_type == "MULTI_HOP":
            s3_retrieved = retrieve_multi_hop(query, index, embedder, passages)
        else:
            s3_retrieved = retrieve_comparison(query, index, embedder, passages)

        pool      = s3_retrieved[:TOP_K * 2] if len(s3_retrieved) > TOP_K else s3_retrieved
        reranked  = rerank_passages(query, pool, top_k=TOP_K)
        s3_answer = generate_answer(query, reranked, query_type=query_type)
        s3_em     = exact_match(s3_answer, gold_answer)

        # ── Hallucination check via verifier ──
        context = " ".join([p["text"] for p in s1_retrieved[:5]])
        s1_verif = verify(context, s1_answer, verifier_model, verifier_tokenizer)
        s1_halluc = 1 if s1_verif["label"] in ("PARTIAL", "UNSUPPORTED") else 0

        context = " ".join([p["text"] for p in s3_retrieved[:5]])
        s3_verif = verify(context, s3_answer, verifier_model, verifier_tokenizer)
        s3_halluc = 1 if s3_verif["label"] in ("PARTIAL", "UNSUPPORTED") else 0

        stage1_em.append(s1_em)
        stage3_em.append(s3_em)
        stage1_halluc.append(s1_halluc)
        stage3_halluc.append(s3_halluc)

        results.append({
            "question":      query,
            "gold":          gold_answer,
            "query_type":    query_type,
            "stage1_answer": s1_answer,
            "stage3_answer": s3_answer,
            "stage1_em":     s1_em,
            "stage3_em":     s3_em,
            "stage1_halluc": s1_halluc,
            "stage3_halluc": s3_halluc,
        })

    # ── Print comparison table ──
    s1_em_score = sum(stage1_em) / len(stage1_em)
    s3_em_score = sum(stage3_em) / len(stage3_em)
    s1_hr       = sum(stage1_halluc) / len(stage1_halluc)
    s3_hr       = sum(stage3_halluc) / len(stage3_halluc)

    print(f"\n{'='*60}")
    print(f"Stage Comparison ({num_samples} samples)")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Stage 1':>10} {'Stage 3':>10} {'Change':>10}")
    print(f"{'-'*55}")
    print(f"{'Exact Match':<25} {s1_em_score:>10.4f} {s3_em_score:>10.4f} "
          f"{s3_em_score - s1_em_score:>+10.4f}")
    print(f"{'Hallucination Rate':<25} {s1_hr:>10.4f} {s3_hr:>10.4f} "
          f"{s3_hr - s1_hr:>+10.4f}")

    # Query type breakdown
    from collections import Counter
    type_counts = Counter(r["query_type"] for r in results)
    print(f"\nQuery type distribution:")
    for qtype, count in type_counts.items():
        print(f"  {qtype}: {count} ({count/len(results)*100:.1f}%)")

    # Save results
    with open("stage3_results.json", "w") as f:
        json.dump({
            "num_samples":    num_samples,
            "stage1_em":      s1_em_score,
            "stage3_em":      s3_em_score,
            "stage1_halluc":  s1_hr,
            "stage3_halluc":  s3_hr,
            "query_types":    dict(type_counts),
            "results":        results,
        }, f, indent=2)
    print("\nResults saved → stage3_results.json")
    return s3_hr, s3_em_score


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

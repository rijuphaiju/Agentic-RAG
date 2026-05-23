"""
Stage 3: Adaptive Retrieval
===========================
Project: Reducing Hallucinations in Agentic RAG Systems
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

from rag_pipeline import (
    load_faiss_index, build_faiss_index, load_hotpotqa_passages,
    generate_answer, normalize_answer, exact_match,
    INDEX_PATH, PASSAGES_PATH, EMBED_MODEL, OLLAMA_MODEL,
)
from verifier_gpu import load_verifier, verify, VERIFIER_PATH

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── import Stage 1 helpers ──
from rag_pipeline import (
    load_faiss_index,
    build_faiss_index,
    load_hotpotqa_passages,
    generate_answer,
    normalize_answer,
    exact_match,
    INDEX_PATH,
    PASSAGES_PATH,
    EMBED_MODEL,
    OLLAMA_MODEL,
)

# ── import Stage 2 verifier ──
from verifier_gpu import load_verifier, verify, VERIFIER_PATH

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
    "which", "either", "neither"
}

MULTI_HOP_WORDS = {
    "who directed", "who wrote", "who founded", "who created",
    "who invented", "who discovered", "what country", "what city",
    "what year", "when was", "where was", "what is the nationality",
    "what did", "who is the", "what was the"
}

def classify_query(query):
    """
    Rule-based query complexity classifier.
    Returns: 'SIMPLE', 'MULTI_HOP', or 'COMPARISON'

    This is the adaptive routing mechanism described in proposal Section 2.5.
    During the experimental phase this can be replaced with a trained classifier.
    """
    q_lower = query.lower()
    tokens  = set(q_lower.split())

    # COMPARISON: contains comparison words + mentions two entities
    if tokens & COMPARISON_WORDS:
        # Check for two named entities (simple heuristic: two capitalized words)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
        if len(capitalized) >= 2:
            return "COMPARISON"
        # Also catch "same nationality", "same country" patterns
        if any(phrase in q_lower for phrase in ["same nationality", "same country",
                                                 "same language", "both from"]):
            return "COMPARISON"

    # MULTI_HOP: requires finding intermediate entity first
    if any(phrase in q_lower for phrase in MULTI_HOP_WORDS):
        return "MULTI_HOP"

    # Contains multiple question-like clauses
    if q_lower.count(" who ") + q_lower.count(" what ") + q_lower.count(" where ") >= 2:
        return "MULTI_HOP"

    return "SIMPLE"


# ─────────────────────────────────────────────
# STEP 2: RETRIEVAL STRATEGIES
# ─────────────────────────────────────────────
def retrieve_simple(query, index, embedder, passages, top_k=TOP_K):
    """
    Standard top-k dense retrieval — same as Stage 1.
    Used for SIMPLE queries.
    """
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(passages):
            results.append({
                "title": passages[idx]["title"],
                "text":  passages[idx]["text"],
                "score": float(score),
                "hop":   1,
            })
    return results


def retrieve_multi_hop(query, index, embedder, passages,
                       top_k=TOP_K_MULTI, max_hops=MAX_HOPS):
    """
    Iterative multi-hop retrieval.
    Each hop uses the previously retrieved passages to reformulate
    the query and retrieve additional supporting evidence.

    Implements the iterative retrieval loop described in proposal Section 6.3.6.
    """
    all_retrieved = []
    seen_titles   = set()
    current_query = query

    for hop in range(1, max_hops + 1):
        # Retrieve for current query
        hop_results = retrieve_simple(current_query, index, embedder, passages, top_k)

        # Add new passages only
        new_passages = []
        for p in hop_results:
            if p["title"] not in seen_titles:
                seen_titles.add(p["title"])
                p["hop"] = hop
                new_passages.append(p)

        all_retrieved.extend(new_passages)

        if not new_passages:
            break

        # Reformulate query using top retrieved passage for next hop
        top_passage = new_passages[0]["text"][:300]
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

    all_retrieved = []
    seen_titles   = set()

    if len(entities) >= 2:
        # Retrieve for each entity separately
        for entity in entities[:2]:
            entity_query   = f"{entity} "
            entity_results = retrieve_simple(
                entity_query, index, embedder, passages, top_k
            )
            for p in entity_results:
                if p["title"] not in seen_titles:
                    seen_titles.add(p["title"])
                    p["entity"] = entity
                    all_retrieved.append(p)
    else:
        # Fallback to standard retrieval if entity extraction fails
        all_retrieved = retrieve_simple(query, index, embedder, passages, top_k * 2)

    # Also retrieve for the full query to catch joint passages
    full_results = retrieve_simple(query, index, embedder, passages, top_k // 2)
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
    Simple query reformulation for multi-hop retrieval.
    Combines original query with key terms from retrieved context.
    """
    # Extract key nouns/names from context (capitalized words)
    key_terms = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?\b', context_snippet)
    key_terms = list(dict.fromkeys(key_terms))[:3]  # deduplicate, keep top 3

    if key_terms:
        return f"{original_query} {' '.join(key_terms)}"
    return original_query


def _extract_entities(query):
    """
    Extract named entities from a comparison query.
    Uses simple capitalization heuristic.
    """
    # Remove common question words
    clean = re.sub(
        r'\b(were|was|is|are|did|do|both|the|and|of|same|different|'
        r'nationality|country|from|in|a|an)\b',
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

    # ── 3. Generate answer ──
    print("\nGenerating answer...")
    answer = generate_answer(query, retrieved[:TOP_K])
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
    from rag_pipeline import retrieve as retrieve_stage1

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

        s3_answer = generate_answer(query, s3_retrieved[:TOP_K])
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

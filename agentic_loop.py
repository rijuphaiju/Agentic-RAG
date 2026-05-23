"""
Stage 4: Agentic Decision Loop
===============================
Project: Reducing Hallucinations in Agentic RAG Systems
Proposal Section: 2.6, 6.3.6, 6.3.6.1

Final stage — integrates all previous stages into a self-correcting loop:
  1. Retrieve → Generate → Verify
  2. If PARTIAL/UNSUPPORTED → reformulate query → re-retrieve → try again
  3. If still failing after MAX_ITERATIONS → ABSTAIN

This implements the agentic loop formalised in proposal Equation 2.9:
  vt = Verifier(at, dt) ∈ {Unsupported, Partial, Supported}

Usage:
  python agentic_loop.py
  python agentic_loop.py --eval
"""

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from datasets import load_dataset
from tqdm import tqdm

# Stage 1
from rag_pipeline import (
    load_faiss_index,
    build_faiss_index,
    load_hotpotqa_passages,
    generate_answer,
    exact_match,
    INDEX_PATH,
    PASSAGES_PATH,
    OLLAMA_MODEL,
)

# Stage 2
from verifier_gpu import load_verifier, verify, VERIFIER_PATH

# Stage 3
from adaptive_retrieval import (
    classify_query,
    retrieve_simple,
    retrieve_multi_hop,
    retrieve_comparison,
    TOP_K,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_ITERATIONS = 3        # maximum re-retrieval attempts before abstaining
CONFIDENCE_THRESHOLD = 0.6  # minimum confidence to accept SUPPORTED answer
ABSTAIN_MESSAGE = "I cannot confidently answer this question based on the available evidence."


# ─────────────────────────────────────────────
# QUERY REFORMULATION
# Used when verifier returns PARTIAL/UNSUPPORTED
# ─────────────────────────────────────────────
def reformulate_query(original_query, iteration, retrieved_passages, answer):
    """
    Reformulates the query for re-retrieval when verification fails.
    Each iteration tries a different reformulation strategy.

    Implements the query reformulation function f(q0, Ct-1)
    from proposal Section 2.6 Equation.
    """
    if iteration == 1:
        # Strategy 1: extract key entities from the original query
        import re
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', original_query)
        if entities:
            return " ".join(entities[:3])
        return original_query

    elif iteration == 2:
        # Strategy 2: use top retrieved passage title + original query keywords
        if retrieved_passages:
            top_title = retrieved_passages[0]["title"]
            # Extract question keywords (nouns, remove stop words)
            stop = {"was", "were", "is", "are", "the", "a", "an", "of",
                    "in", "on", "at", "to", "for", "and", "or", "did",
                    "do", "what", "who", "where", "when", "which", "how"}
            keywords = [w for w in original_query.lower().split()
                       if w not in stop and len(w) > 2]
            return f"{top_title} {' '.join(keywords[:4])}"
        return original_query

    else:
        # Strategy 3: rephrase as a direct entity lookup
        import re
        # Extract quoted or capitalized multi-word phrases
        phrases = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b', original_query)
        if phrases:
            return f"{phrases[0]} biography history"
        return original_query + " explanation facts"


# ─────────────────────────────────────────────
# CORE AGENTIC LOOP
# ─────────────────────────────────────────────
def agentic_query(query, index, embedder, passages,
                  verifier_model, verifier_tokenizer,
                  verbose=True):
    """
    Full agentic RAG pipeline with self-correction loop.

    State transition (proposal Section 2.6):
      For each iteration t:
        dt = retrieve(f(q0, Ct-1))    ← adaptive retrieval
        at = LLM(q0, Ct-1 ∪ {dt})    ← answer generation
        vt = Verifier(at, dt)          ← faithfulness verification

      Terminate when:
        - vt = SUPPORTED with confidence >= threshold  → return answer
        - t >= MAX_ITERATIONS                          → abstain
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

    query_type      = classify_query(query)
    current_query   = query
    all_retrieved   = []
    iteration_log   = []

    if verbose:
        print(f"Query type: {query_type}")

    for iteration in range(1, MAX_ITERATIONS + 1):
        if verbose:
            print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")
            if iteration > 1:
                print(f"Reformulated query: {current_query}")

        # ── RETRIEVAL (adaptive based on query type) ──
        if query_type == "COMPARISON":
            retrieved = retrieve_comparison(current_query, index, embedder, passages)
        elif query_type == "MULTI_HOP":
            retrieved = retrieve_multi_hop(current_query, index, embedder, passages)
        else:
            retrieved = retrieve_simple(current_query, index, embedder, passages)

        # Merge with previously retrieved passages (Ct-1 ∪ {dt})
        seen_titles = {p["title"] for p in all_retrieved}
        for p in retrieved:
            if p["title"] not in seen_titles:
                all_retrieved.append(p)
                seen_titles.add(p["title"])

        if verbose:
            print(f"Retrieved {len(retrieved)} passages "
                  f"({len(all_retrieved)} total accumulated)")
            for i, p in enumerate(retrieved[:3]):
                print(f"  [{i+1}] {p['title']} (score: {p.get('score', 0):.4f})")

        # ── GENERATION ──
        # Use accumulated context from all iterations
        context_passages = all_retrieved[:TOP_K]
        answer = generate_answer(query, context_passages)

        if verbose:
            print(f"\nGenerated answer: {answer}")

        # ── VERIFICATION ──
        context_text = " ".join([p["text"] for p in context_passages])
        verification = verify(context_text, answer, verifier_model, verifier_tokenizer)
        label      = verification["label"]
        confidence = verification["confidence"]

        if verbose:
            icon = {"SUPPORTED": "✅", "PARTIAL": "⚠️", "UNSUPPORTED": "❌"}.get(label, "?")
            print(f"Verification: {icon} {label} (confidence: {confidence:.4f})")

        iteration_log.append({
            "iteration":    iteration,
            "query":        current_query,
            "answer":       answer,
            "label":        label,
            "confidence":   confidence,
            "num_retrieved": len(retrieved),
        })

        # ── DECISION ──
        if label == "SUPPORTED" and confidence >= CONFIDENCE_THRESHOLD:
            # ✅ Answer verified — return it
            if verbose:
                print(f"\n✅ Answer accepted after {iteration} iteration(s).")
            return {
                "query":          query,
                "query_type":     query_type,
                "answer":         answer,
                "status":         "SUPPORTED",
                "iterations":     iteration,
                "abstained":      False,
                "verification":   verification,
                "iteration_log":  iteration_log,
            }

        elif iteration < MAX_ITERATIONS:
            # ⚠️ Answer not verified — reformulate and retry
            if verbose:
                print(f"Answer not verified. Reformulating query for iteration {iteration+1}...")
            current_query = reformulate_query(
                query, iteration, retrieved, answer
            )

        # else: MAX_ITERATIONS reached → fall through to abstention

    # ── ABSTENTION ──
    # Max iterations reached without a verified answer
    if verbose:
        print(f"\n❌ Could not verify answer after {MAX_ITERATIONS} iterations.")
        print(f"Abstaining: {ABSTAIN_MESSAGE}")

    return {
        "query":         query,
        "query_type":    query_type,
        "answer":        ABSTAIN_MESSAGE,
        "status":        "ABSTAINED",
        "iterations":    MAX_ITERATIONS,
        "abstained":     True,
        "verification":  None,
        "iteration_log": iteration_log,
    }


# ─────────────────────────────────────────────
# EVALUATION — All 4 stages compared
# Implements proposal Table 6.2
# ─────────────────────────────────────────────
def evaluate_all_stages(index, embedder, passages,
                        verifier_model, verifier_tokenizer,
                        num_samples=50):
    """
    Evaluates all 4 pipeline stages on HotpotQA validation set.
    Produces the full comparison table from proposal Table 6.2:

      Metric          Stage1  Stage2  Stage3  Stage4
      Exact Match       -       -       -       -
      Hallucination     -       -       -       -
      Abstention Rate   0%      -       -       -
    """
    print(f"\nEvaluating all stages on {num_samples} HotpotQA validation samples...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    metrics = {
        "stage1": {"em": [], "halluc": [], "abstain": []},
        "stage2": {"em": [], "halluc": [], "abstain": []},
        "stage3": {"em": [], "halluc": [], "abstain": []},
        "stage4": {"em": [], "halluc": [], "abstain": []},
    }
    results = []

    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        if i >= num_samples:
            break

        query  = example["question"]
        gold   = example["answer"]
        qtype  = classify_query(query)

        # ── Stage 1: Basic RAG (no verification) ──
        s1_retrieved = retrieve_simple(query, index, embedder, passages)
        s1_answer    = generate_answer(query, s1_retrieved)
        s1_ctx       = " ".join([p["text"] for p in s1_retrieved[:5]])
        s1_verif     = verify(s1_ctx, s1_answer, verifier_model, verifier_tokenizer)
        s1_halluc    = 1 if s1_verif["label"] in ("PARTIAL", "UNSUPPORTED") else 0

        # ── Stage 2: Basic RAG + Verifier (no re-retrieval) ──
        # Same retrieval as Stage 1 but with verification label
        s2_answer  = s1_answer   # same answer, just now verified
        s2_halluc  = s1_halluc   # same hallucination check

        # ── Stage 3: Adaptive Retrieval + Verifier ──
        if qtype == "COMPARISON":
            s3_retrieved = retrieve_comparison(query, index, embedder, passages)
        elif qtype == "MULTI_HOP":
            s3_retrieved = retrieve_multi_hop(query, index, embedder, passages)
        else:
            s3_retrieved = retrieve_simple(query, index, embedder, passages)

        s3_answer = generate_answer(query, s3_retrieved[:TOP_K])
        s3_ctx    = " ".join([p["text"] for p in s3_retrieved[:5]])
        s3_verif  = verify(s3_ctx, s3_answer, verifier_model, verifier_tokenizer)
        s3_halluc = 1 if s3_verif["label"] in ("PARTIAL", "UNSUPPORTED") else 0

        # ── Stage 4: Full Agentic Loop ──
        s4_result  = agentic_query(
            query, index, embedder, passages,
            verifier_model, verifier_tokenizer,
            verbose=False
        )
        s4_answer  = s4_result["answer"]
        s4_abstain = 1 if s4_result["abstained"] else 0
        s4_halluc  = 0 if s4_result["status"] == "SUPPORTED" else (
            0 if s4_result["abstained"] else 1
        )

        # ── Exact Match scores ──
        s1_em = exact_match(s1_answer, gold)
        s2_em = exact_match(s2_answer, gold)
        s3_em = exact_match(s3_answer, gold)
        s4_em = 0 if s4_result["abstained"] else exact_match(s4_answer, gold)

        # Record
        for stage, em, halluc, abstain in [
            ("stage1", s1_em, s1_halluc, 0),
            ("stage2", s2_em, s2_halluc, 0),
            ("stage3", s3_em, s3_halluc, 0),
            ("stage4", s4_em, s4_halluc, s4_abstain),
        ]:
            metrics[stage]["em"].append(em)
            metrics[stage]["halluc"].append(halluc)
            metrics[stage]["abstain"].append(abstain)

        results.append({
            "question":    query,
            "gold":        gold,
            "query_type":  qtype,
            "stage1":      {"answer": s1_answer, "em": s1_em, "halluc": s1_halluc},
            "stage2":      {"answer": s2_answer, "em": s2_em, "halluc": s2_halluc},
            "stage3":      {"answer": s3_answer, "em": s3_em, "halluc": s3_halluc},
            "stage4":      {"answer": s4_answer, "em": s4_em,
                           "halluc": s4_halluc, "abstain": s4_abstain,
                           "iterations": s4_result["iterations"]},
        })

    # ── Compute final metrics ──
    def avg(lst): return sum(lst) / len(lst) if lst else 0

    summary = {}
    for stage in ["stage1", "stage2", "stage3", "stage4"]:
        summary[stage] = {
            "exact_match":        avg(metrics[stage]["em"]),
            "hallucination_rate": avg(metrics[stage]["halluc"]),
            "abstention_rate":    avg(metrics[stage]["abstain"]),
        }

    # ── Print Table 6.2 ──
    print(f"\n{'='*70}")
    print(f"Full Stage Comparison — {num_samples} HotpotQA validation samples")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Stage1':>10} {'Stage2':>10} {'Stage3':>10} {'Stage4':>10}")
    print(f"{'-'*65}")

    for metric, label in [
        ("exact_match",        "Exact Match"),
        ("hallucination_rate", "Hallucination Rate"),
        ("abstention_rate",    "Abstention Rate"),
    ]:
        row = f"{label:<25}"
        for stage in ["stage1", "stage2", "stage3", "stage4"]:
            val = summary[stage][metric]
            row += f" {val:>10.4f}"
        print(row)

    print(f"\nKey finding:")
    s1_h = summary["stage1"]["hallucination_rate"]
    s4_h = summary["stage4"]["hallucination_rate"]
    reduction = (s1_h - s4_h) / s1_h * 100 if s1_h > 0 else 0
    print(f"  Hallucination reduced from {s1_h:.4f} (Stage1) "
          f"to {s4_h:.4f} (Stage4) = {reduction:.1f}% reduction")
    print(f"  Abstention rate: {summary['stage4']['abstention_rate']:.4f} "
          f"(system chose to abstain rather than hallucinate)")

    # Save
    with open("stage4_full_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print("\nResults saved → stage4_full_results.json")
    return summary


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="Run full 4-stage evaluation")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of evaluation samples")
    args = parser.parse_args()

    print("Stage 4: Agentic Decision Loop")
    print(f"Device: {DEVICE.upper()}\n")

    # Load FAISS index
    if os.path.exists(INDEX_PATH) and os.path.exists(PASSAGES_PATH):
        index, embedder, passages = load_faiss_index()
    else:
        passages = load_hotpotqa_passages()
        index, embedder, passages = build_faiss_index(passages)

    # Load verifier
    if not os.path.exists(VERIFIER_PATH):
        print("No verifier found. Run: python verifier_gpu.py --mode train")
        sys.exit(1)

    verifier_model, verifier_tokenizer = load_verifier(VERIFIER_PATH)
    print("All components loaded.\n")

    if args.eval:
        evaluate_all_stages(
            index, embedder, passages,
            verifier_model, verifier_tokenizer,
            num_samples=args.samples
        )
    else:
        # Interactive demo
        print("=== Stage 4: Agentic RAG Demo ===")
        print(f"Max iterations: {MAX_ITERATIONS} | "
              f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print("Type 'eval' to run evaluation, 'quit' to exit.\n")

        while True:
            query = input("Enter your question: ").strip()
            if not query:
                continue
            if query.lower() == "quit":
                break
            elif query.lower() == "eval":
                evaluate_all_stages(
                    index, embedder, passages,
                    verifier_model, verifier_tokenizer,
                    num_samples=args.samples
                )
            else:
                result = agentic_query(
                    query, index, embedder, passages,
                    verifier_model, verifier_tokenizer,
                    verbose=True
                )
                print(f"\n{'='*60}")
                print(f"FINAL STATUS: {result['status']}")
                print(f"FINAL ANSWER: {result['answer']}")
                print(f"Iterations used: {result['iterations']}/{MAX_ITERATIONS}")
                if result['abstained']:
                    print("(System abstained — chose silence over hallucination)")
                print(f"{'='*60}")

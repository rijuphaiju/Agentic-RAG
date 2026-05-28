"""
Stage 4: Agentic Decision Loop
===============================
Project: HARA — Hallucination-Aware Retrieval Agent
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
from Stage_1_RAG_Pipeline import (
    load_faiss_index,
    build_faiss_index,
    load_hotpotqa_passages,
    generate_answer,
    rerank_passages,
    exact_match,
    INDEX_PATH,
    PASSAGES_PATH,
    OLLAMA_MODEL,
)

# Stage 2
from Stage_2_Verifier_GPU import load_verifier, verify, VERIFIER_PATH

# Stage 3
from Stage_3_Adaptive_Retrieval import (
    classify_query,
    retrieve_simple,
    retrieve_multi_hop,
    retrieve_comparison,
    TOP_K,
    _RETRIEVAL_PARAMS,
    _DEFAULT_PARAMS,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE         = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
MAX_ITERATIONS = 5        # maximum re-retrieval attempts before abstaining
CONFIDENCE_THRESHOLD = 0.50          # minimum confidence to accept a SUPPORTED verdict
PARTIAL_ACCEPTANCE_THRESHOLD = 0.50  # accept PARTIAL if confidence ≥ 0.50 — the verifier
                                     # rarely gives > 0.62 for correct multi-hop answers;
                                     # 0.62 still caused 64% BEST_EFFORT on 150 samples
COMPARISON_PARTIAL_THRESHOLD = 0.42  # comparisons are structurally non-verbatim so the
                                     # verifier systematically underscores them


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
    import re

    # Always extract capitalized entity names to preserve them across iterations
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', original_query)
    # Remove single-word question starters like "Who", "Which", "What"
    entities = [e for e in entities
                if e.lower() not in {"who", "what", "where", "when", "which", "how"}]

    if iteration == 1:
        # Strategy 1: key entities + the dominant predicate word from the question.
        # Pure entity-only queries (e.g. "Scott Derrickson Ed Wood") lose the
        # semantic intent and FAISS drifts to unrelated biography pages.
        predicate_words = re.findall(
            r'\b(nationality|founded|formed|born|older|younger|directed|'
            r'wrote|capital|started|created|invented|authored|played)\b',
            original_query, re.IGNORECASE
        )
        predicate = predicate_words[0].lower() if predicate_words else ""
        entity_str = " ".join(entities[:3])
        return f"{entity_str} {predicate}".strip() if predicate else entity_str or original_query

    elif iteration == 2:
        # Strategy 2: key entities + top retrieved passage title
        # Put entities FIRST so the semantic focus stays on them
        top_title = retrieved_passages[0]["title"] if retrieved_passages else ""
        entity_str = " ".join(entities[:3])
        # Only append title if it adds something beyond the entities
        if top_title and top_title not in entity_str:
            return f"{entity_str} {top_title}".strip()
        return entity_str or original_query

    else:
        # Strategy 3: both entities + domain-specific comparison term
        # e.g. "older/younger" → append "birth year born"; "nationality" → "country born"
        attribute_map = {
            r'\bolder\b|\byounger\b':       "birth year born",
            r'\btaller\b|\bshorter\b':      "height",
            r'\bricher\b|\bwealthier\b':    "net worth",
            r'\bnationality\b|\bcountry\b': "nationality country",
            r'\bearlier\b|\blater\b|\bfirst\b': "founded year",
        }
        suffix = "biography facts"  # default
        for pattern, attr in attribute_map.items():
            if re.search(pattern, original_query, re.IGNORECASE):
                suffix = attr
                break

        if len(entities) >= 2:
            return f"{entities[0]} {entities[1]} {suffix}"
        elif entities:
            return f"{entities[0]} {suffix}"
        return original_query + " facts details"


# ─────────────────────────────────────────────
# CORE AGENTIC LOOP
# ─────────────────────────────────────────────
def agentic_query(query, index, embedder, passages,
                  verifier_model, verifier_tokenizer,
                  verbose=True, query_type_override=None, level_override=None):
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

    # Use ground-truth type from dataset when available (evaluation), else regex (live chat)
    if query_type_override == "bridge":
        query_type = "MULTI_HOP"
    elif query_type_override == "comparison":
        query_type = "COMPARISON"
    else:
        query_type = classify_query(query)
    current_query   = query
    all_retrieved   = []
    iteration_log   = []
    best_candidate  = None   # best (answer, label, confidence) seen across all iterations
    _label_rank     = {"SUPPORTED": 2, "PARTIAL": 1, "UNSUPPORTED": 0}

    level = level_override or "medium"
    r_top_k, r_max_hops = _RETRIEVAL_PARAMS.get(
        (query_type, level), _DEFAULT_PARAMS[query_type]
    )

    if verbose:
        print(f"Query type: {query_type} | Level: {level} | top_k={r_top_k}, max_hops={r_max_hops}")

    for iteration in range(1, MAX_ITERATIONS + 1):
        if verbose:
            print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")
            if iteration > 1:
                print(f"Reformulated query: {current_query}")

        # ── RETRIEVAL (adaptive based on query type × level) ──
        if query_type == "COMPARISON":
            retrieved = retrieve_comparison(current_query, index, embedder, passages,
                                            top_k=r_top_k)
        elif query_type == "MULTI_HOP":
            retrieved = retrieve_multi_hop(current_query, index, embedder, passages,
                                           top_k=r_top_k, max_hops=r_max_hops)
        else:
            retrieved = retrieve_simple(current_query, index, embedder, passages,
                                        top_k=r_top_k)

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
        # Sort accumulated context by retrieval score so the best passages from
        # any iteration are always in the top-K window sent to the LLM.
        # Previously this used insertion order, meaning good passages retrieved
        # after reformulation were silently dropped when all_retrieved > TOP_K.
        pool = sorted(
            all_retrieved, key=lambda p: p.get("score", 0), reverse=True
        )[:TOP_K * 2]
        context_passages = rerank_passages(query, pool, top_k=TOP_K)
        answer = generate_answer(query, context_passages, query_type=query_type)

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

        # Track best answer seen so far (fallback if all iterations fail to verify)
        new_rank  = (_label_rank.get(label, 0), confidence)
        best_rank = (_label_rank.get(best_candidate["label"], 0), best_candidate["confidence"]) if best_candidate else (-1, -1)
        if new_rank > best_rank:
            best_candidate = {"answer": answer, "label": label,
                              "confidence": confidence, "verification": verification}

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
            # PARTIAL and UNSUPPORTED both trigger reformulation (report Section 6.3.6):
            # "the agent is programmed to reformulate the query and trigger additional
            #  retrieval steps rather than returning a hallucinated answer"
            if verbose:
                print(f"Answer not verified ({label}). Reformulating for iteration {iteration+1}...")
            current_query = reformulate_query(
                query, iteration, retrieved, answer
            )

        # else: MAX_ITERATIONS reached → fall through to abstention

    # ── ABSTAIN: no verified answer after MAX_ITERATIONS (proposal Section 2.6, 4.3) ──
    # "when the iteration count exceeds a maximum Tmax, at which point the system
    #  abstains rather than returning an unverified answer" — proposal Section 2.6
    if verbose:
        print(f"\n❌ Could not verify an answer after {MAX_ITERATIONS} iterations.")
        print("Abstaining — returning explicit 'insufficient evidence' response.")

    return {
        "query":         query,
        "query_type":    query_type,
        "answer":        "I cannot confidently answer this question based on the available evidence.",
        "status":        "ABSTAINED",
        "iterations":    MAX_ITERATIONS,
        "abstained":     True,
        "verification":  best_candidate["verification"] if best_candidate else {},
        "iteration_log": iteration_log,
    }


# ─────────────────────────────────────────────
# ROUTING LAYER
# Empirical routing based on per-type F1 from Run 3 evaluation:
#   COMPARISON : Stage 4 wins  (0.4451 vs 0.3991)
#   MULTI_HOP  : Stage 3 wins  (0.1543 vs 0.0884)
#   SIMPLE     : Stage 3 wins  (safer; similar scores)
# ─────────────────────────────────────────────
def routed_query(query, index, embedder, passages,
                 verifier_model, verifier_tokenizer,
                 verbose=False):
    """
    Hybrid router: delegates each query to whichever stage scored best
    for that query type in the Run 3 empirical evaluation.

    COMPARISON → full agentic loop (Stage 4)
    MULTI_HOP  → Stage 3 adaptive retrieval + rerank
    SIMPLE     → Stage 3 adaptive retrieval + rerank

    Returns a dict in the same schema as agentic_query() so Stage 6 can
    treat it as a drop-in replacement.
    """
    query_type = classify_query(query)

    if query_type == "COMPARISON":
        # Stage 4's self-consistency check is most valuable for COMPARISON:
        # it catches entity-swap errors (e.g. "Coldplay formed first" when
        # Radiohead did). Delegate the full agentic loop.
        if verbose:
            print(f"[Router] COMPARISON → Stage 4 agentic loop")
        return agentic_query(
            query, index, embedder, passages,
            verifier_model, verifier_tokenizer,
            verbose=verbose,
        )

    else:
        # MULTI_HOP and SIMPLE: Stage 3 retrieval + rerank is more reliable.
        # The agentic loop's query reformulation hurts multi-hop by drifting
        # away from the bridging entity; Stage 3's iterative retrieval is better.
        if verbose:
            print(f"[Router] {query_type} → Stage 3 retrieval + rerank")

        if query_type == "MULTI_HOP":
            retrieved = retrieve_multi_hop(query, index, embedder, passages)
        else:
            retrieved = retrieve_simple(query, index, embedder, passages)

        pool     = retrieved[:TOP_K * 2] if len(retrieved) > TOP_K else retrieved
        reranked = rerank_passages(query, pool, top_k=TOP_K)
        answer   = generate_answer(query, reranked, query_type=query_type)

        context_text = " ".join(p["text"] for p in reranked[:5])
        verification = verify(context_text, answer, verifier_model, verifier_tokenizer)

        label = verification["label"]
        conf  = verification["confidence"]

        if verbose:
            icon  = {"SUPPORTED": "✅", "PARTIAL": "⚠️", "UNSUPPORTED": "❌"}.get(label, "?")
            print(f"[Router] Answer: {answer}")
            print(f"[Router] Verification: {icon} {label} ({conf:.4f})")

        # Abstain if the answer is entirely unsupported by retrieved context.
        # PARTIAL is returned (best available in a single-pass route) and counted
        # as a hallucination in metrics per proposal Section 6.3.6.1.
        if label == "UNSUPPORTED":
            if verbose:
                print("[Router] UNSUPPORTED — abstaining.")
            return {
                "query":          query,
                "query_type":     query_type,
                "answer":         "I cannot confidently answer this question based on the available evidence.",
                "status":         "ABSTAINED",
                "iterations":     1,
                "abstained":      True,
                "verification":   verification,
                "iteration_log":  [],
            }

        return {
            "query":          query,
            "query_type":     query_type,
            "answer":         answer,
            "status":         "STAGE3_ROUTED",
            "iterations":     1,
            "abstained":      False,
            "verification":   verification,
            "iteration_log":  [],
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

        s3_answer = generate_answer(query, s3_retrieved[:TOP_K], query_type=qtype)
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
        # Read actual abstention flag from the result (proposal Section 6.3.6.1)
        s4_abstain = 1 if s4_result.get("abstained") else 0
        # Hallucination: PARTIAL/UNSUPPORTED answers count as hallucinations per
        # proposal Equation 6.7. Abstained queries do NOT count as hallucinations
        # (no answer was returned).
        if s4_result.get("abstained"):
            s4_halluc = 0
        elif s4_result["status"] == "SUPPORTED":
            s4_halluc = 0
        else:
            s4_halluc = 1

        # ── Exact Match scores ──
        s1_em = exact_match(s1_answer, gold)
        s2_em = exact_match(s2_answer, gold)
        s3_em = exact_match(s3_answer, gold)
        s4_em = exact_match(s4_answer, gold)

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
    s4_abs = summary["stage4"]["abstention_rate"]
    print(f"  Stage 4 abstention rate: {s4_abs:.4f} "
          f"(system abstains rather than hallucinating — proposal Section 4.3)")

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
        print("=== HARA — Stage 4: Agentic Loop Demo ===")
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
                print(f"{'='*60}")

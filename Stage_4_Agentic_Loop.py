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
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from datasets import load_dataset
from tqdm import tqdm

# Stage 1
import ollama

from Stage_1_RAG_Pipeline import (
    load_faiss_index,
    build_faiss_index,
    load_hotpotqa_passages,
    generate_answer,
    rerank_passages,
    exact_match,
    llm_judge_supported,
    rag_query as _s1_rag_query,
    INDEX_PATH,
    PASSAGES_PATH,
    OLLAMA_MODEL,
)

# Stage 2
from Stage_2_Verifier_GPU import load_verifier, verify, build_verify_context, VERIFIER_PATH

# Stage 3
from Stage_3_Adaptive_Retrieval import (
    classify_query,
    estimate_complexity,
    retrieve_simple,
    retrieve_multi_hop,
    retrieve_comparison,
    adaptive_retrieve_with_coverage_check,
    _retrieve_for_type,
    TOP_K,
    _RETRIEVAL_PARAMS,   # kept for backward compat
    _DEFAULT_PARAMS,     # kept for backward compat
    _BUDGET,
    _DEFAULT_BUDGET,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE         = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
MAX_ITERATIONS = 7        # maximum re-retrieval attempts before abstaining
CONFIDENCE_THRESHOLD = 0.50          # minimum confidence to accept a SUPPORTED verdict
LOW_SUPPORT_THRESHOLD = 0.15         # used only by Yes/No flip check


# ─────────────────────────────────────────────
# REFUSAL DETECTION
# ─────────────────────────────────────────────
_REFUSAL_RE = re.compile(
    r'\b(cannot provide|i cannot|do not appear|does not appear|'
    r'no information|not mentioned|not in the context|cannot be found|'
    r'not available|i don\'t have|does not contain|'
    r'not found|not determined|not specified|cannot determine|'
    r'information is not|answer is not|year is not|cannot be determined)\b',
    re.IGNORECASE,
)


def _is_refusal(answer: str) -> bool:
    """True when the LLM refused because the entity wasn't in retrieved context."""
    return bool(_REFUSAL_RE.search(answer))


# Question patterns that require a numeric answer.
# Key: regex matching the question. Value: description used in verbose logs.
_NUMERIC_QUESTION_PATTERNS = [
    (re.compile(r'\bpopulation\b',              re.I), "population (expects a number)"),
    (re.compile(r'\bhow many\b',                re.I), "how many (expects a number)"),
    (re.compile(r'\bhow much\b',                re.I), "how much (expects a number)"),
    (re.compile(r'\bwhat year\b',               re.I), "what year (expects a year)"),
    (re.compile(r'\bin what year\b',            re.I), "in what year (expects a year)"),
    (re.compile(r'\bwhat.{0,10}year.{0,10}(?:was|did|were|is)\b', re.I), "year question"),
    (re.compile(r'\bhow (tall|long|far|old|high|wide|deep|large|big|small)\b', re.I),
     "measurement (expects a number)"),
]


def _answer_type_mismatch(question: str, answer: str) -> bool:
    """
    Return True when the question expects a numeric answer but the pipeline
    returned a non-numeric answer (e.g. a place name instead of a population).

    This is a general post-generation check that does not require ground truth.
    It catches cases where retrieval found the correct intermediate entity but
    failed to retrieve the numeric fact the question is actually asking for.

    Examples:
      "what was the population..." -> "New Hampshire"    -> MISMATCH (no number)
      "what was the population..." -> "6,241"            -> OK
      "in what year was X born?"   -> "Meredith"         -> MISMATCH (no year)
      "in what year was X born?"   -> "1945"             -> OK
      "who directed X?"            -> "Christopher Nolan" -> OK (no numeric expectation)
    """
    ans_lower = answer.strip().lower()
    # Ignore refusals -- they are handled separately
    if _is_refusal(answer):
        return False

    for pattern, _desc in _NUMERIC_QUESTION_PATTERNS:
        if pattern.search(question):
            # Question expects a number -- check the answer contains digits
            if not re.search(r'\d', answer):
                return True   # e.g. "New Hampshire" for a population question
            break             # contains digits -- no mismatch for this pattern

    return False




# ─────────────────────────────────────────────
# QUERY REFORMULATION
# Used when verifier returns PARTIAL/UNSUPPORTED
# ─────────────────────────────────────────────
def reformulate_query(original_query, iteration, retrieved_passages, answer):
    """
    Reformulates the query for re-retrieval when verification fails.
    All strategies stay anchored to the original question's entities and
    predicates — no generic expansion terms that cause retrieval drift.

    Implements f(q0, Ct-1) from proposal Section 2.6 Equation.
    """
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', original_query)
    entities = [e for e in entities
                if e.lower() not in {"who", "what", "where", "when", "which", "how"}]

    # When the query has no named entities, anchor on the current answer entity
    # so subsequent hops verify THAT candidate rather than retrieving anything new.
    if not entities and answer:
        ans_entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', answer)
        ans_entities = [e for e in ans_entities
                        if len(e) > 2 and e.lower() not in
                        {"who", "what", "where", "when", "which", "how", "the", "a"}]
        entities = ans_entities[:2]

    if iteration == 1:
        # Strategy 1: key entities + predicate extracted from the original question.
        # Keeping the predicate preserves intent; dropping it lets FAISS drift to
        # unrelated biography pages that share only the entity name.
        predicate_words = re.findall(
            r'\b(nationality|founded|formed|born|older|younger|directed|'
            r'wrote|capital|started|created|invented|authored|played|'
            r'died|married|graduated|established|located|published)\b',
            original_query, re.IGNORECASE
        )
        predicate = predicate_words[0].lower() if predicate_words else ""
        entity_str = " ".join(entities[:3])
        return f"{entity_str} {predicate}".strip() if predicate else entity_str or original_query

    elif iteration == 2:
        # Strategy 2: entities + top retrieved passage title (if it adds signal).
        top_title = retrieved_passages[0]["title"] if retrieved_passages else ""
        entity_str = " ".join(entities[:3])
        if top_title and top_title not in entity_str and len(top_title.split()) <= 5:
            return f"{entity_str} {top_title}".strip()
        return entity_str or original_query

    elif iteration == 3:
        # Strategy 3: domain-specific attribute term extracted from the question.
        # Only use terms that are present in the original question — no generic
        # "biography facts" defaults that cause FAISS to retrieve unrelated content.
        attribute_map = {
            r'\bolder\b|\byounger\b':           "birth year",
            r'\btaller\b|\bshorter\b':          "height",
            r'\bricher\b|\bwealthier\b':        "net worth",
            r'\bnationality\b|\bcountry\b':     "nationality",
            r'\bearlier\b|\blater\b|\bfirst\b': "founded year",
            r'\bdied\b|\bdeath\b':              "death year",
            r'\bborn\b|\bbirth\b':              "birth year",
        }
        entity_str = " ".join(entities[:2]) if entities else ""
        for pattern, attr in attribute_map.items():
            if re.search(pattern, original_query, re.IGNORECASE):
                return f"{entity_str} {attr}".strip() if entity_str else attr
        # No domain match → entities only, no generic suffix
        return entity_str or original_query

    else:
        # Strategy 4+: return the original question verbatim to reset drift.
        # Further reformulation at iteration 4+ would compound the drift from
        # earlier strategies; starting fresh gives retrieval another chance on
        # the unmodified intent.
        return original_query


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
    # Accepts Stage 3's resolved type ("MULTI_HOP"/"COMPARISON"/"SIMPLE") directly
    # so Stage 4 never reclassifies a question Stage 3 already classified.
    if query_type_override in ("MULTI_HOP", "COMPARISON", "SIMPLE"):
        query_type = query_type_override
    elif query_type_override == "bridge":
        query_type = "MULTI_HOP"
    elif query_type_override == "comparison":
        query_type = "COMPARISON"
    else:
        query_type = classify_query(query)

    # Complexity drives retrieval budget — use HotpotQA ground truth when available,
    # otherwise estimate from the question itself (same logic as Stage 3).
    complexity  = level_override if level_override else estimate_complexity(query)
    budget      = _BUDGET.get((query_type, complexity), _DEFAULT_BUDGET[query_type])
    r_top_k     = budget["top_k"]
    r_max_hops  = budget["max_hops"]

    # Confidence-gated escalation (mirrors Stage 3's adaptive_rag_query).
    # 71.1% of HotpotQA "bridge" questions are answerable with single-hop retrieval.
    # Locking in SIMPLE for the whole loop prevents 5 iterations of MULTI_HOP drift
    # producing a fabricated answer worse than the Stage 3 result.
    _SIMPLE_SUFFICIENT = 0.65
    _s4_gate_fired     = False   # True when gate downgraded MULTI_HOP -> SIMPLE
    _type_from_dataset = query_type_override in ("bridge", "comparison",
                                                  "MULTI_HOP", "COMPARISON", "SIMPLE")
    if query_type == "MULTI_HOP" and not _type_from_dataset:
        from Stage_3_Adaptive_Retrieval import estimate_retrieval_confidence
        _sp   = retrieve_simple(query, index, embedder, passages, top_k=r_top_k)
        _sr   = rerank_passages(query, _sp, top_k=TOP_K)
        _sc   = estimate_retrieval_confidence(_sr)
        if _sc >= _SIMPLE_SUFFICIENT:
            if verbose:
                print(f"[Escalation gate] SIMPLE conf={_sc:.3f} >= "
                      f"{_SIMPLE_SUFFICIENT} -- using SIMPLE for entire loop")
            _s4_gate_fired = True
            query_type = "SIMPLE"
            budget     = _BUDGET.get(("SIMPLE", complexity), _DEFAULT_BUDGET["SIMPLE"])
            r_top_k    = budget["top_k"]
            r_max_hops = budget["max_hops"]

    current_query   = query
    all_retrieved   = []
    iteration_log   = []
    best_candidate  = None   # best (answer, label, confidence) seen across all iterations
    _label_rank     = {"SUPPORTED": 2, "PARTIAL": 1, "UNSUPPORTED": 0}
    bridge_ctx      = None   # set by Stage 3 multi-hop decomposition

    if verbose:
        print(f"Type: {query_type} | Complexity: {complexity.upper()} | "
              f"top_k={r_top_k}, max_hops={r_max_hops}")

    # Always run the full agentic loop (up to MAX_ITERATIONS).
    # If Stage 3 (iteration 1) gives a SUPPORTED answer it is returned immediately;
    # restricting to 1 iteration for dataset questions prevented retry on PARTIAL
    # outcomes and silently returned weak answers as BEST_EFFORT.
    _max_iter      = MAX_ITERATIONS
    _actual_iters  = 0  # track real iterations run for accurate UI display

    for iteration in range(1, _max_iter + 1):
        _actual_iters += 1
        if verbose:
            print(f"\n--- Iteration {iteration}/{_max_iter} ---")
            if iteration > 1:
                print(f"Reformulated query: {current_query}")

        # ── RETRIEVAL (Stage 3 adaptive) ──────────────────────────────────────
        # Iteration 1: full Stage 3 pipeline — complexity-aware budget + coverage
        #   check + expansion before handing context to the LLM.
        # Iterations 2+: targeted re-retrieval with reformulated query, merged
        #   with the accumulated context pool from previous iterations so the LLM
        #   always sees the best evidence gathered across the whole agentic loop.
        if iteration == 1:
            pool, context_passages, bridge_ctx, ret_stats = \
                adaptive_retrieve_with_coverage_check(
                    current_query, query_type,
                    index, embedder, passages,
                    budget, verbose=verbose,
                )
            retrieved = pool
            if verbose:
                print(f"  [Stage 3] conf={ret_stats['confidence']:.3f} "
                      f"cov={ret_stats['coverage']:.2f} "
                      f"expansions={ret_stats['expansions_used']}")
        else:
            # Reformulated query — use dispatch helper, then merge + rerank
            retrieved, bridge_ctx = _retrieve_for_type(
                current_query, query_type,
                index, embedder, passages,
                r_top_k, r_max_hops,
            )

        # Accumulate passages across iterations (Ct-1 ∪ {dt})
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

        # ── BUILD CONTEXT FOR GENERATION ──────────────────────────────────────
        # Iteration 1: context_passages already built by Stage 3 (with reranking).
        # Iterations 2+: sort accumulated pool by score, rerank to TOP_K.
        if iteration > 1:
            acc_pool = sorted(
                all_retrieved, key=lambda p: p.get("score", 0), reverse=True
            )[:r_top_k * 2]
            context_passages = rerank_passages(query, acc_pool, top_k=TOP_K)

        # Bridge context from multi-hop decomposition is injected AFTER reranking
        # so the cross-encoder scores real Wikipedia passages only.
        llm_context = list(context_passages)
        if bridge_ctx:
            llm_context = [{"title": "Bridge Finding", "text": bridge_ctx}] + llm_context

        # ── GENERATION (Stage 1) ───────────────────────────────────────────────
        answer = generate_answer(query, llm_context, query_type=query_type)

        # Gate-fired refusal rescue (iteration 1 only): the escalation gate
        # downgraded MULTI_HOP -> SIMPLE, but SIMPLE produced no answer.
        # The cross-encoder gave a false-positive confidence on a passage that
        # looked topically relevant but didn't contain the 2-hop answer chain.
        # Rescue: run full MULTI_HOP on this iteration and lock in MULTI_HOP
        # for the rest of the loop.
        if iteration == 1 and _s4_gate_fired and _is_refusal(answer):
            if verbose:
                print("[Fallback] Gate-SIMPLE refusal — escalating loop to MULTI_HOP")
            mh_budget  = _BUDGET.get(("MULTI_HOP", complexity), _DEFAULT_BUDGET["MULTI_HOP"])
            mh_pool, mh_ctx, mh_bridge, _ = adaptive_retrieve_with_coverage_check(
                query, "MULTI_HOP", index, embedder, passages, mh_budget, verbose,
            )
            mh_llm = list(mh_ctx)
            if mh_bridge:
                mh_llm = [{"title": "Bridge Finding", "text": mh_bridge}] + mh_llm
            mh_answer = generate_answer(query, mh_llm, query_type="MULTI_HOP")
            if not _is_refusal(mh_answer):
                answer           = mh_answer
                context_passages = mh_ctx
                llm_context      = mh_llm
                bridge_ctx       = mh_bridge
                # Switch the loop to MULTI_HOP for remaining iterations
                _s4_gate_fired   = False
                query_type       = "MULTI_HOP"
                budget           = mh_budget
                r_top_k          = mh_budget["top_k"]
                r_max_hops       = mh_budget["max_hops"]

        if verbose:
            print(f"\nGenerated answer: {answer}")

        # ── VERIFICATION ──
        context_text = build_verify_context(context_passages, answer)
        verification = verify(context_text, answer, verifier_model, verifier_tokenizer, question=query)
        label      = verification["label"]
        confidence = verification["confidence"]

        # ── YES/NO FLIP ──
        # When the LLM answers "Yes" or "No" but verifier support is near-zero,
        # the opposite answer may be correct. This catches comparison questions
        # where the LLM defaults to "Yes" (both buildings are office towers =
        # "real estate") but the gold answer requires distinguishing the specific
        # USE (publishing HQ != real estate company). Try the opposite and take
        # whichever answer the verifier supports more strongly.
        _ans_lower = answer.strip().lower().rstrip(".")
        if _ans_lower in ("yes", "no") and verification["scores"].get("SUPPORTED", 1.0) < LOW_SUPPORT_THRESHOLD:
            _opposite     = "No" if _ans_lower == "yes" else "Yes"
            _opp_ctx      = build_verify_context(context_passages, _opposite)
            _opp_verif    = verify(_opp_ctx, _opposite, verifier_model, verifier_tokenizer, question=query)
            _orig_supp    = verification["scores"].get("SUPPORTED", 0.0)
            _opp_supp     = _opp_verif["scores"].get("SUPPORTED", 0.0)
            if _opp_supp > _orig_supp:
                if verbose:
                    print(f"[Yes/No flip] '{answer}' supp={_orig_supp:.3f} < "
                          f"'{_opposite}' supp={_opp_supp:.3f} — flipping answer")
                answer       = _opposite
                verification = _opp_verif
                label        = _opp_verif["label"]
                confidence   = _opp_verif["confidence"]

        # ── DECISION ──
        scores = verification.get("scores", {})

        # ── LLM JUDGE (MULTI_HOP / COMPARISON) ───────────────────────────────
        # Validates SUPPORTED claims before accepting them. Catches entity-role
        # errors (e.g. answer is the writer, not the director the question asks
        # about). Runs BEFORE logging so best_candidate reflects the post-judge
        # label, preventing a judge-rejected SUPPORTED from leaking out as
        # best_candidate["label"] == "SUPPORTED" in the post-loop safety net.
        _run_judge = query_type in ("MULTI_HOP", "COMPARISON")
        if label == "SUPPORTED" and confidence >= CONFIDENCE_THRESHOLD and _run_judge:
            judge_ok = llm_judge_supported(query, answer, context_passages, verbose)
            if not judge_ok:
                if verbose:
                    print(f"[LLM Judge] Overriding SUPPORTED → forcing retry "
                          f"(entity not in correct role in context)")
                orig = scores
                scores = {
                    "SUPPORTED":   orig["PARTIAL"],
                    "PARTIAL":     orig["SUPPORTED"],
                    "UNSUPPORTED": orig["UNSUPPORTED"],
                }
                label      = "PARTIAL"
                confidence = orig["SUPPORTED"]
                verification = {**verification, "label": "PARTIAL",
                                "confidence": confidence, "scores": scores}

        # Log and update best_candidate AFTER judge so stored state is final.
        if verbose:
            icon = {"SUPPORTED": "✅", "PARTIAL": "⚠️", "UNSUPPORTED": "❌"}.get(label, "?")
            print(f"Verification: {icon} {label} (confidence: {confidence:.4f})")

        iteration_log.append({
            "iteration":     iteration,
            "query":         current_query,
            "answer":        answer,
            "label":         label,
            "confidence":    confidence,
            "num_retrieved": len(retrieved),
        })

        new_rank  = (_label_rank.get(label, 0), confidence)
        best_rank = (_label_rank.get(best_candidate["label"], 0), best_candidate["confidence"]) if best_candidate else (-1, -1)
        if new_rank > best_rank:
            best_candidate = {"answer": answer, "label": label,
                              "confidence": confidence, "verification": verification}

        # ── ACCEPT ────────────────────────────────────────────────────────────
        if label == "SUPPORTED" and confidence >= CONFIDENCE_THRESHOLD:
            if verbose:
                print(f"\n✅ Answer accepted after {iteration} iteration(s).")
            return {
                "query":         query,
                "query_type":    query_type,
                "complexity":    complexity,
                "level":         complexity,
                "answer":        answer,
                "status":        "SUPPORTED",
                "iterations":    iteration,
                "abstained":     False,
                "verification":  verification,
                "iteration_log": iteration_log,
            }

        # ── RETRY ─────────────────────────────────────────────────────────────
        # PARTIAL and UNSUPPORTED both trigger reformulation and retry.
        # PARTIAL is not accepted automatically — the verifier is not calibrated
        # well enough on HotpotQA multi-hop for partial acceptance to be safe.
        if iteration < _max_iter:
            if verbose:
                print(f"  [{label}] Reformulating for iteration {iteration + 1}...")
            current_query = reformulate_query(query, iteration, retrieved, answer)
        elif verbose:
            print(f"  [{label}] Max iterations ({_max_iter}) reached — will abstain.")

    # ── POST-LOOP ──────────────────────────────────────────────────────────────
    # Safety net: if best_candidate carries a SUPPORTED label (can only happen if
    # the immediate in-loop return was bypassed — should not occur in normal flow)
    # return it rather than discarding a verified answer.
    if (best_candidate
            and best_candidate["label"] == "SUPPORTED"
            and best_candidate["confidence"] >= CONFIDENCE_THRESHOLD):
        if verbose:
            print(f"\n✅ [Safety net] Returning SUPPORTED answer from best_candidate.")
        return {
            "query":         query,
            "query_type":    query_type,
            "complexity":    complexity,
            "level":         complexity,
            "answer":        best_candidate["answer"],
            "status":        "SUPPORTED",
            "iterations":    _actual_iters,
            "abstained":     False,
            "verification":  best_candidate["verification"],
            "iteration_log": iteration_log,
        }

    # ── ABSTAIN ────────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n❌ Could not verify an answer after {_actual_iters} iteration(s).")
        print("Abstaining — returning explicit 'insufficient evidence' response.")

    return {
        "query":         query,
        "query_type":    query_type,
        "complexity":    complexity,
        "level":         complexity,
        "answer":        "I cannot confidently answer this question based on the available evidence.",
        "status":        "ABSTAINED",
        "iterations":    _actual_iters,
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

        context_text = build_verify_context(reranked, answer)
        verification = verify(context_text, answer, verifier_model, verifier_tokenizer, question=query)

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

        # ── Stage 1: Basic RAG — uses the full hybrid+rerank pipeline ──
        # Must use _s1_rag_query (retrieve_hybrid→rerank→generate) not bare
        # retrieve_simple, otherwise the Stage 1 baseline is weaker than what
        # Stage 1 actually produces and the comparison table is misleading.
        s1_result    = _s1_rag_query(query, index, embedder, passages)
        s1_answer    = s1_result["answer"]
        s1_retrieved = s1_result["retrieved_passages"]
        s1_ctx       = build_verify_context(s1_retrieved, s1_answer)
        s1_verif     = verify(s1_ctx, s1_answer, verifier_model, verifier_tokenizer, question=query)
        s1_halluc    = 1 if s1_verif["label"] in ("PARTIAL", "UNSUPPORTED") else 0

        # ── Stage 2: Same retrieval as Stage 1 + Verifier label shown ──
        # Stage 2's contribution is the verifier; retrieval is identical to Stage 1.
        s2_answer  = s1_answer
        s2_halluc  = s1_halluc

        # ── Stage 3: Adaptive Retrieval + Verifier ──
        if qtype == "COMPARISON":
            s3_retrieved = retrieve_comparison(query, index, embedder, passages)
        elif qtype == "MULTI_HOP":
            s3_retrieved = retrieve_multi_hop(query, index, embedder, passages)
        else:
            s3_retrieved = retrieve_simple(query, index, embedder, passages)

        s3_answer = generate_answer(query, s3_retrieved[:TOP_K], query_type=qtype)
        s3_ctx    = build_verify_context(s3_retrieved, s3_answer)
        s3_verif  = verify(s3_ctx, s3_answer, verifier_model, verifier_tokenizer, question=query)
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

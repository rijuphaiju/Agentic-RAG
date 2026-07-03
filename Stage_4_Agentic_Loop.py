"""
Stage 4: Agentic Reasoning Loop
================================
Project: HARA — Hallucination-Aware Retrieval Agent
Proposal Section: 2.6, 6.3.6, 6.3.6.1

Stage 4 is a diagnosis-driven reasoning agent, not "Stage 3 with more
retries." It calls Stage 3's adaptive_rag_query() exactly once to obtain its
initial state — Stage 3 has already solved query classification, difficulty
estimation, retrieval-strategy selection, coverage-gated expansion, and one
verifier-guided refinement, and none of that is duplicated here.

From that starting point, Stage 4 repeats:
    Diagnose (why did the current answer succeed or fail?)
      -> Choose ONE explicit action (deterministic priority policy, no LLM planner)
      -> Execute it (reusing Stage 3's own retrieval helpers)
      -> Regenerate -> Reverify
until a stopping criterion fires — never a fixed iteration count.

PARTIAL answers are never silently abstained on: memory.best_candidate tracks
the best (label, confidence) seen across the whole episode (including Stage
3's own result), and the final status is derived from it after the loop ends
— SUPPORTED-confident -> "SUPPORTED", any PARTIAL-or-better -> "BEST_EFFORT",
otherwise -> "ABSTAINED". Abstention only happens when nothing better than
UNSUPPORTED was ever reached.

Usage:
  python Stage_4_Agentic_Loop.py
  python Stage_4_Agentic_Loop.py --eval
"""

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import ollama
import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

# ── Stage 1 ──
from Stage_1_RAG_Pipeline import (
    build_example_corpus,
    generate_answer,
    rerank_passages,
    retrieve as _retrieve_dense,
    exact_match,
    llm_judge_supported,
    EMBED_MODEL,
    OLLAMA_MODEL,
    RERANK_POOL,
)

# ── Stage 2 (V2: evidence-grounded self-verification) ──
# Stage 4 calls verify() directly (not the verify_legacy() shim Stage 3
# uses) so it gets the full structured report — overall_status,
# overall_confidence, support_score, coverage, failure_reason,
# recommended_action, claims — instead of just a bare label. load_verifier()
# is the same legacy-shaped loader Stage 3 uses (returns the NLIVerifier
# instance in both unpacked slots) so the two stages can share one
# `verifier_model, verifier_tokenizer = load_verifier(...)` loading call.
from Stage_2_Verifier import load_verifier, verify, normalize_verification, VERIFIER_PATH
from Stage_4_RepairPrompts import augment_query_for_repair, build_repair_instruction

# ── Stage 3 — the retrieval engine. Stage 4 reuses these directly rather
# than reimplementing query classification, budget selection, or any of
# Stage 3's nine retrieval pipelines. ──
from Stage_3_Adaptive_Retrieval import (
    adaptive_rag_query,
    check_evidence_coverage,
    retrieve_simple,
    decompose_and_retrieve_multi_hop,
    _reformulate_query,
    _targeted_expand,
    _expand_missing_comparison_entity,
    _is_refusal,
    TOP_K,
    TOP_K_MULTI,
    COVERAGE_THRESHOLD,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

MAX_ACTIONS           = 4     # hard ceiling on agent actions beyond Stage 3's own initial pass
MAX_ITERATIONS        = MAX_ACTIONS  # legacy alias — Stage_6_Evaluation.py imports this name
CONFIDENCE_THRESHOLD  = 0.50  # verifier confidence required for a hard "SUPPORTED" status
LOW_SUPPORT_THRESHOLD = 0.15  # used by the Yes/No flip check

# Task 5 — comprehensive early-stopping thresholds, checked across all five
# Stage 2 V2 dimensions rather than just label+confidence.
FAST_KEEP_DIMENSION_THRESHOLD  = 0.70  # every dimension must clear this to KEEP immediately
SEVERE_CONTRADICTION_THRESHOLD = 0.30  # below this, contradiction is treated as severe
LOW_COMPLETENESS_THRESHOLD     = 0.60  # "high support but low completeness" composite rule

# Task 3 — failure_reason -> action, read directly off Stage 2's diagnosis
# rather than only through its own recommended_action, so this stays a
# defined, inspectable policy even if Stage 2's own mapping changes.
# CONTRADICTED_BY_EVIDENCE / INSUFFICIENT_EVIDENCE / NO_RELEVANT_PASSAGE are
# handled separately (see _select_action) with the "expand once, then
# abstain" nuance Task 2 asks for, so they're intentionally not in this table.
_REASON_ACTION_TABLE = {
    "ANSWER_TYPE_MISMATCH": "REWRITE",
    "QUESTION_NOT_ANSWERED": "REWRITE",
    "WRONG_ENTITY": "COMPARE",
    "WRONG_YEAR": "REWRITE",
    "WRONG_NUMBER": "REWRITE",
    "PARTIAL_COMPARISON": "COMPARE",
    "MISSING_REQUIRED_ENTITY": "EXPAND",
    "MISSING_REQUIRED_NUMBER": "EXPAND",
    "MISSING_REQUIRED_DATE": "EXPAND",
}

_ABSTAIN_MESSAGE = "I cannot confidently answer this question based on the available evidence."

# Question patterns that expect a numeric answer — used by the answer-type-
# mismatch diagnostic (e.g. "what year..." answered with a person's name).
_NUMERIC_QUESTION_PATTERNS = [
    r'\bpopulation\b',
    r'\bhow many\b',
    r'\bhow much\b',
    r'\bwhat year\b',
    r'\bin what year\b',
    r'\bwhat.{0,10}year.{0,10}(?:was|did|were|is)\b',
    r'\bhow (?:tall|long|far|old|high|wide|deep|large|big|small)\b',
]
import re as _re
_NUMERIC_QUESTION_RE = _re.compile("|".join(_NUMERIC_QUESTION_PATTERNS), _re.IGNORECASE)


def _answer_type_mismatch(question: str, answer: str) -> bool:
    """
    True when the question expects a numeric answer (year, population,
    measurement, ...) but the returned answer contains no digits at all —
    e.g. "what was the population of X?" answered with a place name instead
    of a number. A general, ground-truth-free diagnostic signal: retrieval
    found a plausible-looking entity but not the actual fact being asked for.
    """
    if _is_refusal(answer):
        return False
    if _NUMERIC_QUESTION_RE.search(question) and not _re.search(r'\d', answer):
        return True
    return False


# ─────────────────────────────────────────────
# AGENT MEMORY
# Explicit state across one Stage 4 episode — replaces the old "iteration
# number decides behaviour" loop with state the diagnosis step can reason over.
# ─────────────────────────────────────────────
class AgentMemory:
    """
    Tracks, across the whole episode:
      - accumulated, title-deduplicated evidence (never discarded between actions)
      - which actions have already been attempted, and against which target
        (entity for EXPAND/COMPARE, rewritten query for REWRITE), so the same
        failed strategy is never repeated
      - the full answer history (for stabilization detection)
      - the best (label, confidence)-ranked candidate seen so far, across
        every action including Stage 3's own initial pass
    """

    _LABEL_RANK = {"SUPPORTED": 2, "PARTIAL": 1, "UNSUPPORTED": 0}

    def __init__(self, query: str, query_type: str, level: str):
        self.query = query
        self.query_type = query_type
        self.level = level

        self.evidence: list = []
        self._evidence_titles: set = set()

        self.actions_tried: list = []     # [{"action": str, "target": str|None}, ...]
        self.answer_history: list = []    # [{"action","answer","label","confidence"}, ...]
        self.best_candidate = None        # {"answer","label","confidence","verification"}

    def add_evidence(self, passages: list) -> bool:
        """Merge new passages into accumulated evidence. Returns True if any
        genuinely new (previously unseen) title was added — the "did this
        action find new evidence" signal the diagnosis step needs."""
        found_new = False
        for p in passages:
            if p["title"] not in self._evidence_titles:
                self.evidence.append(p)
                self._evidence_titles.add(p["title"])
                found_new = True
        return found_new

    def record_action(self, action: str, target=None) -> None:
        self.actions_tried.append({"action": action, "target": target})

    def has_tried(self, action: str, target=None) -> bool:
        return any(a["action"] == action and a["target"] == target for a in self.actions_tried)

    def record_answer(self, action: str, answer: str, label: str, confidence: float,
                       verification: dict) -> None:
        self.answer_history.append({
            "action": action, "answer": answer, "label": label, "confidence": confidence,
        })
        new_rank = (self._LABEL_RANK.get(label, 0), confidence)
        best_rank = ((self._LABEL_RANK.get(self.best_candidate["label"], 0), self.best_candidate["confidence"])
                     if self.best_candidate else (-1, -1))
        if new_rank > best_rank:
            self.best_candidate = {
                "answer": answer, "label": label, "confidence": confidence,
                "verification": verification, "action": action,
            }

    def answer_stabilized(self) -> bool:
        """True when the two most recent recorded answers are identical after
        normalization — a sign that further actions are unlikely to change
        anything."""
        if len(self.answer_history) < 2:
            return False
        a = self.answer_history[-1]["answer"].strip().lower()
        b = self.answer_history[-2]["answer"].strip().lower()
        return a == b


# ─────────────────────────────────────────────
# DIAGNOSIS
# The actual decision input — replaces "which iteration number are we on".
# ─────────────────────────────────────────────
def _diagnose(query: str, query_type: str, answer: str, verification: dict,
              reranked_passages: list, memory: AgentMemory, found_new_evidence: bool) -> dict:
    """
    Produces a structured explanation of why the current answer did or did
    not succeed. Every field here is a real decision input for _select_action
    — nothing in the agent loop branches on iteration count.

    `verification` is Stage 2 V2's report (normalized via normalize_verification if
    it came from Stage 3's legacy-shaped result). Beyond label/confidence,
    this now also surfaces failure_reason, recommended_action, and the
    verifier's own per-claim coverage (kept as `verifier_coverage`, distinct
    from `coverage` below — that field is this project's existing
    entity-mention coverage check via check_evidence_coverage(), which
    _select_action's COVERAGE_THRESHOLD logic already depends on and must
    keep meaning exactly what it always has).
    """
    v = normalize_verification(verification) if verification else {}
    label = v.get("overall_status", "UNSUPPORTED")
    confidence = v.get("overall_confidence", 0.0)
    support = v.get("support_score", 0.0)
    hallucination_probability = round(1 - support, 4)

    cov_info = check_evidence_coverage(query, reranked_passages)

    return {
        "label": label,
        "confidence": confidence,
        "support_score": support,
        "hallucination_probability": hallucination_probability,
        "coverage": cov_info["coverage"],
        "missing_entities": cov_info["missing"],
        "found_entities": cov_info["found"],
        "found_new_evidence": found_new_evidence,
        "answer_stabilized": memory.answer_stabilized(),
        "is_refusal": _is_refusal(answer),
        "answer_type_mismatch": _answer_type_mismatch(query, answer),
        # Additive — Stage 2 V2's full five-dimension report, not ignored:
        "failure_reason": v.get("failure_reason"),
        "recommended_action": v.get("recommended_action"),
        "verifier_coverage": v.get("coverage"),
        "claims": v.get("claims", []),
        "contradiction_score": v.get("contradiction_score", 1.0),
        "answer_type_score": v.get("answer_type_score", 0.5),
        "question_coverage_score": v.get("question_coverage_score", 0.5),
        "answer_completeness_score": v.get("answer_completeness_score", 1.0),
        "question_intent": v.get("question_intent"),
    }


# ─────────────────────────────────────────────
# ACTION SELECTION
# Deterministic, priority-ordered — no LLM planner.
# ─────────────────────────────────────────────
def _has_tried_any_expansion(memory: AgentMemory) -> bool:
    """True once any evidence-broadening action (EXPAND/COMPARE/DECOMPOSE)
    has been attempted this episode — used to implement Task 2/5's "expand
    once, then abstain" policy for contradiction/insufficient-evidence."""
    return any(a["action"] in ("EXPAND", "COMPARE", "DECOMPOSE") for a in memory.actions_tried)


def _select_action(diagnosis: dict, query_type: str, memory: AgentMemory) -> str:
    """
    Returns one of KEEP / EXPAND / COMPARE / DECOMPOSE / REWRITE / ABSTAIN.

    "ABSTAIN" here means "stop attempting further actions" — it does NOT by
    itself mean the episode ends with a hard abstention message. The final
    reported status is derived separately from memory.best_candidate after
    the loop ends (see agentic_query), specifically so a PARTIAL best
    candidate is returned as a best-effort answer instead of being discarded.

    Layered priority (each step below is additive on top of the last — none
    of the original heuristics were removed, new dimension-aware checks are
    inserted ahead of them):
      1. Comprehensive fast KEEP — every one of Stage 2's five dimensions
         already looks good, so there is nothing to gain from another action.
      2. Stagnation circuit breaker (unchanged).
      3. Fast ABSTAIN — contradiction is severe AND evidence has already
         been broadened once; further actions are unlikely to fix a genuine
         evidence disagreement.
      4. CONTRADICTED_BY_EVIDENCE / INSUFFICIENT_EVIDENCE / NO_RELEVANT_PASSAGE
         — expand exactly once; if the same reason persists afterward, abstain.
      5. Explicit failure_reason -> action table (Task 3).
      6. High support but incomplete answer (e.g. an unresolved comparison)
         — rephrase rather than retrieve more.
      7. Stage 2's own recommended_action, if still untried (pre-existing).
      8-11. This project's original type-based retrieval heuristics (unchanged).
      12. Nothing left to try.
    """
    contradiction_score = diagnosis.get("contradiction_score", 1.0)
    coverage_score = diagnosis.get("question_coverage_score", 0.5)
    completeness_score = diagnosis.get("answer_completeness_score", 1.0)
    answer_type_score = diagnosis.get("answer_type_score", 0.5)
    reason = diagnosis.get("failure_reason")

    # 1. Comprehensive fast KEEP (Task 5) — strengthens (never loosens) the
    #    original SUPPORTED+confidence+type-mismatch check by additionally
    #    requiring every Stage 2 dimension to clear its own threshold.
    if (diagnosis["label"] == "SUPPORTED"
            and diagnosis["confidence"] >= CONFIDENCE_THRESHOLD
            and not diagnosis["answer_type_mismatch"]
            and answer_type_score >= FAST_KEEP_DIMENSION_THRESHOLD
            and coverage_score >= FAST_KEEP_DIMENSION_THRESHOLD
            and completeness_score >= FAST_KEEP_DIMENSION_THRESHOLD
            and contradiction_score >= FAST_KEEP_DIMENSION_THRESHOLD):
        return "KEEP"

    # 2. Stagnation: the last action found nothing new and the answer hasn't
    #    changed -- repeating any action from here is very unlikely to help.
    if not diagnosis["found_new_evidence"] and diagnosis["answer_stabilized"]:
        return "ABSTAIN"

    # 3. Fast ABSTAIN (Task 5) — severe contradiction after evidence has
    #    already been broadened once this episode.
    if contradiction_score < SEVERE_CONTRADICTION_THRESHOLD and _has_tried_any_expansion(memory):
        return "ABSTAIN"

    # 4. Task 2 — contradiction / insufficient evidence: expand once, then
    #    abstain if the same reason persists.
    if reason in ("CONTRADICTED_BY_EVIDENCE", "INSUFFICIENT_EVIDENCE", "NO_RELEVANT_PASSAGE"):
        if _has_tried_any_expansion(memory):
            return "ABSTAIN"
        return "EXPAND"

    # 5. Task 3 — explicit failure_reason -> action table.
    mapped = _REASON_ACTION_TABLE.get(reason)
    if mapped == "COMPARE":
        target = next((e for e in diagnosis["missing_entities"]
                       if not memory.has_tried("COMPARE", e)), None)
        if target or not memory.has_tried("COMPARE", None):
            return "COMPARE"
    elif mapped == "EXPAND":
        target = next((e for e in diagnosis["missing_entities"]
                       if not memory.has_tried("EXPAND", e)), None)
        if target or not memory.has_tried("EXPAND", None):
            return "EXPAND"
    elif mapped == "REWRITE" and not memory.has_tried("REWRITE"):
        return "REWRITE"

    # 6. Task 3 — high support but incomplete answer: rephrase, don't retrieve.
    if diagnosis["support_score"] >= CONFIDENCE_THRESHOLD and completeness_score < LOW_COMPLETENESS_THRESHOLD:
        if not memory.has_tried("REWRITE"):
            return "REWRITE"

    # 7. Stage 2's recommended_action, if it's still a live, untried option.
    rec = diagnosis.get("recommended_action")
    if rec == "DECOMPOSE" and not memory.has_tried("DECOMPOSE"):
        return "DECOMPOSE"
    if rec == "COMPARE":
        target = next((e for e in diagnosis["missing_entities"]
                       if not memory.has_tried("COMPARE", e)), None)
        if target:
            return "COMPARE"
    if rec == "EXPAND":
        target = next((e for e in diagnosis["missing_entities"]
                       if not memory.has_tried("EXPAND", e)), None)
        if target:
            return "EXPAND"
    if rec == "REWRITE" and not memory.has_tried("REWRITE"):
        return "REWRITE"

    # 8. Bridge questions with a refusal or coverage gap -- try full
    #    decomposition (bridge entity discovery + sub-question generation),
    #    but only once.
    if (query_type == "MULTI_HOP"
            and (diagnosis["is_refusal"] or diagnosis["coverage"] < COVERAGE_THRESHOLD)
            and not memory.has_tried("DECOMPOSE")):
        return "DECOMPOSE"

    # 9. Comparison questions with a specific under-covered entity.
    if query_type == "COMPARISON":
        target = next((e for e in diagnosis["missing_entities"]
                       if not memory.has_tried("COMPARE", e)), None)
        if target:
            return "COMPARE"

    # 10. General coverage gap -- targeted expansion for whichever missing
    #     entity hasn't been targeted yet.
    target = next((e for e in diagnosis["missing_entities"]
                   if not memory.has_tried("EXPAND", e)), None)
    if target:
        return "EXPAND"

    # 11. Coverage looks adequate but the answer is still weak/wrong-typed --
    #     reformulate the query, once.
    if not memory.has_tried("REWRITE"):
        return "REWRITE"

    # 12. Nothing left to try.
    return "ABSTAIN"


# ─────────────────────────────────────────────
# ACTION EXECUTION
# Every action reuses Stage 3's own retrieval helpers directly.
# ─────────────────────────────────────────────
def _llm_rewrite_query(query: str) -> str:
    """
    Last-resort LLM query rewrite — used only when deterministic
    reformulation (_reformulate_query) fails to produce anything different
    from the original question. This and DECOMPOSE (which itself needs one
    LLM call to produce an intermediate bridge answer) are the only two
    places Stage 4 calls the LLM for something other than final answer
    generation — action *selection* itself is never LLM-driven.
    """
    prompt = (
        "Rewrite this question as a short search query using different "
        "wording, keeping the same meaning and all named entities:\n\n"
        f"{query}\n\nRewritten search query:"
    )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 30},
        )
        rewritten = response["message"]["content"].strip().strip('"')
        return rewritten if rewritten else query
    except Exception:
        return query


def _execute_action(action: str, memory: AgentMemory, index, embedder, passages, bm25,
                     diagnosis: dict, verbose: bool):
    """
    Executes the chosen action and returns (new_passages, bridge_ctx, target).
    new_passages are NOT yet merged into memory — the caller does that via
    memory.add_evidence() so found-new-evidence can be measured.
    """
    query = memory.query

    if action == "EXPAND":
        target = next((e for e in diagnosis["missing_entities"]
                       if not memory.has_tried("EXPAND", e)), None)
        if target is not None:
            new_passages = _targeted_expand(query, [], [target], index, embedder, passages,
                                             bm25, top_k=TOP_K_MULTI)
            return new_passages, None, target
        # No specific missing entity (e.g. a CONTRADICTED_BY_EVIDENCE or
        # INSUFFICIENT_EVIDENCE diagnosis with no named gap) — broaden the
        # evidence pool generically against the plain query instead of being
        # a no-op, so this repair path always makes a genuine retrieval attempt.
        if memory.has_tried("EXPAND", None):
            return [], None, None
        new_passages = retrieve_simple(query, index, embedder, passages, top_k=RERANK_POOL, bm25=bm25)
        return new_passages, None, None

    if action == "COMPARE":
        target = next((e for e in diagnosis["missing_entities"]
                       if not memory.has_tried("COMPARE", e)), None)
        if target is not None:
            new_passages = _expand_missing_comparison_entity(query, [], [target], index, embedder, passages)
            return new_passages, None, target
        # No specific under-covered entity — WRONG_ENTITY/PARTIAL_COMPARISON
        # diagnoses can route here without a named gap; fall back to a
        # generic broadening retrieval rather than a no-op.
        if memory.has_tried("COMPARE", None):
            return [], None, None
        new_passages = retrieve_simple(query, index, embedder, passages, top_k=RERANK_POOL, bm25=bm25)
        return new_passages, None, None

    if action == "DECOMPOSE":
        new_passages, bridge_ctx = decompose_and_retrieve_multi_hop(
            query, index, embedder, passages, top_k=TOP_K_MULTI, bm25=bm25,
        )
        return new_passages, bridge_ctx, None

    if action == "REWRITE":
        top_text = memory.evidence[0]["text"][:300] if memory.evidence else ""
        rewritten = _reformulate_query(query, top_text) if top_text else query
        if rewritten.strip().lower() == query.strip().lower():
            rewritten = _llm_rewrite_query(query)
        new_passages = retrieve_simple(rewritten, index, embedder, passages,
                                        top_k=RERANK_POOL, bm25=bm25)
        return new_passages, None, rewritten

    return [], None, None


# ─────────────────────────────────────────────
# VERIFICATION ENRICHMENT
# Shared by Stage 3's initial result and every subsequent action's result.
# ─────────────────────────────────────────────
def _verify_and_enrich(query: str, answer: str, reranked_passages: list, query_type: str,
                        verifier_model, verifier_tokenizer, verbose: bool,
                        apply_role_check: bool = True, existing_verification: dict = None):
    """
    Runs Stage 2 V2 verification (verify() directly — the full structured
    report, not the legacy label-only shim Stage 3 uses), then applies two
    Stage-4-level refinements on top:
      - LLM-judge role-mismatch demotion (SUPPORTED -> PARTIAL when the
        answer's entity isn't in the correct semantic role).
      - Yes/No flip: when the answer is a literal yes/no with near-zero
        verifier support, try the opposite literal answer and keep whichever
        the verifier supports more strongly.

    `existing_verification`: when provided (Stage 3's own adaptive_rag_query()
    already computed a verification — including its own role-check demotion —
    for this exact answer), that result is reused directly instead of calling
    verify() again, normalized to this report's shape via normalize_verification().
    Passing apply_role_check=False WITHOUT also passing existing_verification
    would silently re-verify from scratch and discard Stage 3's already-
    detected demotion, which is exactly the bug this parameter exists to
    prevent.

    Returns (answer, verification, label, confidence) — `verification` is
    the full report (overall_status/overall_confidence/support_score/
    coverage/failure_reason/recommended_action/claims), not a bare label.
    """
    if existing_verification is not None:
        verification = normalize_verification(existing_verification)
    else:
        verification = verify(query, answer, reranked_passages, verifier_model)

        if (apply_role_check and verification["overall_status"] == "SUPPORTED"
                and query_type in ("MULTI_HOP", "COMPARISON")):
            if not llm_judge_supported(query, answer, reranked_passages, verbose):
                verification = {
                    **verification,
                    "overall_status": "PARTIAL",
                    "overall_confidence": verification["support_score"],
                    "failure_reason": verification["failure_reason"] or "WRONG_ENTITY",
                    "recommended_action": "COMPARE" if query_type == "COMPARISON" else "REWRITE",
                }
                if verbose:
                    print("[LLM Judge] SUPPORTED → PARTIAL (entity not in correct role)")

    ans_lower = answer.strip().lower().rstrip(".")
    if ans_lower in ("yes", "no") and verification["support_score"] < LOW_SUPPORT_THRESHOLD:
        opposite = "No" if ans_lower == "yes" else "Yes"
        opp_verif = verify(query, opposite, reranked_passages, verifier_model)
        if opp_verif["support_score"] > verification["support_score"]:
            if verbose:
                print(f"[Yes/No flip] '{answer}' → '{opposite}'")
            answer, verification = opposite, opp_verif

    return answer, verification, verification["overall_status"], verification["overall_confidence"]


# ─────────────────────────────────────────────
# CORE AGENTIC LOOP
# ─────────────────────────────────────────────
def agentic_query(query, index, embedder, passages,
                  verifier_model, verifier_tokenizer,
                  verbose=True, query_type_override=None, level_override=None,
                  bm25=None):
    """
    Stage 4: diagnosis-driven agentic reasoning loop.

    Step 0 calls Stage 3's adaptive_rag_query() exactly once — this is the
    agent's initial state. Stage 4 does not classify the query, estimate
    difficulty, select a retrieval strategy, or perform coverage-gated
    expansion itself; Stage 3 already did all of that.

    From there: diagnose -> select one action -> execute -> regenerate ->
    reverify -> repeat, until a stopping criterion fires (confidence reached,
    no new evidence + stable answer, every applicable action exhausted, or
    the hard MAX_ACTIONS ceiling). The final status is derived from the best
    candidate seen across the whole episode — PARTIAL is never silently
    abstained on.
    """
    if verbose:
        print(f"\n{'='*60}\nQuery: {query}\n{'='*60}")

    # ── Step 0: run Stage 3 exactly once — the agent's initial state ──
    s3_result = adaptive_rag_query(
        query, index, embedder, passages,
        verifier_model, verifier_tokenizer, verbose=verbose,
        query_type_override=query_type_override, level_override=level_override,
        bm25=bm25,
    )
    query_type = s3_result["query_type"]
    level      = s3_result["level"]
    reranked   = s3_result["reranked"]

    memory = AgentMemory(query, query_type, level)
    memory.add_evidence(s3_result["retrieved"])

    # Reuse Stage 3's own verification directly — it already includes Stage
    # 3's own role-check demotion — rather than re-verifying from scratch,
    # which would silently discard that demotion. Only enrich with the
    # Yes/No flip on top of it here.
    answer, verification, label, confidence = _verify_and_enrich(
        query, s3_result["answer"], reranked, query_type,
        verifier_model, verifier_tokenizer, verbose,
        existing_verification=s3_result["verification"],
    )
    memory.record_answer("STAGE3", answer, label, confidence, verification)

    v0 = normalize_verification(verification) if verification else {}
    iteration_log = [{
        "iteration": 1, "action": "STAGE3", "query": query, "answer": answer,
        "label": label, "confidence": confidence,
        "num_retrieved": len(s3_result["retrieved"]),
        # Task 6 — full diagnostic trail for thesis-demo debugging:
        "failure_reason": v0.get("failure_reason"),
        "recommended_action": v0.get("recommended_action"),
        "coverage": v0.get("question_coverage_score"),
        "support_score": v0.get("support_score"),
        "answer_type_score": v0.get("answer_type_score"),
        "completeness_score": v0.get("answer_completeness_score"),
        "repair_prompt": None,
    }]
    actions_selected = []   # every action _select_action returned, including the
                            # terminal KEEP/ABSTAIN — the real "why did the loop end" record

    if verbose:
        icon = {"SUPPORTED": "✅", "PARTIAL": "⚠️", "UNSUPPORTED": "❌"}.get(label, "?")
        print(f"\n[Stage 3 initial state] pipeline={s3_result['retrieval_strategy']} "
              f"{icon} {label} (confidence: {confidence:.4f})")

    # ── Diagnosis-driven agent loop ──
    found_new = True     # Stage 3's own retrieval is "new" relative to nothing before it
    action_count = 0

    while action_count < MAX_ACTIONS:
        diagnosis = _diagnose(query, query_type, answer, verification, reranked, memory, found_new)
        action = _select_action(diagnosis, query_type, memory)
        actions_selected.append(action)

        if verbose:
            print(f"\n[Diagnosis] label={diagnosis['label']} conf={diagnosis['confidence']:.3f} "
                  f"coverage={diagnosis['coverage']:.2f} "
                  f"missing={diagnosis['missing_entities'][:3]} "
                  f"type_mismatch={diagnosis['answer_type_mismatch']}")
            print(f"[Decision] {action}")

        if action in ("KEEP", "ABSTAIN"):
            break

        action_count += 1
        new_passages, bridge_ctx, target = _execute_action(
            action, memory, index, embedder, passages, bm25, diagnosis, verbose,
        )
        memory.record_action(action, target)
        found_new = memory.add_evidence(new_passages)

        # Rerank the FULL accumulated evidence so generation always sees the
        # best evidence gathered across the whole episode, not just this action's.
        reranked = rerank_passages(query, memory.evidence, top_k=TOP_K)
        context_passages = list(reranked)
        if bridge_ctx:
            context_passages = [{"title": "Bridge Finding", "text": bridge_ctx}] + context_passages

        # Task 2/4 — diagnosis-specific repair prompt: appended to (never
        # replacing) the question, so Stage 1's own leading-word prompt
        # dispatch inside generate_answer() is untouched. Retrieval above
        # already used the plain query/target — only generation is guided.
        repair_instruction = build_repair_instruction(diagnosis)
        generation_query = f"{query}\n\n{repair_instruction}" if repair_instruction else query

        answer = generate_answer(generation_query, context_passages, query_type=query_type)
        answer, verification, label, confidence = _verify_and_enrich(
            query, answer, reranked, query_type, verifier_model, verifier_tokenizer, verbose,
            apply_role_check=True,
        )
        memory.record_answer(action, answer, label, confidence, verification)

        v_iter = normalize_verification(verification) if verification else {}
        iteration_log.append({
            "iteration": action_count + 1,
            "action": action,
            "query": target if isinstance(target, str) else query,
            "answer": answer, "label": label, "confidence": confidence,
            "num_retrieved": len(new_passages),
            # Task 6 — full diagnostic trail:
            "failure_reason": v_iter.get("failure_reason"),
            "recommended_action": v_iter.get("recommended_action"),
            "coverage": v_iter.get("question_coverage_score"),
            "support_score": v_iter.get("support_score"),
            "answer_type_score": v_iter.get("answer_type_score"),
            "completeness_score": v_iter.get("answer_completeness_score"),
            "repair_prompt": repair_instruction,
        })

        if verbose:
            icon = {"SUPPORTED": "✅", "PARTIAL": "⚠️", "UNSUPPORTED": "❌"}.get(label, "?")
            print(f"[{action}] {answer}  {icon} {label} (confidence: {confidence:.4f})")
            if repair_instruction:
                print(f"[Repair prompt] {repair_instruction}")

    # ── Final status: derived from the best candidate across the WHOLE
    # episode, never a silent abstain on PARTIAL. ──
    best = memory.best_candidate
    if best is None:
        status, final_answer, final_verif = "ABSTAINED", _ABSTAIN_MESSAGE, {}
    elif best["label"] == "SUPPORTED" and best["confidence"] >= CONFIDENCE_THRESHOLD:
        status, final_answer, final_verif = "SUPPORTED", best["answer"], best["verification"]
    elif best["label"] in ("SUPPORTED", "PARTIAL"):
        status, final_answer, final_verif = "BEST_EFFORT", best["answer"], best["verification"]
    else:
        status, final_answer, final_verif = "ABSTAINED", _ABSTAIN_MESSAGE, best["verification"]

    # Task 6 — record the final decision against the last iteration for
    # thesis-demo debugging (every other field on this trail already lives
    # per-iteration; the final decision only exists once, at the end).
    iteration_log[-1]["final_decision"] = status

    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL STATUS: {status}")
        print(f"FINAL ANSWER: {final_answer}")
        print(f"Actions used: {action_count}/{MAX_ACTIONS}")

    return {
        "query":            query,
        "query_type":       query_type,
        "complexity":       level,
        "level":            level,
        "answer":           final_answer,
        "status":           status,
        "iterations":       action_count + 1,   # +1 for Stage 3's own initial pass
        "abstained":        status == "ABSTAINED",
        "verification":     final_verif,
        "iteration_log":    iteration_log,
        "actions_selected": actions_selected,     # every action _select_action chose, in order,
                                                   # including the terminal KEEP/ABSTAIN
        "recovered_via":    (best["action"] if best and best["action"] != "STAGE3" else None),
    }


# ─────────────────────────────────────────────
# EVALUATION — All 4 stages compared, official distractor protocol
# Implements proposal Table 6.2
# ─────────────────────────────────────────────
def evaluate_all_stages(embedder, verifier_model, verifier_tokenizer, num_samples=50):
    """
    Evaluates all 4 pipeline stages under the official distractor protocol:
    every validation question builds its own temporary corpus via
    build_example_corpus(), used identically by all four stages, then
    discarded before the next question.
    """
    from Stage_1_RAG_Pipeline import rag_query as _s1_rag_query

    print(f"\nEvaluating all stages on {num_samples} HotpotQA validation samples...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    metrics = {s: {"em": [], "halluc": [], "abstain": []} for s in
               ["stage1", "stage2", "stage3", "stage4"]}
    results = []
    skipped = 0

    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        if i >= num_samples:
            break

        query = example["question"]
        gold  = example["answer"]
        qtype = example.get("type", "bridge")
        level = example.get("level", "medium")

        ex_index, ex_passages, ex_bm25 = build_example_corpus(example, embedder)
        if ex_index is None:
            skipped += 1
            continue

        # ── Stage 1: basic hybrid RAG ──
        s1_result = _s1_rag_query(query, ex_index, embedder, ex_passages, bm25=ex_bm25)
        s1_answer = s1_result["answer"]
        s1_verif  = verify(query, s1_answer, s1_result["retrieved_passages"], verifier_model)
        s1_halluc = 1 if s1_verif["overall_status"] in ("PARTIAL", "UNSUPPORTED") else 0

        # ── Stage 2: same retrieval as Stage 1 — the verifier label is its contribution ──
        s2_answer, s2_halluc = s1_answer, s1_halluc

        # ── Stage 3: adaptive retrieval planner ──
        s3_result = adaptive_rag_query(
            query, ex_index, embedder, ex_passages, verifier_model, verifier_tokenizer,
            verbose=False, query_type_override=qtype, level_override=level, bm25=ex_bm25,
        )
        s3_answer = s3_result["answer"]
        s3_verif  = s3_result["verification"] or {"label": "UNSUPPORTED"}
        s3_halluc = 1 if s3_verif["label"] in ("PARTIAL", "UNSUPPORTED") else 0

        # ── Stage 4: diagnosis-driven agent ──
        s4_result = agentic_query(
            query, ex_index, embedder, ex_passages, verifier_model, verifier_tokenizer,
            verbose=False, query_type_override=qtype, level_override=level, bm25=ex_bm25,
        )
        s4_answer  = s4_result["answer"]
        s4_abstain = 1 if s4_result.get("abstained") else 0
        # BEST_EFFORT (a PARTIAL best candidate) still counts as a hallucination
        # per the project's Eq. 6.7 — only true SUPPORTED and abstentions don't.
        s4_halluc = 0 if (s4_result["status"] == "SUPPORTED" or s4_result.get("abstained")) else 1

        s1_em, s2_em = exact_match(s1_answer, gold), exact_match(s2_answer, gold)
        s3_em, s4_em = exact_match(s3_answer, gold), exact_match(s4_answer, gold)

        for stage, em, halluc, abstain in [
            ("stage1", s1_em, s1_halluc, 0), ("stage2", s2_em, s2_halluc, 0),
            ("stage3", s3_em, s3_halluc, 0), ("stage4", s4_em, s4_halluc, s4_abstain),
        ]:
            metrics[stage]["em"].append(em)
            metrics[stage]["halluc"].append(halluc)
            metrics[stage]["abstain"].append(abstain)

        results.append({
            "question": query, "gold": gold, "query_type": qtype, "level": level,
            "stage1": {"answer": s1_answer, "em": s1_em, "halluc": s1_halluc},
            "stage2": {"answer": s2_answer, "em": s2_em, "halluc": s2_halluc},
            "stage3": {"answer": s3_answer, "em": s3_em, "halluc": s3_halluc},
            "stage4": {"answer": s4_answer, "em": s4_em, "halluc": s4_halluc,
                      "abstain": s4_abstain, "status": s4_result["status"],
                      "iterations": s4_result["iterations"]},
        })

    def avg(lst): return sum(lst) / len(lst) if lst else 0

    summary = {s: {
        "exact_match": avg(metrics[s]["em"]),
        "hallucination_rate": avg(metrics[s]["halluc"]),
        "abstention_rate": avg(metrics[s]["abstain"]),
    } for s in ["stage1", "stage2", "stage3", "stage4"]}

    print(f"\n{'='*70}")
    print(f"Full Stage Comparison ({len(results)} samples"
          f"{f', {skipped} skipped' if skipped else ''})")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Stage1':>10} {'Stage2':>10} {'Stage3':>10} {'Stage4':>10}")
    print(f"{'-'*65}")
    for metric, label in [
        ("exact_match", "Exact Match"),
        ("hallucination_rate", "Hallucination Rate"),
        ("abstention_rate", "Abstention Rate"),
    ]:
        row = f"{label:<25}"
        for stage in ["stage1", "stage2", "stage3", "stage4"]:
            row += f" {summary[stage][metric]:>10.4f}"
        print(row)

    s1_h, s4_h = summary["stage1"]["hallucination_rate"], summary["stage4"]["hallucination_rate"]
    reduction = (s1_h - s4_h) / s1_h * 100 if s1_h > 0 else 0
    print(f"\nHallucination reduced from {s1_h:.4f} (Stage1) to {s4_h:.4f} (Stage4) "
          f"= {reduction:.1f}% reduction")
    print(f"Stage 4 abstention rate: {summary['stage4']['abstention_rate']:.4f}")

    with open("stage4_full_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print("\nResults saved → stage4_full_results.json")
    return summary


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run full 4-stage evaluation")
    parser.add_argument("--samples", type=int, default=50, help="Number of evaluation samples")
    args = parser.parse_args()

    print("Stage 4: Agentic Reasoning Loop")
    print(f"Device: {DEVICE.upper()}\n")

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    if not os.path.exists(VERIFIER_PATH):
        print("No verifier found. Run: python Stage_2_Verifier_GPU.py --mode train")
        sys.exit(1)
    verifier_model, verifier_tokenizer = load_verifier(VERIFIER_PATH)
    print("All components loaded.\n")

    if args.eval:
        evaluate_all_stages(embedder, verifier_model, verifier_tokenizer, num_samples=args.samples)
    else:
        print("Loading HotpotQA train + validation splits (distractor)...")
        train_dataset = load_dataset("hotpot_qa", "distractor", split="train")
        val_dataset   = load_dataset("hotpot_qa", "distractor", split="validation")
        combined_dataset = concatenate_datasets([train_dataset, val_dataset])
        train_size = len(train_dataset)

        print("=== HARA — Stage 4: Agentic Reasoning Demo ===")
        print(f"{len(combined_dataset)} questions loaded "
              f"({train_size} train + {len(val_dataset)} validation).")
        print(f"Max actions: {MAX_ACTIONS} | Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"Enter an example index (0-{len(combined_dataset)-1}), 'eval', or 'quit'.\n")

        while True:
            cmd = input("Example index / 'eval' / 'quit': ").strip()
            if cmd.lower() == "quit":
                break
            elif cmd.lower() == "eval":
                evaluate_all_stages(embedder, verifier_model, verifier_tokenizer, num_samples=args.samples)
            elif cmd.isdigit():
                idx = int(cmd)
                if not (0 <= idx < len(combined_dataset)):
                    print(f"Index out of range (0-{len(combined_dataset)-1}).")
                    continue
                origin = "train" if idx < train_size else "validation"
                example = combined_dataset[idx]
                print(f"\n[{origin}] Gold answer: {example['answer']}")

                ex_index, ex_passages, ex_bm25 = build_example_corpus(example, embedder)
                if ex_index is None:
                    print("This example has an empty context — skipping.")
                    continue

                result = agentic_query(
                    example["question"], ex_index, embedder, ex_passages,
                    verifier_model, verifier_tokenizer, verbose=True,
                    query_type_override=example.get("type"),
                    level_override=example.get("level"),
                    bm25=ex_bm25,
                )
                print(f"\n{'='*60}")
                print(f"FINAL STATUS: {result['status']}")
                print(f"FINAL ANSWER: {result['answer']}")
                print(f"Actions used: {result['iterations']}")
            else:
                print("Enter an example index, 'eval', or 'quit'.")

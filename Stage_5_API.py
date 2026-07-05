"""
Stage 5: FastAPI Backend
Exposes the RAG pipeline stages as a REST API for the React frontend.

Retrieval methodology: official HotpotQA distractor protocol, matching the
redesigned Stage 1. There is no global knowledge base and no persistent
FAISS index — every request is matched against its own HotpotQA example at
lookup time, and Stage 1 builds a temporary per-question corpus from that
example's own ~10-article context (see Stage_1_RAG_Pipeline.build_example_corpus).
Only questions that exist verbatim in the loaded HotpotQA train+validation
splits can be answered; this API is a benchmark harness, not an open-domain
service.

Usage:
    python Stage_5_API.py
    # API available at http://localhost:8001
    # React frontend connects via proxy at http://localhost:5173
"""

import json
import os
import re
import string
import sys
from collections import Counter
from contextlib import asynccontextmanager
from typing import Literal

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from datasets import load_dataset

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

from Stage_1_RAG_Pipeline import (
    build_example_corpus,
    rag_query as run_s1,
    EMBED_MODEL,
)
from Stage_3_Adaptive_Retrieval import adaptive_rag_query as run_s3
from Stage_4_Agentic_Loop import agentic_query as run_s4
from Stage_2_Verifier import load_verifier, verify, legacy_scores, VERIFIER_PATH


# ── Shared pipeline state (loaded once at startup) ──
_pipeline: dict = {}


# ── Live evaluation helpers ──

def _normalize(text: str) -> str:
    """Aggressive normalization for EM/F1 scoring (strips articles + punctuation)."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    return ' '.join(text.split())


def _normalize_question(text: str) -> str:
    """Light normalization for the question -> HotpotQA-example lookup key.

    Deliberately lighter than _normalize() above: that function strips
    articles/punctuation for EM/F1 scoring, which would raise collision risk
    if reused as a lookup key across ~98k distinct questions.
    """
    return text.lower().strip()


def _live_metrics(predicted: str, gold: str) -> dict:
    """Compute EM, Precision, Recall, F1 for one prediction vs gold answer."""
    p = _normalize(predicted)
    g = _normalize(gold)
    em = int(p == g)
    p_toks = p.split()
    g_toks = g.split()
    if not p_toks or not g_toks:
        return {"em": em, "precision": 0.0, "recall": 0.0, "f1": 0.0, "gold_answer": gold}
    common = sum((Counter(p_toks) & Counter(g_toks)).values())
    if common == 0:
        return {"em": em, "precision": 0.0, "recall": 0.0, "f1": 0.0, "gold_answer": gold}
    prec = common / len(p_toks)
    rec  = common / len(g_toks)
    f1   = 2 * prec * rec / (prec + rec)
    return {
        "em":           em,
        "precision":    round(prec, 4),
        "recall":       round(rec,  4),
        "f1":           round(f1,   4),
        "gold_answer":  gold,
    }


def _lookup_example(question: str) -> dict:
    """Finds the HotpotQA example matching `question` (after light
    normalization). Raises HTTPException(404) if the question isn't part of
    the loaded benchmark — this API only answers questions from the loaded
    HotpotQA train+validation splits, never open-domain queries, so there is
    never a "retrieve over unrelated examples" fallback.
    """
    example = _pipeline["question_lookup"].get(_normalize_question(question))
    if example is None:
        raise HTTPException(
            status_code=404,
            detail="This question is not part of the loaded HotpotQA benchmark "
                   "(train + validation splits). Only benchmark questions can be answered.",
        )
    return example


def _build_request_corpus(example: dict):
    """Builds the temporary per-question retrieval corpus for one request,
    exactly as the redesigned Stage 1 does: build_example_corpus() scoped to
    this example's own ~10-article context only. No global corpus is ever
    built, and no whole-dataset embedding happens — only this one question's
    handful of chunks get embedded (and Stage 1's own process-lifetime
    embedding cache means repeated/overlapping articles across requests are
    not re-embedded).
    """
    embedder = _pipeline["embedder"]
    ex_index, ex_passages, ex_bm25 = build_example_corpus(example, embedder)
    if ex_index is None:
        raise HTTPException(
            status_code=422,
            detail="This HotpotQA example has an empty context and cannot be retrieved from.",
        )
    return ex_index, embedder, ex_passages, ex_bm25


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No global FAISS index / no whole-corpus embedding is built here — only
    # a plain in-memory question -> example lookup. The temporary per-question
    # retrieval corpus is built later, per request, in _build_request_corpus().
    print("Loading HotpotQA train split...")
    train_data = load_dataset("hotpot_qa", "distractor", split="train")
    print("Loading HotpotQA validation split...")
    val_data = load_dataset("hotpot_qa", "distractor", split="validation")

    print("\nLoaded:")
    print(f"  Train:      {len(train_data):,}")
    print(f"  Validation: {len(val_data):,}")
    print(f"  Total:      {len(train_data) + len(val_data):,}\n")

    question_lookup: dict = {}
    for split_data in (train_data, val_data):
        for ex in split_data:
            question_lookup[_normalize_question(ex["question"])] = {
                "id": ex["id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "type": ex["type"],
                "level": ex["level"],
                "context": ex["context"],
                "supporting_facts": ex["supporting_facts"],
            }
    _pipeline["question_lookup"] = question_lookup
    print(f"Question lookup ready — {len(question_lookup):,} unique questions indexed.\n")

    print(f"Loading embedding model: {EMBED_MODEL}")
    _pipeline["embedder"] = SentenceTransformer(EMBED_MODEL)

    print("Loading verifier (Stage 2 V2 — pretrained NLI, no local checkpoint needed)...")
    try:
        vm, vt = load_verifier(VERIFIER_PATH)
        _pipeline["verifier_model"]     = vm
        _pipeline["verifier_tokenizer"] = vt
        print("Verifier loaded.")
    except Exception:
        _pipeline["verifier_model"]     = None
        _pipeline["verifier_tokenizer"] = None
        print("Verifier failed to load (see traceback above) — stages 2-4 will be unavailable.")

    print("API ready at http://localhost:8000\n")
    yield
    _pipeline.clear()


app = FastAPI(title="HARA — Hallucination-Aware Retrieval Agent API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──
class ChatRequest(BaseModel):
    question: str
    stage: Literal["stage1", "stage2", "stage3", "stage4"] = "stage4"


class ChatResponse(BaseModel):
    answer:   str
    stage:    str
    metadata: dict


# ── Endpoints ──
@app.get("/health")
def health():
    return {
        "status":           "ok",
        "verifier_loaded":  _pipeline.get("verifier_model") is not None,
        "index_vectors":    0,  # no persistent global index under the distractor protocol
        "questions_loaded": len(_pipeline.get("question_lookup", {})),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    example = _lookup_example(req.question)
    ex_index, embedder, ex_passages, ex_bm25 = _build_request_corpus(example)
    gold_ans = example["answer"]
    meta = {"type": example["type"], "level": example["level"]}

    if req.stage == "stage1":
        result = run_s1(req.question, ex_index, embedder, ex_passages, bm25=ex_bm25)
        return ChatResponse(
            answer=result["answer"],
            stage="stage1",
            metadata={
                "eval":             _live_metrics(result["answer"], gold_ans),
                "retrieved_titles": [p["title"] for p in result["retrieved_passages"]],
            },
        )

    elif req.stage == "stage2":
        vm = _pipeline.get("verifier_model")
        if vm is None:
            raise HTTPException(
                status_code=503,
                detail="Verifier model not loaded — check server startup logs.",
            )
        # Stage 2 = Stage 1 retrieval + generation, then the verifier checks
        # the answer against evidence. It does NOT run its own separate
        # retrieval — doing so would retrieve different passages and
        # generate a different answer, making Stage 2 incomparable to
        # Stage 1 in the paper's Table 6.2.
        s1_result   = run_s1(req.question, ex_index, embedder, ex_passages, bm25=ex_bm25)
        answer      = s1_result["answer"]
        s1_passages = s1_result["retrieved_passages"]

        report        = verify(req.question, answer, s1_passages, vm)
        support_score = report["support_score"]

        return ChatResponse(
            answer=answer,
            stage="stage2",
            metadata={
                # Backward-compatible fields existing clients already read:
                "label":            report["overall_status"],
                "confidence":       round(float(report["overall_confidence"]), 4),
                "support_score":    round(float(support_score), 4),
                "scores":           legacy_scores(report["overall_status"], support_score),
                # Additive — failure_reason/recommended_action/claims/etc:
                "failure_reason":     report.get("failure_reason"),
                "recommended_action": report.get("recommended_action"),
                "verification":     report,
                "eval":             _live_metrics(answer, gold_ans),
                "retrieved_titles": [p["title"] for p in s1_passages],
            },
        )

    elif req.stage == "stage3":
        vm = _pipeline.get("verifier_model")
        vt = _pipeline.get("verifier_tokenizer")
        if vm is None:
            raise HTTPException(
                status_code=503,
                detail="Verifier model not loaded — check server startup logs.",
            )
        # Stage 3 is now redesigned to match this architecture directly: it
        # receives this request's own temporary per-question corpus AND the
        # matching per-question bm25 object, so its adaptive retrieval planner
        # runs full hybrid (dense + BM25 + RRF) retrieval scoped to just this
        # question — no global BM25 fallback involved anywhere.
        result = run_s3(req.question, ex_index, embedder, ex_passages, vm, vt, verbose=False,
                        query_type_override=meta.get("type"),
                        level_override=meta.get("level"),
                        bm25=ex_bm25)
        # Stage 3 now returns Stage 2 V2's native report directly
        # (overall_status/overall_confidence/support_score/failure_reason/
        # recommended_action/claims/question_intent — no legacy shape left).
        verif  = result.get("verification") or {}
        scores = legacy_scores(verif.get("overall_status", "UNSUPPORTED"), verif.get("support_score", 0.0))
        return ChatResponse(
            answer=result["answer"],
            stage="stage3",
            metadata={
                "query_type":         result["query_type"],
                "level":              result["level"],
                "retrieval_strategy": result["retrieval_strategy"],
                "retrieval_params":   result["retrieval_params"],
                "num_retrieved":      result["num_retrieved"],
                # Backward-compatible fields existing clients already read:
                "label":              verif.get("overall_status", "UNKNOWN"),
                "confidence":         round(float(verif.get("overall_confidence", 0)), 4),
                "support_score":      round(float(verif.get("support_score", 0)), 4),
                "scores":             scores,
                # Additive — failure_reason/recommended_action/claims/etc:
                "failure_reason":     verif.get("failure_reason"),
                "recommended_action": verif.get("recommended_action"),
                "verification":       verif,
                "eval":               _live_metrics(result["answer"], gold_ans),
                "retrieved_titles":   [p["title"] for p in result["retrieved"][:10]],
            },
        )

    else:  # stage4
        vm = _pipeline.get("verifier_model")
        vt = _pipeline.get("verifier_tokenizer")
        if vm is None:
            raise HTTPException(
                status_code=503,
                detail="Verifier model not loaded — check server startup logs.",
            )

        # Stage 4 now calls Stage 3 internally exactly once as its own initial
        # state, so there's no need to run Stage 3 separately here first (the
        # old code did, then patched the result back in if Stage 4 "downgraded"
        # it — that compensating logic is now obsolete: Stage 4's own
        # best-candidate tracking spans its whole episode, including that
        # internal Stage 3 call, so it can never return worse than Stage 3
        # would have on its own). Running Stage 3 twice here would have just
        # doubled the retrieval/generation cost for every Stage 4 request.
        result = run_s4(req.question, ex_index, embedder, ex_passages, vm, vt, verbose=False,
                        query_type_override=meta.get("type"),
                        level_override=meta.get("level"),
                        bm25=ex_bm25)

        # result["verification"] is already Stage 2 V2's report shape (or {}
        # when every action in the episode was UNSUPPORTED and the agent
        # abstained from the start) — synthesize the backward-compatible
        # label/confidence/scores fields from it for existing clients.
        verif = result["verification"] or {}
        support_score = verif.get("support_score", 0.0)
        scores = legacy_scores(verif.get("overall_status", "UNSUPPORTED"), support_score)
        iter_log = [
            {
                "iteration":    it["iteration"],
                "query":        it["query"],
                "answer":       it["answer"],
                "label":        it["label"],
                "confidence":   float(it["confidence"]),
                "num_retrieved":it["num_retrieved"],
            }
            for it in result["iteration_log"]
        ]

        return ChatResponse(
            answer=result["answer"],
            stage="stage4",
            metadata={
                "status":              result["status"],
                "query_type":          result["query_type"],
                "iterations":          int(result["iterations"]),
                # Backward-compatible fields existing clients already read:
                "verification_scores": scores,
                "confidence":          round(float(verif.get("overall_confidence", 0.0)), 4),
                "support_score":       round(float(support_score), 4),
                # Additive — the full structured report, including claims/coverage:
                "failure_reason":      verif.get("failure_reason"),
                "recommended_action":  verif.get("recommended_action"),
                "verification":        verif,
                "actions_selected":    result.get("actions_selected", []),
                "recovered_via":       result.get("recovered_via"),
                "iteration_log":       iter_log,
                "eval":                _live_metrics(result["answer"], gold_ans),
            },
        )


@app.get("/results")
def get_results():
    """Serve pre-computed evaluation results for the UI results table."""
    path = "evaluation_results.json"
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="No evaluation results found. Run: python Stage_6_Evaluation.py --samples 50",
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    uvicorn.run("Stage_5_API:app", host="0.0.0.0", port=8001, reload=True)

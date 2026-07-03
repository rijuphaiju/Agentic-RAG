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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

from datasets import load_dataset

from Stage_1_RAG_Pipeline import (
    build_example_corpus,
    rag_query as run_s1,
    EMBED_MODEL,
)
from Stage_3_Adaptive_Retrieval import adaptive_rag_query as run_s3
from Stage_4_Agentic_Loop import agentic_query as run_s4
from Stage_2_Verifier_GPU import load_verifier, verify, build_verify_context, VERIFIER_PATH


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

    print("Loading verifier...")
    if os.path.exists(VERIFIER_PATH):
        vm, vt = load_verifier(VERIFIER_PATH)
        _pipeline["verifier_model"]     = vm
        _pipeline["verifier_tokenizer"] = vt
        print("Verifier loaded.")
    else:
        _pipeline["verifier_model"]     = None
        _pipeline["verifier_tokenizer"] = None
        print("Verifier not found — Stage 4 will be unavailable.")

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
        vt = _pipeline.get("verifier_tokenizer")
        if vm is None:
            raise HTTPException(
                status_code=503,
                detail="Verifier model not loaded. Run: python Stage_2_Verifier_GPU.py --mode train",
            )
        # Stage 2 = Stage 1 retrieval + generation, then the verifier classifies
        # the answer.  It does NOT run its own separate retrieval — doing so would
        # retrieve different passages and generate a different answer, making
        # Stage 2 incomparable to Stage 1 in the paper's Table 6.2.
        s1_result   = run_s1(req.question, ex_index, embedder, ex_passages, bm25=ex_bm25)
        answer      = s1_result["answer"]
        s1_passages = s1_result["retrieved_passages"]

        ctx          = build_verify_context(s1_passages, answer)
        verification = verify(ctx, answer, vm, vt, question=req.question)
        scores       = {k: round(float(v), 4) for k, v in verification["scores"].items()}

        return ChatResponse(
            answer=answer,
            stage="stage2",
            metadata={
                "label":            verification["label"],
                "confidence":       round(float(verification["confidence"]), 4),
                "scores":           scores,
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
                detail="Verifier model not loaded. Run: python Stage_2_Verifier_GPU.py --mode train",
            )
        # Minimum compatibility layer (Stage 3 has not been redesigned yet):
        # adaptive_rag_query() still expects a shared (index, embedder, passages)
        # triple, so we hand it this request's own temporary per-question corpus
        # instead of a global one — its internal retrieval stays correctly scoped
        # to just this question. One known limitation: any retrieve_hybrid() call
        # inside Stage 3 that doesn't pass bm25 explicitly will run dense-only for
        # this request, since the module-level BM25 fallback is intentionally
        # never populated here (populating it globally would not be
        # request-safe for a concurrent API server). Acceptable for now — Stage 3
        # will be redesigned separately.
        result = run_s3(req.question, ex_index, embedder, ex_passages, vm, vt, verbose=False,
                        query_type_override=meta.get("type"),
                        level_override=meta.get("level"))
        verif  = result.get("verification") or {}
        scores = {k: round(float(v), 4) for k, v in verif.get("scores", {}).items()}
        return ChatResponse(
            answer=result["answer"],
            stage="stage3",
            metadata={
                "query_type":         result["query_type"],
                "level":              result["level"],
                "retrieval_strategy": result["retrieval_strategy"],
                "retrieval_params":   result["retrieval_params"],
                "num_retrieved":      result["num_retrieved"],
                "label":              verif.get("label", "UNKNOWN"),
                "confidence":         round(float(verif.get("confidence", 0)), 4),
                "scores":             scores,
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
                detail="Verifier model not loaded. Run: python Stage_2_Verifier_GPU.py --mode train",
            )

        # Run Stage 3 first so Stage 4 starts from Stage 3's retrieved passages
        # and verified answer instead of re-retrieving from scratch. Same
        # minimum-compatibility-layer note as the stage3 branch above applies.
        s3_result = run_s3(req.question, ex_index, embedder, ex_passages, vm, vt, verbose=False,
                           query_type_override=meta.get("type"),
                           level_override=meta.get("level"))
        s3_label   = (s3_result.get("verification") or {}).get("label", "UNSUPPORTED")
        s3_answer  = s3_result.get("answer", "")
        s3_qtype   = s3_result.get("query_type")

        # Every question reaching this point already matched a loaded HotpotQA
        # example (see _lookup_example, which 404s otherwise) — unlike the old
        # open-domain design, there is no "question outside the dataset"
        # abstention branch needed here anymore.
        if s3_label == "SUPPORTED":
            result = run_s4(req.question, ex_index, embedder, ex_passages, vm, vt, verbose=False,
                            query_type_override=s3_qtype or meta.get("type"),
                            level_override=meta.get("level"))
            # Stage 4 did not independently verify — fall back to Stage 3's answer
            s4_label = (result.get("verification") or {}).get("label", "UNSUPPORTED")
            if s4_label != "SUPPORTED":
                result["answer"]       = s3_answer
                result["status"]       = "SUPPORTED"
                result["abstained"]    = False
                result["verification"] = s3_result.get("verification", {})
        else:
            result = run_s4(req.question, ex_index, embedder, ex_passages, vm, vt, verbose=False,
                            query_type_override=s3_qtype or meta.get("type"),
                            level_override=meta.get("level"))

        # Convert numpy/torch scalars → native Python floats for JSON serialization
        # result["verification"] may be {} when all iterations were UNSUPPORTED (abstain)
        raw_scores = result["verification"].get("scores", {"SUPPORTED": 0.0, "PARTIAL": 0.0, "UNSUPPORTED": 1.0})
        scores = {k: float(v) for k, v in raw_scores.items()}
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
                "verification_scores": scores,
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

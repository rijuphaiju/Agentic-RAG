"""
Stage 5: FastAPI Backend
Exposes the RAG pipeline stages as a REST API for the React frontend.

Usage:
    python Stage_5_API.py
    # API available at http://localhost:8001
    # React frontend connects via proxy at http://localhost:5173
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Literal

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from datasets import load_dataset

from Stage_1_RAG_Pipeline import (
    load_faiss_index, rag_query as run_s1,
    build_faiss_index, load_hotpotqa_passages,
    INDEX_PATH, PASSAGES_PATH,
)
from Stage_2_RAG_Pipeline import rag_query_stage2 as run_s2
from Stage_3_Adaptive_Retrieval import adaptive_rag_query as run_s3
from Stage_4_Agentic_Loop import agentic_query as run_s4
from Stage_2_Verifier_GPU import load_verifier, verify, VERIFIER_PATH


# ── Shared pipeline state (loaded once at startup) ──
_pipeline: dict = {}


# ── Live evaluation helpers ──
import re, string
from collections import Counter

def _normalize(text):
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    return ' '.join(text.split())

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading FAISS index...")
    if os.path.exists(INDEX_PATH) and os.path.exists(PASSAGES_PATH):
        index, embedder, passages = load_faiss_index()
    else:
        passages = load_hotpotqa_passages()
        index, embedder, passages = build_faiss_index(passages)

    _pipeline["index"]    = index
    _pipeline["embedder"] = embedder
    _pipeline["passages"] = passages

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

    print("Building HotpotQA question-type/level/answer lookup...")
    type_lookup: dict[str, dict] = {}
    for split in ("train", "validation"):
        for ex in load_dataset("hotpot_qa", "distractor", split=split):
            type_lookup[ex["question"].lower().strip()] = {
                "type":   ex["type"],
                "level":  ex["level"],
                "answer": ex["answer"],   # gold answer for live EM/F1 computation
            }
    _pipeline["type_lookup"] = type_lookup
    print(f"Type/level/answer lookup ready — {len(type_lookup):,} questions indexed.")

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
        "status":          "ok",
        "verifier_loaded": _pipeline.get("verifier_model") is not None,
        "index_vectors":   _pipeline["index"].ntotal if "index" in _pipeline else 0,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    idx = _pipeline["index"]
    emb = _pipeline["embedder"]
    pas = _pipeline["passages"]

    # ── Gold answer + live metrics for any HotpotQA question ──
    q_meta   = _pipeline.get("type_lookup", {}).get(req.question.lower().strip()) or {}
    gold_ans = q_meta.get("answer")

    if req.stage == "stage1":
        result = run_s1(req.question, idx, emb, pas)
        return ChatResponse(
            answer=result["answer"],
            stage="stage1",
            metadata={
                "eval":             _live_metrics(result["answer"], gold_ans) if gold_ans else None,
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
        s1_result  = run_s1(req.question, idx, emb, pas)
        answer     = s1_result["answer"]
        s1_passages = s1_result["retrieved_passages"]

        from Stage_2_Verifier_GPU import build_verify_context, verify
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
                "eval":             _live_metrics(answer, gold_ans) if gold_ans else None,
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
        meta = _pipeline.get("type_lookup", {}).get(req.question.lower().strip()) or {}
        result = run_s3(req.question, idx, emb, pas, vm, vt, verbose=False,
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
                "eval":               _live_metrics(result["answer"], gold_ans) if gold_ans else None,
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
        meta = _pipeline.get("type_lookup", {}).get(req.question.lower().strip()) or {}
        result = run_s4(req.question, idx, emb, pas, vm, vt, verbose=False,
                        query_type_override=meta.get("type"),
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
                "eval":                _live_metrics(result["answer"], gold_ans) if gold_ans else None,
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

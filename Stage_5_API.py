"""
Stage 5: FastAPI Backend
Exposes the RAG pipeline stages as a REST API for the React frontend.

Usage:
    python Stage_5_API.py
    # API available at http://localhost:8000
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

from Stage_1_RAG_Pipeline import (
    load_faiss_index, rag_query as run_s1,
    build_faiss_index, load_hotpotqa_passages,
    INDEX_PATH, PASSAGES_PATH,
)
from Stage_2_RAG_Pipeline import rag_query_stage2 as run_s2
from Stage_4_Agentic_Loop import agentic_query as run_s4
from Stage_2_Verifier_GPU import load_verifier, VERIFIER_PATH


# ── Shared pipeline state (loaded once at startup) ──
_pipeline: dict = {}


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
    stage: Literal["stage1", "stage2", "stage4"] = "stage4"


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

    if req.stage == "stage1":
        result = run_s1(req.question, idx, emb, pas)
        return ChatResponse(
            answer=result["answer"],
            stage="stage1",
            metadata={
                "retrieved_titles": [p["title"] for p in result["retrieved_passages"]],
            },
        )

    elif req.stage == "stage2":
        result = run_s2(req.question, idx, emb, pas, verbose=False)
        return ChatResponse(
            answer=result["answer"],
            stage="stage2",
            metadata={
                "faithfulness":     round(result["faithfulness_score"], 3),
                "confidence":       round(result["confidence_score"], 3),
                "fallback_used":    result["used_fallback"],
                "retrieved_titles": [p["title"] for p in result["retrieved_passages"]],
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
        result = run_s4(req.question, idx, emb, pas, vm, vt, verbose=False)

        # Convert numpy/torch scalars → native Python floats for JSON serialization
        scores = {k: float(v) for k, v in result["verification"]["scores"].items()}
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

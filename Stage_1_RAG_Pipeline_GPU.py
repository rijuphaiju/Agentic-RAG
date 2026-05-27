"""
Stage 1: Basic RAG Pipeline — GPU Accelerated
==============================================
Project: HARA — Hallucination-Aware Retrieval Agent
Dataset: HotpotQA
LLM: Ollama (local)
Retrieval: FAISS (CPU index) + sentence-transformers (GPU embeddings)

GPU changes vs original:
  - SentenceTransformer loads onto CUDA automatically
  - Batch size 64 → 256 for faster GPU throughput
  - encode() calls pass device=DEVICE explicitly
  - DEVICE auto-detects CUDA, falls back to CPU gracefully

Metrics added vs original:
  - Token-level F1 alongside Exact Match in evaluate()
  - Results JSON now includes both em and f1 per sample
"""

import json
import re
import string
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ollama
import pickle
import os
import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
OLLAMA_MODEL  = "llama3.2"
TOP_K         = 10
MAX_PASSAGES  = 90000
INDEX_PATH    = "faiss_index.bin"
PASSAGES_PATH = "passages.pkl"

# GPU auto-detection — no changes needed when switching machines
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256 if DEVICE == "cuda" else 64

print(f"[Device] Using: {DEVICE.upper()}"
      + (f" — {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else " (CPU fallback)"))


# ─────────────────────────────────────────────
# STEP 1: LOAD & PREPARE HOTPOTQA PASSAGES
# ─────────────────────────────────────────────
def load_hotpotqa_passages(max_passages=MAX_PASSAGES):
    print("Loading HotpotQA dataset...")
    dataset = load_dataset("hotpot_qa", "distractor", split="train")

    passages, seen_titles = [], set()
    for example in tqdm(dataset, desc="Extracting passages"):
        for title, sentences in zip(
            example["context"]["title"],
            example["context"]["sentences"]
        ):
            if title not in seen_titles:
                seen_titles.add(title)
                passages.append({"title": title, "text": " ".join(sentences)})
        if len(passages) >= max_passages:
            break

    print(f"Extracted {len(passages)} unique passages.")
    return passages


# ─────────────────────────────────────────────
# STEP 2: BUILD / LOAD FAISS INDEX
# ─────────────────────────────────────────────
def build_faiss_index(passages, embed_model_name=EMBED_MODEL):
    """
    Embeds all passages on GPU and stores them in a FAISS index.
    GPU: SentenceTransformer on DEVICE, batch_size=256.
    Index stays on CPU — sufficient for 90k vectors (~5ms search).
    """
    print(f"Loading embedding model: {embed_model_name} on {DEVICE.upper()}")
    embedder = SentenceTransformer(embed_model_name, device=DEVICE)

    print(f"Generating embeddings (batch_size={BATCH_SIZE}, device={DEVICE.upper()})...")
    texts = [p["text"] for p in passages]
    embeddings = embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        device=DEVICE,
        convert_to_numpy=True,
    )
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(PASSAGES_PATH, "wb") as f:
        pickle.dump(passages, f)

    print(f"FAISS index built and saved. Total vectors: {index.ntotal}")
    return index, embedder, passages


def load_faiss_index(embed_model_name=EMBED_MODEL):
    print("Loading existing FAISS index from disk...")
    index = faiss.read_index(INDEX_PATH)
    with open(PASSAGES_PATH, "rb") as f:
        passages = pickle.load(f)
    embedder = SentenceTransformer(embed_model_name, device=DEVICE)
    print(f"Loaded index with {index.ntotal} vectors. Embedder on {DEVICE.upper()}.")
    return index, embedder, passages


# ─────────────────────────────────────────────
# STEP 3: RETRIEVE RELEVANT PASSAGES
# ─────────────────────────────────────────────
def retrieve(query, index, embedder, passages, top_k=TOP_K):
    """GPU: query embedding runs on DEVICE."""
    query_vec = embedder.encode(
        [query], convert_to_numpy=True, device=DEVICE
    ).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(passages):
            results.append({
                "title": passages[idx]["title"],
                "text":  passages[idx]["text"],
                "score": float(score),
            })
    return results


# ─────────────────────────────────────────────
# STEP 4: GENERATE ANSWER WITH OLLAMA
# ─────────────────────────────────────────────
def generate_answer(query, retrieved_passages, model=OLLAMA_MODEL):
    context = "\n\n".join([
        f"[{i+1}] {p['title']}:\n{p['text']}"
        for i, p in enumerate(retrieved_passages)
    ])
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()


# ─────────────────────────────────────────────
# STEP 5: FULL RAG PIPELINE
# ─────────────────────────────────────────────
def rag_query(query, index, embedder, passages):
    print(f"\n{'='*60}\nQuery: {query}\n{'='*60}")

    retrieved = retrieve(query, index, embedder, passages)
    print(f"\nTop {len(retrieved)} retrieved passages:")
    for i, p in enumerate(retrieved):
        print(f"  [{i+1}] {p['title']} (score: {p['score']:.4f})")

    print("\nGenerating answer...")
    answer = generate_answer(query, retrieved)
    print(f"\nAnswer: {answer}")

    return {"query": query, "retrieved_passages": retrieved, "answer": answer}


# ─────────────────────────────────────────────
# STEP 6: METRICS
# ─────────────────────────────────────────────
def normalize_answer(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text.strip()


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    """
    Token-level F1 — standard HotpotQA metric.
    Gives partial credit when predicted answer shares tokens
    with the gold answer, even without an exact string match.
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ─────────────────────────────────────────────
# STEP 7: EVALUATE ON HOTPOTQA VALIDATION SET
# ─────────────────────────────────────────────
def evaluate(index, embedder, passages, num_samples=100):
    """
    Evaluates on HotpotQA validation set.
    Reports Exact Match + Token F1 — the Stage 1 baseline.
    """
    print(f"\nEvaluating on {num_samples} HotpotQA validation samples...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    results   = []
    em_scores = []
    f1_scores = []

    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        if i >= num_samples:
            break

        query       = example["question"]
        gold_answer = example["answer"]

        retrieved   = retrieve(query, index, embedder, passages)
        pred_answer = generate_answer(query, retrieved)

        em = exact_match(pred_answer, gold_answer)
        f1 = f1_score(pred_answer, gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        results.append({
            "question":         query,
            "gold":             gold_answer,
            "predicted":        pred_answer,
            "exact_match":      em,
            "f1":               round(f1, 4),
            "retrieved_titles": [p["title"] for p in retrieved],
        })

    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print(f"\n{'='*60}")
    print(f"Stage 1 Baseline Results ({num_samples} samples) — GPU accelerated")
    print(f"{'='*60}")
    print(f"Exact Match : {avg_em:.4f}  ({avg_em*100:.1f}%)")
    print(f"Token F1    : {avg_f1:.4f}  ({avg_f1*100:.1f}%)")
    print(f"(Stage 2+ should improve both scores)")

    with open("stage1_results.json", "w") as f:
        json.dump({
            "exact_match": avg_em,
            "f1":          avg_f1,
            "results":     results,
        }, f, indent=2)
    print("Results saved → stage1_results.json")
    return results, avg_em, avg_f1


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists(INDEX_PATH) and os.path.exists(PASSAGES_PATH):
        index, embedder, passages = load_faiss_index()
    else:
        passages = load_hotpotqa_passages()
        index, embedder, passages = build_faiss_index(passages)

    print("\n=== Stage 1: Basic RAG Demo (GPU) ===")
    print("Commands: 'eval' → run evaluation | 'quit' → exit\n")

    while True:
        query = input("Enter your question: ").strip()
        if query.lower() == "quit":
            break
        elif query.lower() == "eval":
            evaluate(index, embedder, passages, num_samples=50)
        elif query:
            rag_query(query, index, embedder, passages)

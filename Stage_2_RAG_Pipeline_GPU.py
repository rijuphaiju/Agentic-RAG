"""
Stage 2: RAG Pipeline with Hallucination Detection — GPU Accelerated
=====================================================================
Project: HARA — Hallucination-Aware Retrieval Agent
Dataset: HotpotQA
LLM: Ollama (local)
Retrieval: FAISS (CPU index) + sentence-transformers (GPU embeddings)

GPU changes vs original Stage 2:
  - SentenceTransformer loads onto CUDA automatically
  - Batch size 64 → 256 for faster GPU throughput
  - encode() calls pass device=DEVICE explicitly

Metrics added vs original Stage 2:
  - Token-level F1 alongside Exact Match in evaluate_stage2()
  - Stage comparison now shows both EM and F1 deltas vs Stage 1
  - Results JSON includes f1 per sample
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

# GPU auto-detection
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256 if DEVICE == "cuda" else 64

# Stage 2 thresholds
FAITHFULNESS_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD   = 0.4

print(f"[Device] Using: {DEVICE.upper()}"
      + (f" — {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else " (CPU fallback)"))


# ─────────────────────────────────────────────
# STEP 1–3: LOAD PASSAGES / BUILD / LOAD INDEX / RETRIEVE
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


def build_faiss_index(passages, embed_model_name=EMBED_MODEL):
    print(f"Loading embedding model: {embed_model_name} on {DEVICE.upper()}")
    embedder = SentenceTransformer(embed_model_name, device=DEVICE)
    print(f"Generating embeddings (batch_size={BATCH_SIZE}, device={DEVICE.upper()})...")
    texts = [p["text"] for p in passages]
    embeddings = embedder.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True,
        device=DEVICE, convert_to_numpy=True,
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
# STEP 4: GENERATE ANSWER
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
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()


# ─────────────────────────────────────────────
# STEP 5: FAITHFULNESS SCORING
# ─────────────────────────────────────────────
def score_faithfulness(query, answer, retrieved_passages, model=OLLAMA_MODEL):
    """
    Asks the LLM to rate how well the answer is grounded in retrieved context.
    Returns (score: float 0–1, reason: str).

    Rubric:
      1.0 → every claim directly supported by context
      0.7 → most claims supported, minor inferences present
      0.4 → some claims supported, significant unsupported content
      0.0 → answer contradicts or ignores context entirely
    """
    context = "\n\n".join([
        f"[{i+1}] {p['title']}:\n{p['text']}"
        for i, p in enumerate(retrieved_passages)
    ])
    prompt = f"""You are an impartial fact-checker evaluating whether an AI answer is grounded in its source context.

Context (the ONLY source of truth):
{context}

Question: {query}
Answer to evaluate: {answer}

Rate the faithfulness of the answer to the provided context on a scale from 0.0 to 1.0:
  1.0 = Every claim is directly supported by the context
  0.7 = Most claims are supported; minor inference present
  0.4 = Some claims supported; significant unsupported content exists
  0.0 = Answer contradicts context or introduces outside information

Respond in this exact format (no other text):
SCORE: <float between 0.0 and 1.0>
REASON: <one sentence explanation>"""

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    raw      = response["message"]["content"].strip()
    score    = _parse_score(raw, "SCORE")
    reason   = _parse_field(raw, "REASON")
    return score, reason


# ─────────────────────────────────────────────
# STEP 6: CONFIDENCE ESTIMATION
# ─────────────────────────────────────────────
def estimate_confidence(query, answer, retrieved_passages, model=OLLAMA_MODEL):
    """
    Asks the LLM how confident it is the answer is correct.
    Separate from faithfulness — a grounded answer can still be
    uncertain if the context is vague or incomplete.
    Returns (confidence: float 0–1, rationale: str).
    """
    context_titles = ", ".join([p["title"] for p in retrieved_passages])
    prompt = f"""You previously answered a question using retrieved Wikipedia passages.
Evaluate your own confidence in the correctness of your answer.

Question: {query}
Your answer: {answer}
Retrieved sources: {context_titles}

How confident are you that your answer is correct?
  1.0 = Very confident — context clearly supports this answer
  0.7 = Moderately confident — context supports it but has gaps
  0.4 = Low confidence — context is ambiguous or incomplete
  0.0 = Not confident — context doesn't support this answer

Respond in this exact format (no other text):
CONFIDENCE: <float between 0.0 and 1.0>
RATIONALE: <one sentence>"""

    response   = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    raw        = response["message"]["content"].strip()
    confidence = _parse_score(raw, "CONFIDENCE")
    rationale  = _parse_field(raw, "RATIONALE")
    return confidence, rationale


# ─────────────────────────────────────────────
# STEP 7: FALLBACK GENERATION
# ─────────────────────────────────────────────
def generate_fallback_answer(query, retrieved_passages, model=OLLAMA_MODEL):
    """
    Conservative re-prompt triggered when faithfulness < threshold.
    Forces inline citations to anchor the model to the context.
    """
    context = "\n\n".join([
        f"[{i+1}] {p['title']}:\n{p['text']}"
        for i, p in enumerate(retrieved_passages)
    ])
    prompt = f"""You are a careful assistant. Your previous answer may have included information not found in the source context.

Re-answer the question using ONLY information explicitly stated in the numbered passages below.
- Cite passage numbers like [1], [2] when using information from them.
- If the context is insufficient, respond exactly: "The provided context does not contain enough information to answer this question."
- Do NOT infer, guess, or use outside knowledge.

Context:
{context}

Question: {query}

Careful answer (cite sources):"""

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _parse_score(text, key):
    match = re.search(rf"{key}:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return 0.5   # neutral fallback if parsing fails


def _parse_field(text, key):
    match = re.search(rf"{key}:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ─────────────────────────────────────────────
# STEP 8: METRICS
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
# STEP 9: FULL STAGE 2 PIPELINE
# ─────────────────────────────────────────────
def rag_query_stage2(query, index, embedder, passages, verbose=True):
    """
    Full Stage 2 pipeline (GPU accelerated):
      Query → Embed (GPU) → Retrieve → Generate → Score Faithfulness
           → Estimate Confidence → Fallback if needed → Return
    """
    if verbose:
        print(f"\n{'='*60}\nQuery: {query}\n{'='*60}")

    # 1. Retrieve
    retrieved = retrieve(query, index, embedder, passages)
    if verbose:
        print(f"\nTop {len(retrieved)} retrieved passages:")
        for i, p in enumerate(retrieved):
            print(f"  [{i+1}] {p['title']} (score: {p['score']:.4f})")

    # 2. Generate initial answer
    if verbose: print("\nGenerating answer...")
    answer = generate_answer(query, retrieved)
    if verbose: print(f"\nInitial Answer: {answer}")

    # 3. Score faithfulness
    if verbose: print("\nScoring faithfulness...")
    faithfulness, faith_reason = score_faithfulness(query, answer, retrieved)
    if verbose: print(f"  Faithfulness: {faithfulness:.2f} — {faith_reason}")

    # 4. Estimate confidence
    if verbose: print("Estimating confidence...")
    confidence, conf_rationale = estimate_confidence(query, answer, retrieved)
    if verbose: print(f"  Confidence:   {confidence:.2f} — {conf_rationale}")

    # 5. Fallback if faithfulness too low
    used_fallback = False
    if faithfulness < FAITHFULNESS_THRESHOLD:
        if verbose:
            print(f"\n⚠  Faithfulness {faithfulness:.2f} < {FAITHFULNESS_THRESHOLD}. Triggering fallback...")
        answer        = generate_fallback_answer(query, retrieved)
        used_fallback = True
        if verbose: print(f"\nFallback Answer: {answer}")

    # 6. Flag low confidence
    is_low_confidence = confidence < CONFIDENCE_THRESHOLD
    if verbose:
        if is_low_confidence:
            print(f"\n⚠  Low confidence ({confidence:.2f}). Answer may be uncertain.")
        print(f"\n{'='*60}")
        print(f"Final Answer: {answer}")
        print(f"Faithfulness: {faithfulness:.2f} | Confidence: {confidence:.2f} | Fallback: {used_fallback}")
        print('='*60)

    return {
        "query":               query,
        "answer":              answer,
        "faithfulness_score":  faithfulness,
        "faithfulness_reason": faith_reason,
        "confidence_score":    confidence,
        "confidence_rationale":conf_rationale,
        "used_fallback":       used_fallback,
        "low_confidence_flag": is_low_confidence,
        "retrieved_passages":  retrieved,
    }


# ─────────────────────────────────────────────
# STEP 10: EVALUATION (Stage 2)
# ─────────────────────────────────────────────
def evaluate_stage2(index, embedder, passages, num_samples=100):
    """
    Evaluates Stage 2 pipeline on HotpotQA validation set.
    Reports: Exact Match, Token F1, avg faithfulness, avg confidence,
    fallback rate, low-confidence rate.
    Auto-compares both EM and F1 against stage1_results.json.
    """
    print(f"\nEvaluating Stage 2 on {num_samples} HotpotQA validation samples...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    results        = []
    em_scores      = []
    f1_scores      = []
    faith_scores   = []
    conf_scores    = []
    fallback_count = 0
    low_conf_count = 0

    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        if i >= num_samples:
            break

        query       = example["question"]
        gold_answer = example["answer"]

        result = rag_query_stage2(query, index, embedder, passages, verbose=False)

        em = exact_match(result["answer"], gold_answer)
        f1 = f1_score(result["answer"], gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)
        faith_scores.append(result["faithfulness_score"])
        conf_scores.append(result["confidence_score"])
        if result["used_fallback"]:      fallback_count += 1
        if result["low_confidence_flag"]:low_conf_count += 1

        results.append({
            "question":           query,
            "gold":               gold_answer,
            "predicted":          result["answer"],
            "exact_match":        em,
            "f1":                 round(f1, 4),
            "faithfulness":       result["faithfulness_score"],
            "faith_reason":       result["faithfulness_reason"],
            "confidence":         result["confidence_score"],
            "conf_rationale":     result["confidence_rationale"],
            "used_fallback":      result["used_fallback"],
            "low_confidence":     result["low_confidence_flag"],
            "retrieved_titles":   [p["title"] for p in result["retrieved_passages"]],
        })

    avg_em         = sum(em_scores)    / len(em_scores)
    avg_f1         = sum(f1_scores)    / len(f1_scores)
    avg_faith      = sum(faith_scores) / len(faith_scores)
    avg_conf       = sum(conf_scores)  / len(conf_scores)
    fallback_rate  = fallback_count    / num_samples
    low_conf_rate  = low_conf_count    / num_samples

    print(f"\n{'='*60}")
    print(f"Stage 2 Results ({num_samples} samples) — GPU accelerated")
    print(f"{'='*60}")
    print(f"Exact Match Score    : {avg_em:.4f}  ({avg_em*100:.1f}%)")
    print(f"Token F1 Score       : {avg_f1:.4f}  ({avg_f1*100:.1f}%)")
    print(f"Avg Faithfulness     : {avg_faith:.4f}")
    print(f"Avg Confidence       : {avg_conf:.4f}")
    print(f"Fallback Rate        : {fallback_rate:.2%}  ({fallback_count}/{num_samples})")
    print(f"Low-Confidence Rate  : {low_conf_rate:.2%}  ({low_conf_count}/{num_samples})")

    # Compare with Stage 1 if available
    if os.path.exists("stage1_results.json"):
        with open("stage1_results.json") as f:
            stage1 = json.load(f)
        s1_em = stage1.get("exact_match")
        s1_f1 = stage1.get("f1")

        print(f"\n{'─'*40}")
        print("Stage 1 → Stage 2 comparison:")
        if s1_em is not None:
            d_em = avg_em - s1_em
            print(f"  EM  : {s1_em:.4f} → {avg_em:.4f}  ({'+' if d_em>=0 else ''}{d_em:.4f})")
        if s1_f1 is not None:
            d_f1 = avg_f1 - s1_f1
            print(f"  F1  : {s1_f1:.4f} → {avg_f1:.4f}  ({'+' if d_f1>=0 else ''}{d_f1:.4f})")
        print(f"{'─'*40}")

    output = {
        "exact_match":         avg_em,
        "f1":                  avg_f1,
        "avg_faithfulness":    avg_faith,
        "avg_confidence":      avg_conf,
        "fallback_rate":       fallback_rate,
        "low_confidence_rate": low_conf_rate,
        "results":             results,
    }
    with open("stage2_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved → stage2_results.json")
    return results, output


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists(INDEX_PATH) and os.path.exists(PASSAGES_PATH):
        index, embedder, passages = load_faiss_index()
    else:
        passages = load_hotpotqa_passages()
        index, embedder, passages = build_faiss_index(passages)

    print("\n=== Stage 2: RAG + Faithfulness Scoring (GPU) ===")
    print("Commands: 'eval' → run evaluation | 'quit' → exit\n")

    while True:
        query = input("Enter your question: ").strip()
        if query.lower() == "quit":
            break
        elif query.lower() == "eval":
            evaluate_stage2(index, embedder, passages, num_samples=50)
        elif query:
            rag_query_stage2(query, index, embedder, passages, verbose=True)

"""
Stage 2: RAG Pipeline with Faithfulness Verifier
=================================================
Project: HARA — Hallucination-Aware Retrieval Agent
Dataset: HotpotQA
LLM: Ollama (local)
Retrieval: FAISS + sentence-transformers

What's new in Stage 2 vs Stage 1:
  - Faithfulness Verifier: DistilBERT transformer classifies answers as
    SUPPORTED / PARTIAL / UNSUPPORTED based on retrieved context
  - Adds hallucination detection over the Stage 1 baseline
"""

import json
import re
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ollama
import pickle
import os

from Stage_2_Verifier_GPU import verify

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EMBED_MODEL       = "all-MiniLM-L6-v2"
OLLAMA_MODEL      = "llama3.2"          # or "mistral"
TOP_K             = 10
MAX_PASSAGES      = 90000
INDEX_PATH        = "faiss_index.bin"
PASSAGES_PATH     = "passages.pkl"

# Stage 2 thresholds
# 0.5 caused ~90% fallback rate on HotpotQA because multi-hop context rarely
# contains a verbatim sentence that perfectly mirrors the answer.
# Lowered to 0.15 — only trigger fallback for clearly unfaithful outputs.
FAITHFULNESS_THRESHOLD = 0.15
CONFIDENCE_THRESHOLD   = 0.4   # below this → mark as low-confidence


# ─────────────────────────────────────────────
# STEP 1–3: UNCHANGED FROM STAGE 1
# (load passages, build/load FAISS index, retrieve)
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
    print(f"Loading embedding model: {embed_model_name}")
    embedder = SentenceTransformer(embed_model_name)
    print("Generating embeddings...")
    texts = [p["text"] for p in passages]
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(PASSAGES_PATH, "wb") as f:
        pickle.dump(passages, f)
    print(f"FAISS index built. Total vectors: {index.ntotal}")
    return index, embedder, passages


def load_faiss_index(embed_model_name=EMBED_MODEL):
    print("Loading existing FAISS index from disk...")
    index = faiss.read_index(INDEX_PATH)
    with open(PASSAGES_PATH, "rb") as f:
        passages = pickle.load(f)
    embedder = SentenceTransformer(embed_model_name)
    print(f"Loaded index with {index.ntotal} vectors.")
    return index, embedder, passages


def retrieve(query, index, embedder, passages, top_k=TOP_K):
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(passages):
            results.append({
                "title": passages[idx]["title"],
                "text":  passages[idx]["text"],
                "score": float(score)
            })
    return results


# ─────────────────────────────────────────────
# STEP 4: GENERATE ANSWER (same as Stage 1)
# ─────────────────────────────────────────────
def generate_answer(query, retrieved_passages, model=OLLAMA_MODEL):
    context = "\n\n".join([
        f"[{i+1}] {p['title']}:\n{p['text']}"
        for i, p in enumerate(retrieved_passages)
    ])
    prompt = f"""You are a precise question-answering assistant.
Answer using ONLY the provided context. Always give your best answer — never refuse.
Give a direct answer in 1-2 sentences. Do NOT start with 'Based on' or 'According to'.

Context:
{context}

Question: {query}

Final Answer:"""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": 80},
    )
    return response["message"]["content"].strip()


# ─────────────────────────────────────────────
# STEP 5 (NEW): FAITHFULNESS SCORING
# ─────────────────────────────────────────────
def score_faithfulness(query, answer, retrieved_passages, model=OLLAMA_MODEL):
    """
    Asks the LLM to rate how well the answer is supported by the retrieved context.
    Returns a float in [0.0, 1.0] and a brief explanation.

    Scoring rubric sent to the LLM:
      1.0 → Every claim in the answer is directly supported by the context.
      0.7 → Most claims are supported; minor inferences present.
      0.4 → Some claims are supported, but significant unsupported content exists.
      0.0 → Answer contradicts or ignores the context entirely.
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

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response["message"]["content"].strip()

    # Parse score from response
    score = _parse_score(raw, key="SCORE")
    reason = _parse_field(raw, key="REASON")
    return score, reason


# ─────────────────────────────────────────────
# STEP 6 (NEW): CONFIDENCE ESTIMATION
# ─────────────────────────────────────────────
def estimate_confidence(query, answer, retrieved_passages, model=OLLAMA_MODEL):
    """
    Asks the LLM how confident it is that its answer is correct.
    This is separate from faithfulness — an answer can be faithful (grounded)
    but still uncertain if the context itself is vague or incomplete.

    Returns a float in [0.0, 1.0] and a brief rationale.
    """
    context_titles = ", ".join([p["title"] for p in retrieved_passages])

    prompt = f"""You previously answered a question using retrieved Wikipedia passages.
Evaluate your own confidence in the correctness of your answer.

Question: {query}
Your answer: {answer}
Retrieved sources: {context_titles}

How confident are you that your answer is correct?
  1.0 = Very confident — the context clearly and unambiguously supports this answer
  0.7 = Moderately confident — context supports it but has gaps
  0.4 = Low confidence — context is ambiguous or incomplete
  0.0 = Not confident — context doesn't really support this answer

Respond in this exact format (no other text):
CONFIDENCE: <float between 0.0 and 1.0>
RATIONALE: <one sentence>"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response["message"]["content"].strip()

    confidence = _parse_score(raw, key="CONFIDENCE")
    rationale  = _parse_field(raw, key="RATIONALE")
    return confidence, rationale


# ─────────────────────────────────────────────
# STEP 7 (NEW): FALLBACK GENERATION
# ─────────────────────────────────────────────
def generate_fallback_answer(query, retrieved_passages, model=OLLAMA_MODEL):
    """
    Conservative re-prompt triggered when faithfulness score is too low.
    Instructs the model to be explicit about uncertainty and to cite passages.
    This reduces hallucination by anchoring the model more strictly to context.
    """
    context = "\n\n".join([
        f"[{i+1}] {p['title']}:\n{p['text']}"
        for i, p in enumerate(retrieved_passages)
    ])

    prompt = f"""You are a careful assistant. Re-answer the question using ONLY the numbered passages below.
- State the answer directly and cite the passage like [1] or [2].
- If the context is limited, give your best answer from what is available — never refuse to answer.
- Keep the answer to 1-2 sentences.

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
# HELPERS: Parse LLM scoring output
# ─────────────────────────────────────────────
def _parse_score(text, key):
    """Extract a float value after a KEY: prefix. Returns 0.5 on failure."""
    match = re.search(rf"{key}:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return max(0.0, min(1.0, val))   # clamp to [0, 1]
    return 0.5   # neutral fallback if parsing fails


def _parse_field(text, key):
    """Extract text after a KEY: prefix on a line."""
    match = re.search(rf"{key}:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ─────────────────────────────────────────────
# STEP 8 (NEW): FULL STAGE 2 PIPELINE
# ─────────────────────────────────────────────
def rag_query_stage2(query, index, embedder, passages,
                     verifier_model=None, verifier_tokenizer=None,
                     verbose=True):
    """
    Stage 2 pipeline (report Section 6.3.2):
      Query → Retrieve (FAISS) → Generate (LLM) → Verify (DistilBERT)
                                                 → SUPPORTED / PARTIAL / UNSUPPORTED
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

    # 1. Retrieve
    retrieved = retrieve(query, index, embedder, passages)
    if verbose:
        print(f"\nTop {len(retrieved)} retrieved passages:")
        for i, p in enumerate(retrieved):
            print(f"  [{i+1}] {p['title']} (score: {p['score']:.4f})")

    # 2. Generate answer
    if verbose:
        print("\nGenerating answer...")
    answer = generate_answer(query, retrieved)
    if verbose:
        print(f"\nAnswer: {answer}")

    # 3. Verify with DistilBERT faithfulness verifier
    if verifier_model is None or verifier_tokenizer is None:
        raise RuntimeError(
            "Verifier not loaded. Run: python Stage_2_Verifier_GPU.py --mode train"
        )
    context = " ".join(p["text"] for p in retrieved)
    verification = verify(context, answer, verifier_model, verifier_tokenizer)
    label      = verification["label"]
    confidence = verification["confidence"]
    scores     = verification["scores"]

    if verbose:
        icon = {"SUPPORTED": "✅", "PARTIAL": "⚠️", "UNSUPPORTED": "❌"}.get(label, "?")
        print(f"\nVerification: {icon} {label} (confidence: {confidence:.4f})")
        print(f"  Scores → SUPPORTED: {scores['SUPPORTED']:.4f} | "
              f"PARTIAL: {scores['PARTIAL']:.4f} | "
              f"UNSUPPORTED: {scores['UNSUPPORTED']:.4f}")
        print('='*60)

    return {
        "query":             query,
        "answer":            answer,
        "label":             label,
        "confidence":        confidence,
        "scores":            scores,
        "retrieved_passages": retrieved,
    }


# ─────────────────────────────────────────────
# STEP 9: EVALUATION (Stage 2)
# ─────────────────────────────────────────────
def normalize_answer(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(__import__('string').punctuation)}]", "", text)
    return text.strip()


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def evaluate_stage2(index, embedder, passages, num_samples=100):
    """
    Evaluate Stage 2 pipeline on HotpotQA validation set.
    Tracks: Exact Match, average faithfulness, average confidence,
    fallback rate, and low-confidence rate.
    Compare results against stage1_results.json if it exists.
    """
    print(f"\nEvaluating Stage 2 on {num_samples} HotpotQA validation samples...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    results    = []
    em_scores  = []
    faith_scores = []
    conf_scores  = []
    fallback_count = 0
    low_conf_count = 0

    for i, example in enumerate(tqdm(dataset)):
        if i >= num_samples:
            break

        query       = example["question"]
        gold_answer = example["answer"]

        result = rag_query_stage2(query, index, embedder, passages, verbose=False)

        em = exact_match(result["answer"], gold_answer)
        em_scores.append(em)
        faith_scores.append(result["faithfulness_score"])
        conf_scores.append(result["confidence_score"])
        if result["used_fallback"]:
            fallback_count += 1
        if result["low_confidence_flag"]:
            low_conf_count += 1

        results.append({
            "question":          query,
            "gold":              gold_answer,
            "predicted":         result["answer"],
            "exact_match":       em,
            "faithfulness":      result["faithfulness_score"],
            "faith_reason":      result["faithfulness_reason"],
            "confidence":        result["confidence_score"],
            "conf_rationale":    result["confidence_rationale"],
            "used_fallback":     result["used_fallback"],
            "low_confidence":    result["low_confidence_flag"],
            "retrieved_titles":  [p["title"] for p in result["retrieved_passages"]],
        })

    # Aggregate metrics
    avg_em    = sum(em_scores) / len(em_scores)
    avg_faith = sum(faith_scores) / len(faith_scores)
    avg_conf  = sum(conf_scores) / len(conf_scores)
    fallback_rate  = fallback_count / num_samples
    low_conf_rate  = low_conf_count / num_samples

    # Print report
    print(f"\n{'='*60}")
    print(f"Stage 2 Results ({num_samples} samples)")
    print(f"{'='*60}")
    print(f"Exact Match Score:       {avg_em:.4f}  ({avg_em*100:.1f}%)")
    print(f"Avg Faithfulness Score:  {avg_faith:.4f}")
    print(f"Avg Confidence Score:    {avg_conf:.4f}")
    print(f"Fallback Rate:           {fallback_rate:.2%}  ({fallback_count}/{num_samples})")
    print(f"Low-Confidence Rate:     {low_conf_rate:.2%}  ({low_conf_count}/{num_samples})")

    # Compare with Stage 1 if available
    if os.path.exists("stage1_results.json"):
        with open("stage1_results.json") as f:
            stage1 = json.load(f)
        s1_em = stage1.get("exact_match", None)
        if s1_em is not None:
            delta = avg_em - s1_em
            print(f"\nStage 1 EM:  {s1_em:.4f} ({s1_em*100:.1f}%)")
            print(f"Stage 2 EM:  {avg_em:.4f} ({avg_em*100:.1f}%)")
            sign = "+" if delta >= 0 else ""
            print(f"ΔExact Match: {sign}{delta:.4f} ({sign}{delta*100:.1f}%)")

    # Save results
    output = {
        "exact_match":       avg_em,
        "avg_faithfulness":  avg_faith,
        "avg_confidence":    avg_conf,
        "fallback_rate":     fallback_rate,
        "low_confidence_rate": low_conf_rate,
        "results":           results
    }
    with open("stage2_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to stage2_results.json")

    return results, output


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Build or load FAISS index (reuses Stage 1 index if present)
    if os.path.exists(INDEX_PATH) and os.path.exists(PASSAGES_PATH):
        index, embedder, passages = load_faiss_index()
    else:
        passages = load_hotpotqa_passages()
        index, embedder, passages = build_faiss_index(passages)

    print("\n=== Stage 2: RAG with Faithfulness Scoring & Hallucination Detection ===")
    print("Commands: 'eval' → run evaluation | 'quit' → exit\n")

    while True:
        query = input("Enter your question: ").strip()
        if query.lower() == "quit":
            break
        elif query.lower() == "eval":
            evaluate_stage2(index, embedder, passages, num_samples=50)
        elif query:
            rag_query_stage2(query, index, embedder, passages, verbose=True)

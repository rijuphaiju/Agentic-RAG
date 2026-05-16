"""
Stage 1: Basic RAG Pipeline
===========================
Project: Reducing Hallucinations in Agentic RAG Systems
Dataset: HotpotQA
LLM: Ollama (local)
Retrieval: FAISS + sentence-transformers
"""

import json
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ollama
import pickle
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # lightweight, fast, good quality
OLLAMA_MODEL = "llama3.2"           # or "mistral" — run: ollama pull llama3.2
TOP_K = 10                           # number of passages to retrieve
MAX_PASSAGES = 90000                 # limit dataset passages for quick testing
INDEX_PATH = "faiss_index.bin"
PASSAGES_PATH = "passages.pkl"


# ─────────────────────────────────────────────
# STEP 1: LOAD & PREPARE HOTPOTQA PASSAGES
# ─────────────────────────────────────────────
def load_hotpotqa_passages(max_passages=MAX_PASSAGES):
    """
    Loads HotpotQA and extracts Wikipedia passages from supporting facts.
    Each passage = one article's context sentences joined together.
    """
    print("Loading HotpotQA dataset...")
    dataset = load_dataset("hotpot_qa", "distractor", split="train")

    passages = []
    seen_titles = set()

    for example in tqdm(dataset, desc="Extracting passages"):
        for title, sentences in zip(
            example["context"]["title"],
            example["context"]["sentences"]
        ):
            if title not in seen_titles:
                seen_titles.add(title)
                passage_text = " ".join(sentences)
                passages.append({
                    "title": title,
                    "text": passage_text
                })
        if len(passages) >= max_passages:
            break

    print(f"Extracted {len(passages)} unique passages.")
    return passages


# ─────────────────────────────────────────────
# STEP 2: BUILD FAISS INDEX
# ─────────────────────────────────────────────
def build_faiss_index(passages, embed_model_name=EMBED_MODEL):
    """
    Embeds all passages and stores them in a FAISS index for fast similarity search.
    Saves index and passages to disk so you don't rebuild every run.
    """
    print(f"Loading embedding model: {embed_model_name}")
    embedder = SentenceTransformer(embed_model_name)

    print("Generating embeddings (this may take a few minutes)...")
    texts = [p["text"] for p in passages]
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine sim after normalization
    index.add(embeddings)

    # Save to disk
    faiss.write_index(index, INDEX_PATH)
    with open(PASSAGES_PATH, "wb") as f:
        pickle.dump(passages, f)

    print(f"FAISS index built and saved. Total vectors: {index.ntotal}")
    return index, embedder, passages


def load_faiss_index(embed_model_name=EMBED_MODEL):
    """Load pre-built index from disk."""
    print("Loading existing FAISS index from disk...")
    index = faiss.read_index(INDEX_PATH)
    with open(PASSAGES_PATH, "rb") as f:
        passages = pickle.load(f)
    embedder = SentenceTransformer(embed_model_name)
    print(f"Loaded index with {index.ntotal} vectors.")
    return index, embedder, passages


# ─────────────────────────────────────────────
# STEP 3: RETRIEVE RELEVANT PASSAGES
# ─────────────────────────────────────────────
def retrieve(query, index, embedder, passages, top_k=TOP_K):
    """
    Embeds the query, searches FAISS, returns top-k passages.
    This is the core retrieval step in your RAG pipeline.
    """
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(passages):
            results.append({
                "title": passages[idx]["title"],
                "text": passages[idx]["text"],
                "score": float(score)
            })
    return results


# ─────────────────────────────────────────────
# STEP 4: GENERATE ANSWER WITH OLLAMA
# ─────────────────────────────────────────────
def generate_answer(query, retrieved_passages, model=OLLAMA_MODEL):
    """
    Sends the query + retrieved context to local Ollama LLM.
    This is the generation step — no verification yet (Stage 1 baseline).
    """
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
# STEP 5: FULL RAG PIPELINE (Stage 1)
# ─────────────────────────────────────────────
def rag_query(query, index, embedder, passages):
    """
    Full Stage 1 pipeline:
    Query → Embed → Retrieve → Generate → Return Answer
    No verification. This is the hallucination baseline.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    # Retrieve
    retrieved = retrieve(query, index, embedder, passages)
    print(f"\nTop {len(retrieved)} retrieved passages:")
    for i, p in enumerate(retrieved):
        print(f"  [{i+1}] {p['title']} (score: {p['score']:.4f})")

    # Generate
    print("\nGenerating answer...")
    answer = generate_answer(query, retrieved)
    print(f"\nAnswer: {answer}")

    return {
        "query": query,
        "retrieved_passages": retrieved,
        "answer": answer
    }


# ─────────────────────────────────────────────
# STEP 6: EVALUATE ON HOTPOTQA TEST SET
# ─────────────────────────────────────────────
def normalize_answer(text):
    """Lowercase + remove punctuation for Exact Match scoring."""
    import re, string
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text.strip()


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def evaluate(index, embedder, passages, num_samples=100):
    """
    Runs evaluation on HotpotQA validation set.
    Computes Exact Match — the baseline hallucination reference (Stage 1).
    """
    print(f"\nEvaluating on {num_samples} HotpotQA validation samples...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    results = []
    em_scores = []

    for i, example in enumerate(tqdm(dataset)):
        if i >= num_samples:
            break

        query = example["question"]
        gold_answer = example["answer"]

        retrieved = retrieve(query, index, embedder, passages)
        pred_answer = generate_answer(query, retrieved)

        em = exact_match(pred_answer, gold_answer)
        em_scores.append(em)

        results.append({
            "question": query,
            "gold": gold_answer,
            "predicted": pred_answer,
            "exact_match": em,
            "retrieved_titles": [p["title"] for p in retrieved]
        })

    avg_em = sum(em_scores) / len(em_scores)
    print(f"\n{'='*60}")
    print(f"Stage 1 Baseline Results ({num_samples} samples)")
    print(f"{'='*60}")
    print(f"Exact Match Score: {avg_em:.4f} ({avg_em*100:.1f}%)")
    print(f"(This is your hallucination baseline — Stage 2+ should improve this)")

    # Save results
    with open("stage1_results.json", "w") as f:
        json.dump({"exact_match": avg_em, "results": results}, f, indent=2)
    print("Results saved to stage1_results.json")

    return results, avg_em


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Build or load index
    if os.path.exists(INDEX_PATH) and os.path.exists(PASSAGES_PATH):
        index, embedder, passages = load_faiss_index()
    else:
        passages = load_hotpotqa_passages()
        index, embedder, passages = build_faiss_index(passages)

    # Interactive demo
    print("\n=== Stage 1: Basic RAG Demo ===")
    print("Type 'eval' to run evaluation, 'quit' to exit.\n")

    while True:
        query = input("Enter your question: ").strip()
        if query.lower() == "quit":
            break
        elif query.lower() == "eval":
            evaluate(index, embedder, passages, num_samples=50)
        elif query:
            rag_query(query, index, embedder, passages)

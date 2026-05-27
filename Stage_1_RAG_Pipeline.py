"""
Stage 1: Basic RAG Pipeline
===========================
Project: Reducing Hallucinations in Agentic RAG Systems
Dataset: HotpotQA
LLM: Ollama (local)
Retrieval: FAISS + sentence-transformers
"""

import json
import re
import string
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
import ollama
import pickle
import os

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EMBED_MODEL    = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL   = "llama3.2"
TOP_K          = 10
MAX_PASSAGES   = 150000
INDEX_PATH     = "faiss_index.bin"
PASSAGES_PATH  = "passages.pkl"

_reranker = None   # lazy-loaded cross-encoder
_bm25     = None   # lazy-loaded BM25 index (built in-memory, no disk write)


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
# STEP 2b: BM25 HYBRID RETRIEVAL
# ─────────────────────────────────────────────

def init_bm25(passages):
    """
    Build an in-memory BM25 index over all passages.
    Call once after loading/building the FAISS index.
    No disk write — avoids storage overhead.
    """
    global _bm25
    if _bm25 is not None:
        return
    if not _BM25_AVAILABLE:
        print("rank-bm25 not installed — hybrid retrieval disabled.")
        print("  Install with: pip install rank-bm25")
        return
    print("Building BM25 index (in-memory, one-time)...")
    tokenized = [p["text"].lower().split() for p in tqdm(passages, desc="BM25")]
    _bm25 = _BM25Okapi(tokenized)
    print(f"BM25 index ready ({len(passages):,} passages).")


def retrieve_hybrid(query, index, embedder, passages, top_k=TOP_K, alpha=0.5):
    """
    BM25 + dense fusion retrieval.

    alpha controls the blend weight:
      alpha=0.5  equal weight (default — good general balance)
      alpha=0.7  favour BM25 (better for exact-name lookups)
      alpha=0.3  favour dense (better for semantic paraphrase)

    Falls back silently to dense-only when BM25 is not initialised.
    """
    # ── Dense retrieval — wider pool for candidate fusion ──
    dense_k = min(top_k * 4, len(passages))
    q_vec   = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    d_scores, d_indices = index.search(q_vec, dense_k)

    d_arr  = d_scores[0]
    d_min, d_max = float(d_arr.min()), float(d_arr.max())
    d_norm = (d_arr - d_min) / (d_max - d_min + 1e-9)

    if _bm25 is not None:
        # ── BM25 scores over full corpus ──
        bm25_scores = np.array(_bm25.get_scores(query.lower().split()), dtype="float32")
        bm25_max    = float(bm25_scores.max())
        bm25_norm   = bm25_scores / (bm25_max + 1e-9)

        # ── Weighted hybrid score: alpha*BM25 + (1-alpha)*dense ──
        candidate_scores: dict[int, float] = {}
        for score, idx in zip(d_norm, d_indices[0]):
            iidx = int(idx)
            if iidx < len(passages):
                candidate_scores[iidx] = (
                    alpha * float(bm25_norm[iidx]) + (1 - alpha) * float(score)
                )

        # Expand with top BM25 hits not already in dense pool
        for iidx in np.argsort(bm25_scores)[::-1][:top_k * 4]:
            iidx = int(iidx)
            if iidx not in candidate_scores and iidx < len(passages):
                candidate_scores[iidx] = alpha * float(bm25_norm[iidx])

        sorted_items = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    else:
        # Dense-only fallback
        sorted_items = [
            (int(idx), float(s))
            for s, idx in zip(d_norm, d_indices[0])
            if int(idx) < len(passages)
        ][:top_k]

    return [
        {"title": passages[idx]["title"], "text": passages[idx]["text"],
         "score": score, "hop": 1}
        for idx, score in sorted_items
    ]


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
# STEP 4: ANSWER EXTRACTION + GENERATION
# ─────────────────────────────────────────────
def _extract_answer(raw_text):
    """
    Post-processes the LLM output to return a short, clean answer.
    Looks for 'Final Answer:' first; falls back to stripping hedging phrases.
    """
    text = raw_text.strip()

    # Primary: pull out everything after "Final Answer:" on that line
    m = re.search(r'(?:Final Answer|Answer)\s*:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        ans = m.group(1).strip().rstrip('.,')
        if ans:
            text = ans

    # Fallback: strip common hedging prefixes
    hedges = (
        "based on the context provided, ",
        "based on the provided context, ",
        "based on the information provided, ",
        "based on the context, ",
        "according to the context, ",
        "according to the provided context, ",
        "according to the passages, ",
        "from the context, ",
        "the context indicates that ",
        "the context states that ",
        "the passages indicate that ",
        "based on the available information, ",
    )
    lower = text.lower()
    for hedge in hedges:
        if lower.startswith(hedge):
            text = text[len(hedge):]
            text = text[0].upper() + text[1:] if text else text
            break

    # Allow up to 2 sentences; trim anything beyond that
    first = text.find('. ')
    if first != -1:
        second = text.find('. ', first + 2)
        if second != -1:
            text = text[:second + 1].strip()

    # Strip any trailing "Explanation:" section the LLM appended
    exp_match = re.search(r'[.\s]*\bExplanation\b\s*:', text, re.IGNORECASE)
    if exp_match:
        trimmed = text[:exp_match.start()].strip()
        if trimmed:
            text = trimmed

    # For yes/no answers: reduce to just "Yes" or "No".
    # HotpotQA gold is "yes"/"no"; including the explanation sentence kills EM.
    yesno = re.match(r'^(Yes|No)\b', text, re.IGNORECASE)
    if yesno:
        text = yesno.group(1).capitalize()
    else:
        # Copula shortening: extract the predicate NP from "X is/was Y" patterns.
        # Closes the EM/F1 gap: LLM often emits "The capital is Paris" when gold is "Paris".
        # Guard: only fires when the extracted phrase is ≤ 5 tokens (avoids over-trimming).
        cop = re.search(
            r'\b(?:is|was|are|were)\s+([A-Z][^\n.?!]{2,50}?)(?:\.|,|\s+(?:in|of|from|by|at)\s|$)',
            text
        )
        if cop:
            candidate = cop.group(1).strip().rstrip('., ')
            if 1 <= len(candidate.split()) <= 5:
                text = candidate

    return text


_STOPNAMES = frozenset({
    "the", "a", "an", "in", "on", "at", "by", "he", "she", "it",
    "his", "her", "their", "yes", "no", "this", "that", "its", "one",
})

def _constrain_simple_answer(answer, question):
    """
    Post-process SIMPLE answers to the shortest valid span for the question type.
    Applied after _extract_answer as a type-specific safety net.

    who   → longest capitalized name (person)
    when  → 4-digit year or full date
    where → first capitalized location phrase (≤ 3 words)
    other → unchanged
    """
    q = question.lower().strip()

    if q.startswith("who "):
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)
        names = [n for n in names if n.lower() not in _STOPNAMES and len(n) > 2]
        if names:
            names.sort(key=lambda n: len(n.split()), reverse=True)
            return names[0]

    elif re.match(r'^(?:when |what year|in what year)', q):
        m = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', answer)
        if m:
            return m.group(1)
        m = re.search(
            r'\b(?:January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\s+(?:\d{1,2},\s+)?\d{4}\b',
            answer,
        )
        if m:
            return m.group(0)

    elif q.startswith("where "):
        locs = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', answer)
        locs = [l for l in locs if l.lower() not in _STOPNAMES and len(l) > 2]
        if locs:
            return locs[0]

    return answer


def generate_answer(query, retrieved_passages, model=OLLAMA_MODEL, query_type=None):
    """
    Sends the query + retrieved context to local Ollama LLM and returns a
    short, clean answer via structured prompt + _extract_answer post-processing.
    """
    context = "\n\n".join([
        f"[{i+1}] {p['title']}:\n{p['text']}"
        for i, p in enumerate(retrieved_passages)
    ])

    q_lower = query.lower().strip()
    is_yesno = q_lower.startswith((
        "are ", "were ", "is ", "was ", "did ", "do ", "does ",
        "have ", "has ", "can ", "could ", "would ",
    ))

    if is_yesno:
        # Chain-of-thought: reason first, then commit to Yes/No.
        # Prevents the LLM from committing to wrong answer before working through facts.
        prompt = (
            f"You are a precise question-answering assistant. "
            f"Answer using ONLY the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Instructions: Read the context carefully. Write one sentence identifying "
            f"the key facts, then write 'Final Answer: Yes' or 'Final Answer: No' "
            f"followed by a brief reason.\n\n"
            f"Reasoning:"
        )
        num_predict = 120

    elif query_type == "COMPARISON":
        # Structured chain-of-thought to prevent entity-swapping errors.
        prompt = (
            f"You are a precise question-answering assistant. "
            f"Answer using ONLY the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Instructions:\n"
            f"1. Find the relevant attribute (e.g. birth year, founding date) for each entity in the context.\n"
            f"   Note: 'first' = earlier date (smaller year number); 'older' = born earlier (smaller birth year).\n"
            f"2. State both values explicitly, then compare them.\n"
            f"3. Write your Final Answer in one sentence naming the answer and briefly stating why.\n\n"
            f"Reasoning:"
        )
        num_predict = 180

    elif query_type == "SIMPLE":
        # Route by first question word to get the tightest possible answer span.
        # HotpotQA gold answers are 1-5 words; verbosity kills both EM and Precision.
        if q_lower.startswith("who "):
            prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer with the person's name only. No other words.\n"
                f"Name:"
            )
            num_predict = 8
        elif re.match(r'^(?:when |what year|in what year)', q_lower):
            prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer with only the year or date. No other words.\n"
                f"Year:"
            )
            num_predict = 8
        elif q_lower.startswith("where "):
            prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer with only the place name. No other words.\n"
                f"Place:"
            )
            num_predict = 8
        else:
            # what/which/other — ultra-concise, 1-5 word cap
            prompt = (
                f"Answer using ONLY the provided context. "
                f"Give the answer in as few words as possible — 1 to 5 words maximum. "
                f"No explanation, no sentence — just the key fact.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            )
            num_predict = 20

    elif query_type == "MULTI_HOP":
        # Chain-of-thought: identify the bridge entity first, then answer.
        # Multi-hop questions require following a chain (A→B→answer), so
        # forcing the LLM to name the intermediate entity reduces entity drift.
        prompt = (
            f"You are a precise question-answering assistant. "
            f"Answer using ONLY the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Instructions:\n"
            f"1. Identify the intermediate entity or fact that connects the question to the answer.\n"
            f"2. State that intermediate fact in one short phrase.\n"
            f"3. Use it to find the final answer.\n"
            f"4. Write 'Final Answer: <answer>' — as few words as possible.\n\n"
            f"Reasoning:"
        )
        num_predict = 120

    else:
        # Unknown type — allow a short sentence since reasoning is needed.
        prompt = (
            f"You are a precise question-answering assistant. "
            f"Answer using ONLY the provided context.\n"
            f"Give a direct answer in 1-2 sentences. "
            f"Do NOT start with 'Based on' or 'According to'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Final Answer:"
        )
        num_predict = 80

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": num_predict},
    )
    answer = _extract_answer(response["message"]["content"])
    if query_type == "SIMPLE":
        answer = _constrain_simple_answer(answer, query)
    return answer


# ─────────────────────────────────────────────
# STEP 4b: CROSS-ENCODER RERANKING
# ─────────────────────────────────────────────
def rerank_passages(query, passages, top_k=TOP_K):
    """
    Reranks passages using a cross-encoder for accurate query-passage relevance.
    Lazy-loads the model on first call. Returns top_k passages sorted by score.
    """
    global _reranker
    if _reranker is None:
        print(f"Loading cross-encoder reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

    if not passages:
        return passages

    pairs  = [(query, p["text"][:512]) for p in passages]
    scores = _reranker.predict(pairs)

    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    result = []
    for p, score in ranked[:top_k]:
        p_copy = dict(p)
        p_copy["rerank_score"] = float(score)
        result.append(p_copy)
    return result


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

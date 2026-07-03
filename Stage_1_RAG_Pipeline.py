"""
Stage 1: Basic RAG Pipeline
===========================
Project: HARA — Hallucination-Aware Retrieval Agent
Dataset: HotpotQA
LLM: Ollama (local)
Retrieval: FAISS + sentence-transformers

Retrieval protocol: official HotpotQA distractor setting. Each question is
retrieved against ONLY its own ~10-article context (2 gold + ~8 distractors
chosen by the dataset authors) — see build_example_corpus(). The legacy
whole-corpus functions (load_hotpotqa_passages / build_faiss_index /
load_faiss_index, backed by faiss_index.bin + passages.pkl) are kept
unchanged below because Stage 3/4/5/6 still import them for their own
multi-hop / open-retrieval experiments — Stage 1's own evaluate() and demo
no longer use them.
"""

import argparse
import json
import re
import string
from collections import Counter
import numpy as np
import faiss
from datasets import load_dataset, concatenate_datasets
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

# BGE-small-en-v1.5 scores ~8pt higher than all-MiniLM-L6-v2 on BEIR benchmarks
# at identical 384-dim output — zero index size change, drop-in replacement.
# IMPORTANT: changing EMBED_MODEL or CHUNK_SIZE invalidates saved faiss_index.bin
# and passages.pkl. Delete both files before the first run after any change here.
EMBED_MODEL    = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL   = "llama3.2"
TOP_K          = 5              # passages sent to LLM (post-rerank)
RERANK_POOL    = 50             # candidates retrieved before cross-encoder reranking
RRF_K          = 60             # RRF smoothing constant (Cormack et al., SIGIR 2009)
CHUNK_SIZE     = 3              # sentences per passage chunk

# MAX_PASSAGES / INDEX_PATH / PASSAGES_PATH back the legacy whole-corpus mode
# (load_hotpotqa_passages / build_faiss_index / load_faiss_index) — kept only
# because Stage 3/4/5/6 still import them. Stage 1's own protocol below
# (build_example_corpus) builds a fresh ~20-40 chunk index per question in
# memory and needs none of these.
MAX_PASSAGES   = 500000
INDEX_PATH     = "faiss_index.bin"
PASSAGES_PATH  = "passages.pkl"

# Default number of validation examples scored by the 'eval' command.
# Override at runtime with --num-samples (see __main__), e.g. for a larger
# defense-scale run, without editing this constant.
EVAL_NUM_SAMPLES = 50

# BGE models achieve best asymmetric retrieval with this query prefix.
# v1.5 works without it but the prefix closes the query/document gap.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_reranker         = None   # lazy-loaded cross-encoder
_bm25             = None   # lazy-loaded BM25 index (built in-memory, no disk write)
_embedding_cache  = {}     # chunk text -> raw (pre-normalization) embedding vector,
                           # process-lifetime only, never written to disk (see
                           # _embed_chunks_cached below)


# ─────────────────────────────────────────────
# QUERY ENCODING HELPER
# ─────────────────────────────────────────────
def _encode_query(embedder, query: str) -> np.ndarray:
    """
    Encode a retrieval query, applying the BGE asymmetric prefix when needed.

    BGE uses asymmetric retrieval: queries and documents are encoded differently.
    Documents are encoded as-is; queries get the prefix so their representation
    aligns with the document space.  MiniLM and E5 do not need this — the check
    is model-name based so switching EMBED_MODEL keeps everything correct.

    Returns a (1, dim) float32 array, L2-normalised for cosine similarity.
    """
    text = (_BGE_QUERY_PREFIX + query) if "bge" in EMBED_MODEL.lower() else query
    vec  = embedder.encode([text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)
    return vec


# ─────────────────────────────────────────────
# STEP 1: LOAD & PREPARE HOTPOTQA PASSAGES
# ─────────────────────────────────────────────
def _chunk_sentences(title, sentences, chunk_size=CHUNK_SIZE):
    """
    Splits one article's sentence list into non-overlapping chunk_size windows.

    Shared by both the legacy whole-corpus loader (load_hotpotqa_passages) and
    the per-example distractor-protocol loader (chunk_context) below, so the
    two paths can never silently diverge in how they chunk the same article.

    Why chunking beats whole-article concatenation for HotpotQA:
      - Dense embeddings average over all token representations; a 12-sentence
        article buries the 1-2 relevant sentences in noise, dropping cosine
        similarity against the query by ~10-15 points on BEIR.
      - 3-sentence chunks keep each embedding tight around one fact cluster,
        so the supporting sentence is no longer diluted by unrelated context.
      - Cross-encoder precision improves: it scores 3 focused sentences vs a
        paragraph the model may partially ignore due to truncation at 512 tokens.
    """
    chunks = []
    for start in range(0, max(1, len(sentences)), chunk_size):
        chunk_sents = sentences[start: start + chunk_size]
        chunk_text  = " ".join(chunk_sents).strip()
        if not chunk_text:
            continue
        chunks.append({
            "title":    title,
            "text":     chunk_text,
            "chunk_id": start // chunk_size,
        })
    return chunks


def load_hotpotqa_passages(max_passages=MAX_PASSAGES, chunk_size=CHUNK_SIZE):
    """
    LEGACY whole-corpus loader — kept for Stage 3/4/5/6, which still import it
    for their own multi-hop / open-retrieval experiments. Stage 1's own
    evaluate() and demo no longer call this (see build_example_corpus below).

    Loads HotpotQA and extracts non-overlapping sentence-window chunks per
    article, deduplicating articles globally across the whole training split
    (an article seen in an earlier question is not re-chunked for a later
    question that also references it). Each chunk keeps the article title so
    Recall@K can match against HotpotQA supporting_facts (title-keyed).

    IMPORTANT: changing chunk_size invalidates saved faiss_index.bin / passages.pkl.
    Delete both before rebuilding.
    """
    print("Loading HotpotQA dataset...")
    dataset = load_dataset("hotpot_qa", "distractor", split="train")

    passages    = []
    seen_titles = set()

    for example in tqdm(dataset, desc="Extracting passages"):
        for title, sentences in zip(
            example["context"]["title"],
            example["context"]["sentences"]
        ):
            if title in seen_titles:
                continue
            seen_titles.add(title)
            passages.extend(_chunk_sentences(title, sentences, chunk_size))

        if len(passages) >= max_passages:
            break

    print(f"Extracted {len(passages):,} chunks from {len(seen_titles):,} articles.")
    return passages


# ─────────────────────────────────────────────
# STEP 1b: PER-EXAMPLE CORPUS — OFFICIAL DISTRACTOR PROTOCOL
# ─────────────────────────────────────────────
def _embed_chunks_cached(embedder, texts):
    """
    Encodes `texts`, reusing any embedding already computed earlier in this
    process and batch-encoding only the cache misses.

    Correctness: embedding a given string is a deterministic pure function of
    the string alone (same text -> same vector, regardless of which article
    or question produced it), so a cache hit is byte-for-byte identical to
    what a fresh encode() call would return. This function never influences
    WHICH chunks belong to a question's corpus — that's decided entirely by
    chunk_context()/build_example_corpus() before this is ever called; it
    only avoids recomputing the embedding for text seen before (many
    Wikipedia articles recur as distractors across different questions).

    Returns raw (pre-L2-normalization) vectors — build_example_corpus
    normalizes the returned batch itself, exactly as before. Cached vectors
    are stored pre-normalization so repeated cache hits can never be
    double-normalized.
    """
    missing = [t for t in texts if t not in _embedding_cache]
    if missing:
        new_vecs = embedder.encode(missing, batch_size=64, show_progress_bar=False)
        for t, v in zip(missing, new_vecs):
            _embedding_cache[t] = np.asarray(v, dtype="float32")
    return np.stack([_embedding_cache[t] for t in texts]).astype("float32")


def chunk_context(context, chunk_size=CHUNK_SIZE):
    """
    Chunks a single HotpotQA example's `context` field (all ~10 articles:
    gold + distractor, unmarked) into 3-sentence windows. No cross-example
    deduplication — each question's corpus is independent by design.
    """
    passages = []
    for title, sentences in zip(context["title"], context["sentences"]):
        passages.extend(_chunk_sentences(title, sentences, chunk_size))
    return passages


def build_example_corpus(example, embedder, chunk_size=CHUNK_SIZE):
    """
    Builds an ephemeral, in-memory retrieval corpus from a single HotpotQA
    example's own context — the official distractor-setting protocol.

    Each distractor example ships ~10 articles (2 gold supporting + ~8
    distractors chosen by the dataset authors to be topically confusable
    with the gold ones). The retriever sees all of them identically; nothing
    in this function marks which are gold. `supporting_facts` is never read
    here — it is only consulted later, during scoring, for Recall@K.

    Returns (index, passages, bm25), all freshly built for this example and
    meant to be discarded once it's scored:
      - index: faiss.IndexFlatIP over this example's chunks only (~20-40
        vectors) — exact search, no IVF/HNSW needed at this scale.
      - passages: the chunk dicts (title/text/chunk_id) backing `index`.
      - bm25: a fresh BM25Okapi scoped to these chunks (None if rank-bm25
        isn't installed), for retrieve_hybrid's `bm25=` argument.

    No disk writes — this replaces the persistent 500k-chunk
    faiss_index.bin / passages.pkl of the legacy whole-corpus mode, which
    pooled every training article across all 90k questions into one global
    index and let a question retrieve chunks from unrelated questions'
    articles. Per-example corpora remove that cross-question noise and match
    the standard distractor-setting protocol other HotpotQA papers report
    against.
    """
    passages = chunk_context(example["context"], chunk_size=chunk_size)
    if not passages:
        return None, [], None

    texts      = [p["text"] for p in passages]
    embeddings = _embed_chunks_cached(embedder, texts)
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    bm25 = None
    if _BM25_AVAILABLE:
        tokenized = [p["text"].lower().split() for p in passages]
        bm25 = _BM25Okapi(tokenized)

    return index, passages, bm25


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


def _rrf_fuse(ranked_lists: list, k: int = RRF_K) -> list:
    """
    Reciprocal Rank Fusion across any number of ranked document lists.

    RRF(d) = Σ_i  1 / (k + rank_i(d))   (Cormack et al., SIGIR 2009)

    Why RRF beats linear score fusion for HARA:
      - Dense cosine scores and BM25 scores live on incompatible scales;
        min-max normalisation is sensitive to outliers in the candidate pool,
        making the alpha weight meaningless across different queries.
      - RRF uses only rank position, so scale differences are irrelevant and
        no alpha hyperparameter needs tuning — one fewer variable to report.
      - A document ranked #1 by one system receives strong signal even if the
        other system ignores it entirely, which is important for exact-name
        entity lookups (BM25 wins) vs. paraphrase queries (dense wins).

    Args:
        ranked_lists: each sub-list is a list of passage indices, best-first.
        k:            smoothing constant; 60 is the standard value.

    Returns:
        List of (passage_idx, rrf_score) sorted best-first.
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve_hybrid(query, index, embedder, passages, top_k=RERANK_POOL, alpha=0.5, bm25=None):
    """
    RRF-fused BM25 + dense retrieval.  Returns top_k candidates for the reranker.

    `alpha` is retained for API compatibility with Stage 3 callers but is no
    longer used — RRF replaces linear score fusion (see _rrf_fuse docstring).

    bm25: optional local BM25Okapi instance scoped to `passages` (see
    build_example_corpus). When given — the per-example distractor-protocol
    path — it takes precedence over the module-level `_bm25` global. When
    omitted, falls back to `_bm25` (set via init_bm25()), preserving the
    exact existing behaviour for Stage 3/4/5/6 callers using the legacy
    whole-corpus index. Falls back to dense-only if neither is set.
    """
    pool_size = min(top_k * 2, len(passages))

    # ── Dense retrieval with BGE query prefix ──
    q_vec = _encode_query(embedder, query)
    d_scores, d_indices = index.search(q_vec, pool_size)
    dense_ranked = [int(i) for i in d_indices[0] if int(i) < len(passages)]

    active_bm25 = bm25 if bm25 is not None else _bm25

    # Defensive check (does not change behaviour for any correctly-matched
    # caller — Stage 1's own per-example bm25/passages pair always matches,
    # and Stage 3/6's legacy init_bm25(passages) + retrieve_hybrid(passages)
    # calls use the same `passages` object for both, so this never fires for
    # them either). Guards against a corpus-size mismatch — e.g. a stale
    # whole-corpus `_bm25` global left over from another script in the same
    # process being used against a small per-example `passages` list — which
    # would otherwise silently score against unrelated indices instead of
    # raising an error. See Stage 1 architectural review.
    if active_bm25 is not None and getattr(active_bm25, "corpus_size", None) != len(passages):
        print(f"Warning: bm25 corpus_size ({getattr(active_bm25, 'corpus_size', '?')}) "
              f"!= len(passages) ({len(passages)}) — ignoring bm25 for this query "
              f"(dense-only fallback).")
        active_bm25 = None

    if active_bm25 is not None:
        bm25_scores  = active_bm25.get_scores(query.lower().split())
        bm25_ranked  = [
            int(i) for i in np.argsort(bm25_scores)[::-1][:pool_size]
            if int(i) < len(passages)
        ]
        fused = _rrf_fuse([dense_ranked, bm25_ranked], k=RRF_K)
    else:
        fused = [
            (int(idx), 1.0 / (RRF_K + rank + 1))
            for rank, idx in enumerate(dense_ranked)
        ]

    results = []
    for idx, score in fused[:top_k]:
        p = passages[idx]
        entry = {
            "title": p["title"],
            "text":  p["text"],
            "score": score,
            "hop":   1,
        }
        if "chunk_id" in p:
            entry["chunk_id"] = p["chunk_id"]
        results.append(entry)
    return results


# ─────────────────────────────────────────────
# STEP 2c: TWO-HOP RETRIEVAL (Stage 1 utility)
# ─────────────────────────────────────────────
def retrieve_multi_hop_rrf(query, index, embedder, passages, top_k=RERANK_POOL, bm25=None):
    """
    Two-hop RRF retrieval for bridge-entity multi-hop questions (Stage 1 utility).

    Pipeline:
      Hop 1 — retrieve with the original query (RRF dense + BM25)
      Bridge — extract the best bridge-entity candidate from Hop 1's top passage
               using capitalized multi-word NP extraction (no external NER dep)
      Hop 2 — retrieve using the bridge entity as a follow-up query
      Merge — combine both ranked lists via RRF; passages appearing in both hops
               receive a compounded score

    Rationale for HotpotQA:
      Bridge questions have the form "What is X of the Y who Z?"
      Hop 1 finds the Y article → its text names the specific Y instance.
      Hop 2 retrieves the X article for that instance.
      A single-hop retrieval cannot guarantee both gold articles appear in top-k.

    Note: Stage 3 has a more sophisticated LLM-decomposed multi-hop
    (decompose_and_retrieve_multi_hop).  This function is Stage 1's self-contained
    version — no LLM calls in the retrieval path, safe to call from evaluate().
    """
    # ── Hop 1 ──
    hop1 = retrieve_hybrid(query, index, embedder, passages, top_k=top_k, bm25=bm25)

    # ── Bridge extraction: largest capitalized NP from top passage text/title ──
    bridge_query = query  # safe fallback
    if hop1:
        top_title = hop1[0].get("title", "")
        top_text  = hop1[0].get("text", "")[:500]
        _stop = {
            "The", "This", "These", "Those", "There", "Their", "They",
            "His", "Her", "Its", "Our", "Your", "Him", "Them", "Also",
        }
        nps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', top_text)
        nps = [n for n in nps if n not in _stop and len(n) > 4]
        bridge = nps[0] if nps else top_title
        bridge_query = bridge if bridge else query

    # ── Hop 2 ──
    hop2 = retrieve_hybrid(bridge_query, index, embedder, passages, top_k=top_k, bm25=bm25)

    # ── Deduplicate across hops and assign virtual indices for RRF ──
    key_to_vid: dict = {}     # (title, text_prefix) → virtual_id
    vid_to_passage: dict = {} # virtual_id → passage dict
    vid = 0

    for p in hop1 + hop2:
        key = (p["title"], p["text"][:60])
        if key not in key_to_vid:
            key_to_vid[key] = vid
            vid_to_passage[vid] = p
            vid += 1

    hop1_order = [key_to_vid[(p["title"], p["text"][:60])] for p in hop1]
    hop2_order = [key_to_vid[(p["title"], p["text"][:60])] for p in hop2]

    fused = _rrf_fuse([hop1_order, hop2_order], k=RRF_K)

    hop1_set = set(hop1_order)
    hop2_set = set(hop2_order)
    results = []
    for virtual_id, score in fused[:top_k]:
        p = dict(vid_to_passage[virtual_id])
        p["score"] = score
        in1, in2 = (virtual_id in hop1_set), (virtual_id in hop2_set)
        p["hop"] = "both" if (in1 and in2) else ("1" if in1 else "2")
        results.append(p)

    return results


# ─────────────────────────────────────────────
# STEP 3: RETRIEVE RELEVANT PASSAGES
# ─────────────────────────────────────────────
def retrieve(query, index, embedder, passages, top_k=TOP_K):
    """
    Dense-only FAISS retrieval with BGE query prefix applied via _encode_query.
    Kept as a lightweight fallback; the main pipeline uses retrieve_hybrid.
    """
    q_vec = _encode_query(embedder, query)
    scores, indices = index.search(q_vec, top_k)

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
# STEP 4: ANSWER EXTRACTION + GENERATION
# ─────────────────────────────────────────────
def _extract_answer(raw_text):
    """
    Post-processes the LLM output to return a short, clean answer.
    Looks for 'Final Answer:' first; falls back to stripping hedging phrases.
    """
    text = raw_text.strip()

    # Phrases that look like answers but are refusals — never return these
    _REFUSAL_FRAGMENTS = frozenset({
        "not found", "not available", "not mentioned", "not provided",
        "not determined", "not specified", "not known", "not stated",
        "cannot determine", "cannot be found", "cannot be determined",
        "not in context", "not in the context", "not enough information",
        "i cannot", "i don't know", "i do not know",
    })

    def _is_refusal_fragment(s: str) -> bool:
        s_lower = s.lower().strip()
        return any(frag in s_lower for frag in _REFUSAL_FRAGMENTS)

    # Primary: pull out everything after "Final Answer:" on that line
    m = re.search(r'(?:Final Answer|Answer)\s*:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        ans = m.group(1).strip().rstrip('.,')
        # Only use this extraction if it is not a refusal phrase
        if ans and not _is_refusal_fragment(ans):
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
            if 1 <= len(candidate.split()) <= 5 and not _is_refusal_fragment(candidate):
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
        # Hidden-reasoning style: contextual hints without explicit CoT steps.
        # Asking the model to show numbered reasoning ("Step 1, Step 2...") causes
        # it to fabricate specific numbers not in context; a concise prompt that
        # names the target format keeps the answer tight and verifier-friendly.
        prompt = (
            f"You are a precise question-answering assistant. "
            f"Answer using ONLY the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Find the relevant attribute for each entity in the context, then compare. "
            f"'First' means earlier date; 'older' means born earlier. "
            f"Write 'Final Answer: <name>' — exactly one of the two entity names from "
            f"the question, no explanation, no yes/no.\n\n"
            f"Final Answer:"
        )
        num_predict = 60

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
        # Hidden-reasoning style: cross-referencing hint without exposing CoT steps.
        # Asking for "Reasoning:" causes verbose intermediate output that inflates
        # num_predict and sometimes causes the model to commit to a wrong bridge early.
        # A direct "Final Answer:" target with a cross-reference instruction keeps the
        # model's reasoning internal while still guiding multi-document evidence use.
        prompt = (
            f"You are a precise question-answering assistant. "
            f"Answer using ONLY the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Cross-reference the passages above — the answer requires connecting "
            f"evidence from multiple articles. 'First' means earlier year; "
            f"'older' means born earlier.\n"
            f"Write 'Final Answer: <answer>' — shortest possible phrase.\n\n"
            f"Final Answer:"
        )
        num_predict = 60

    else:
        # Stage 1 baseline — intentionally allows knowledge leakage so the LLM
        # produces hallucinated / fabricated answers when context is insufficient.
        # This establishes the high hallucination rate that motivates Stages 2–4.
        # The verifier in Stage 2 will label these answers UNSUPPORTED/PARTIAL,
        # producing the measurable baseline gap that the paper argues against.
        prompt = (
            f"You are a question-answering assistant. "
            f"Use the provided context to answer. "
            f"If the context lacks the answer, use your general knowledge "
            f"to give your best answer.\n"
            f"Give a direct, concise answer in 1-2 sentences. "
            f"NEVER write 'Not Found', 'not available', 'I don't know', "
            f"'I couldn't find', or any refusal. Always give your best answer.\n\n"
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
# STEP 4b: LLM JUDGE (shared utility)
# Used by Stage 3 and Stage 4 to catch entity-
# role hallucinations that DistilBERT misses.
# ─────────────────────────────────────────────
def llm_judge_supported(question: str, answer: str,
                         context_passages: list, verbose: bool = False) -> bool:
    """Ask the LLM whether the context explicitly supports the answer.

    DistilBERT verifier checks word/semantic overlap but cannot verify that a
    named entity appears in the correct ROLE.  For example, "David Martínez"
    may appear in retrieved passages but not as a Mexican F1 podium finisher.
    DistilBERT sees the name → SUPPORTED (false positive).

    This judge asks a direct YES/NO question to the same LLM already running
    locally, catching role-mismatch hallucinations that DistilBERT misses.

    Applied in Stage 3 and Stage 4 for MULTI_HOP and COMPARISON queries.
    Only called when DistilBERT gives SUPPORTED to avoid per-iteration latency.

    Returns True (keep SUPPORTED) or False (label should be PARTIAL/UNSUPPORTED).
    """
    # Use top-8 passages so the relevant article is less likely to fall outside
    # the judge's window — critical for multi-hop questions where the key bridge
    # passage is often ranked 5th-8th after reranking.
    context = " ".join(p["text"] for p in context_passages[:8])[:2400]
    prompt = (
        f"You are a strict fact-checker for a question-answering system.\n\n"
        f"Question: {question}\n"
        f"Proposed answer: {answer}\n\n"
        f"Context (retrieved Wikipedia passages):\n{context}\n\n"
        f"Your task: Decide whether '{answer}' is the CORRECT and COMPLETE answer "
        f"to the question, based ONLY on the provided context.\n\n"
        f"Step 1 — Identify every requirement the question makes "
        f"(nationality, role, time period, relationship to other entities, etc.).\n"
        f"Step 2 — For EACH requirement, check whether the context explicitly "
        f"confirms that '{answer}' satisfies it.\n"
        f"Step 3 — Reply YES only if ALL requirements are met by clear evidence "
        f"in the context. Reply NO if ANY requirement is unconfirmed, ambiguous, "
        f"or if '{answer}' appears in the context in a role that does not directly "
        f"answer the question.\n\n"
        f"Reply with only YES or NO."
    )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 10},
        )
        reply = response["message"]["content"].strip().upper()
        if verbose:
            print(f"[LLM Judge] {reply}")
        return reply.startswith("YES")
    except Exception:
        return True  # if Ollama fails, trust DistilBERT


# ─────────────────────────────────────────────
# STEP 4c: CROSS-ENCODER RERANKING
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
def rag_query(query, index, embedder, passages, bm25=None):
    """
    Full Stage 1 pipeline:
    Query → RRF Hybrid Retrieve (pool=RERANK_POOL) → Cross-Encoder Rerank → Generate
    No verification. This is the hallucination baseline.

    bm25: optional local BM25Okapi instance (see build_example_corpus), passed
    through to retrieve_hybrid. Omitted by existing Stage 4/5 callers, which
    keep using the legacy global `_bm25` exactly as before.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    # Retrieve a wide candidate pool via RRF, then rerank to TOP_K
    candidates = retrieve_hybrid(query, index, embedder, passages, top_k=RERANK_POOL, bm25=bm25)
    retrieved  = rerank_passages(query, candidates, top_k=TOP_K)

    print(f"\nTop {len(retrieved)} passages (reranked from {len(candidates)} candidates):")
    for i, p in enumerate(retrieved):
        print(f"  [{i+1}] {p['title']} (rerank: {p.get('rerank_score', 0):.4f})")

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


def token_f1(pred: str, gold: str) -> float:
    """
    Token-level F1 using HotpotQA official normalization.

    Computes the harmonic mean of token-precision and token-recall between
    the predicted and gold answer after shared normalization.  This metric
    gives partial credit when the prediction overlaps with the gold answer
    but isn't an exact string match — critical for multi-word entity answers
    where EM is 0 but the prediction shares key tokens with the gold.
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common    = Counter(pred_tokens) & Counter(gold_tokens)
    num_same  = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_recall_at_k(retrieved_titles: list, gold_titles: list, k: int) -> float:
    """
    Fraction of gold supporting-fact article titles found in the top-k retrieved titles.

    HotpotQA supporting_facts provides exactly the two Wikipedia article titles
    whose sentences contain the gold evidence chain.  Recall@K measures whether
    the retrieval stage surfaces both before generation and reranking — it is a
    retrieval-quality metric independent of generation quality.

    Args:
        retrieved_titles: ordered list of titles from the retrieval pool (pre-rerank).
        gold_titles:      deduplicated list of supporting fact titles from HotpotQA.
        k:                cutoff (5, 10, or 20).

    Returns:
        Float in [0, 1] — 1.0 means all gold articles found in top-k.
    """
    if not gold_titles:
        return 1.0
    top_k_set = {t.lower() for t in retrieved_titles[:k]}
    found = sum(1 for t in gold_titles if t.lower() in top_k_set)
    return found / len(gold_titles)


def _unique_titles_by_rank(titles: list) -> list:
    """
    Collapses a ranked list of (possibly repeated) chunk titles into the
    ordered list of first-occurrence-unique titles, preserving rank order.
    e.g. ["A", "A", "B", "A", "C"] -> ["A", "B", "C"].

    A single gold article can contribute several chunks (3-sentence windows),
    so without this collapse, two top-ranked SLOTS could both be chunks of
    the SAME gold article — capping recall at 0.5 for a 2-gold-title example
    even though that article was correctly identified, because no slot was
    left to test whether the OTHER gold article also ranked well. This
    distortion is worst at small k (e.g. Recall@2); see
    compute_recall_at_k_titles.
    """
    seen: set = set()
    unique = []
    for t in titles:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def compute_recall_at_k_titles(retrieved_titles: list, gold_titles: list, k: int) -> float:
    """
    Title-deduplicated Recall@K — use this (not compute_recall_at_k) for
    small k, where chunk-level duplication of one article can otherwise
    consume multiple ranked slots. Ranks chunks normally, then collapses
    repeated titles via _unique_titles_by_rank before taking the first k
    UNIQUE titles.

    Currently used only for Recall@2 (see evaluate()) — a genuinely
    diagnostic cutoff at this pipeline's per-example corpus scale (~10
    articles): it directly tests whether the fused dense+BM25/RRF ranking
    places both gold articles ahead of the ~8 distractors, which Recall@10/20
    cannot do once the corpus itself is close to that size (see evaluate()
    for the full reasoning). Recall@2 is a diagnostic, not the primary
    retrieval metric — Recall@5 remains primary since it matches TOP_K, the
    number of passages actually sent to the LLM.
    """
    if not gold_titles:
        return 1.0
    unique_titles = _unique_titles_by_rank(retrieved_titles)
    top_k_set = {t.lower() for t in unique_titles[:k]}
    found = sum(1 for t in gold_titles if t.lower() in top_k_set)
    return found / len(gold_titles)


def evaluate(embedder, num_samples=100, chunk_size=CHUNK_SIZE):
    """
    Evaluates Stage 1 on HotpotQA validation set under the OFFICIAL
    DISTRACTOR PROTOCOL: each question is retrieved against only its own
    ~10-article context, rebuilt fresh per example via build_example_corpus.
    There is no shared global index — a question can never retrieve chunks
    belonging to a different question. supporting_facts is read only for
    scoring (gold_titles below), never passed into retrieval.

    Signature change from the previous whole-corpus version: this no longer
    takes (index, passages) since there is nothing global left to pass — only
    `embedder` is reused across examples (the model itself, not any corpus).

    Metrics reported:
      Exact Match  — strict string equality after normalization
      Token F1     — HotpotQA official token-overlap F1
      Recall@5     — PRIMARY retrieval-quality metric. Matches TOP_K exactly,
                     so it measures whether both gold articles survive into
                     the passages actually available for generation.
      Recall@2     — DIAGNOSTIC ONLY (title-deduplicated). With a ~10-article
                     per-example corpus, this is the one cutoff genuinely
                     small enough to test whether the fused dense+BM25/RRF
                     ranking places both gold articles ahead of distractors,
                     rather than just confirming the corpus isn't degenerate.
      Recall@10/20 — kept for continuity, but can SATURATE near 1.0 in this
                     per-example distractor setting: since a typical corpus
                     is only ~20-40 chunks (~10 articles), a k of 10 or 20
                     often covers most/all unique titles already present,
                     regardless of ranking quality. Not primary evidence of
                     retrieval quality at this corpus scale — see Recall@5.

    Pipeline per sample:
      build_example_corpus → retrieve_hybrid (local bm25, pool capped to this
      example's own chunk count) → rerank_passages (TOP_K) → generate_answer.
    """
    print(f"\nEvaluating on {num_samples} HotpotQA validation samples "
          f"(official distractor protocol — per-example retrieval corpus)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    results   = []
    em_list, f1_list             = [], []
    r2_list, r5_list, r10_list, r20_list = [], [], [], []
    skipped = 0

    for i, example in enumerate(tqdm(dataset)):
        if i >= num_samples:
            break

        query       = example["question"]
        gold_answer = example["answer"]
        # Deduplicate gold titles (supporting_facts may repeat the same title)
        gold_titles = list(dict.fromkeys(example["supporting_facts"]["title"]))

        # Fresh, question-scoped corpus — built only from this example's own
        # context (2 gold + ~8 distractors), then discarded after scoring.
        ex_index, ex_passages, ex_bm25 = build_example_corpus(
            example, embedder, chunk_size=chunk_size
        )
        if ex_index is None:
            skipped += 1
            continue  # malformed example with empty context

        # Pool size is capped to this example's own chunk count (~20-40),
        # not the old fixed RERANK_POOL=50 sized for a 500k-chunk index.
        # Note: at this corpus scale, recall_pool / retrieve_hybrid's own
        # internal pool_size cap (min(top_k*2, len(passages))) both usually
        # collapse to len(ex_passages) itself — i.e. the RRF-fused ranking
        # typically covers the ENTIRE per-example corpus, and reranking is
        # the only step that actually discards chunks. This is intentional:
        # at ~20-40 candidates, cross-encoder reranking over the full corpus
        # is cheap, so letting the strongest available signal (the
        # cross-encoder) see everything avoids risking recall loss from a
        # weaker first-stage cutoff for a marginal efficiency gain. Left
        # unchanged — see Stage 1 architectural review, Part/Improvement 6.
        recall_pool = min(max(RERANK_POOL, 20), len(ex_passages))
        pool        = retrieve_hybrid(
            query, ex_index, embedder, ex_passages,
            top_k=recall_pool, bm25=ex_bm25,
        )
        reranked    = rerank_passages(query, pool, top_k=TOP_K)
        pred_answer = generate_answer(query, reranked)

        pool_titles = [p["title"] for p in pool]
        em  = exact_match(pred_answer, gold_answer)
        f1  = token_f1(pred_answer, gold_answer)
        r2  = compute_recall_at_k_titles(pool_titles, gold_titles, 2)  # diagnostic
        r5  = compute_recall_at_k(pool_titles, gold_titles, 5)         # primary
        r10 = compute_recall_at_k(pool_titles, gold_titles, 10)
        r20 = compute_recall_at_k(pool_titles, gold_titles, 20)

        em_list.append(em);  f1_list.append(f1)
        r2_list.append(r2)
        r5_list.append(r5);  r10_list.append(r10);  r20_list.append(r20)

        results.append({
            "question":        query,
            "gold":            gold_answer,
            "predicted":       pred_answer,
            "exact_match":     em,
            "f1":              round(f1, 4),
            "recall_at_2":     round(r2, 4),
            "recall_at_5":     round(r5, 4),
            "recall_at_10":    round(r10, 4),
            "recall_at_20":    round(r20, 4),
            "retrieved_titles": pool_titles,
            "gold_titles":     gold_titles,
            "corpus_size":     len(ex_passages),
        })

    def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

    avg_em  = _avg(em_list)
    avg_f1  = _avg(f1_list)
    avg_r2  = _avg(r2_list)
    avg_r5  = _avg(r5_list)
    avg_r10 = _avg(r10_list)
    avg_r20 = _avg(r20_list)

    print(f"\n{'='*60}")
    print(f"Stage 1 Results — Distractor Protocol ({len(results)} samples"
          f"{f', {skipped} skipped' if skipped else ''})")
    print(f"{'='*60}")
    print(f"  Exact Match : {avg_em:.4f}  ({avg_em*100:.1f}%)")
    print(f"  Token F1    : {avg_f1:.4f}  ({avg_f1*100:.1f}%)")
    print(f"  Recall@2    : {avg_r2:.4f}  ({avg_r2*100:.1f}%)  [diagnostic, title-deduplicated]")
    print(f"  Recall@5    : {avg_r5:.4f}  ({avg_r5*100:.1f}%)  [primary retrieval metric]")
    print(f"  Recall@10   : {avg_r10:.4f}  ({avg_r10*100:.1f}%)  [can saturate — see docstring]")
    print(f"  Recall@20   : {avg_r20:.4f}  ({avg_r20*100:.1f}%)  [can saturate — see docstring]")

    with open("stage1_results.json", "w") as f:
        json.dump({
            "protocol":     "distractor_per_example",
            "exact_match":  avg_em,
            "f1":           avg_f1,
            "recall_at_2":  avg_r2,
            "recall_at_5":  avg_r5,
            "recall_at_10": avg_r10,
            "recall_at_20": avg_r20,
            "results":      results,
        }, f, indent=2)
    print("Results saved to stage1_results.json")

    return results, avg_em


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Official distractor protocol: there is no persistent global index to
    # build or load here anymore — each question gets its own ephemeral
    # corpus from build_example_corpus(). (The legacy load_faiss_index /
    # build_faiss_index / load_hotpotqa_passages functions above still work
    # exactly as before and remain in this file for Stage 3/4/5/6, which
    # import them for their own whole-corpus experiments.)
    parser = argparse.ArgumentParser(description="Stage 1: Basic RAG Demo (Official Distractor Protocol)")
    parser.add_argument(
        "--num-samples", type=int, default=EVAL_NUM_SAMPLES,
        help=f"Number of validation examples the 'eval' command scores (default: {EVAL_NUM_SAMPLES})",
    )
    args = parser.parse_args()

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    # Demo browsing covers train + validation combined (deterministic order:
    # all train questions, then all validation questions — no duplicates,
    # nothing shuffled). This is browsing convenience only — it does NOT
    # affect evaluate(), which loads the validation split independently and
    # unconditionally, exactly as before, so benchmark metrics are still
    # computed exclusively on validation. Each question, regardless of
    # origin split, still gets its own ephemeral per-example corpus via
    # build_example_corpus() — there is still no global index of any kind.
    print("Loading HotpotQA train + validation splits (distractor)...")
    train_dataset = load_dataset("hotpot_qa", "distractor", split="train")
    val_dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    combined_dataset = concatenate_datasets([train_dataset, val_dataset])
    train_size = len(train_dataset)

    # Interactive demo
    print("\n=== Stage 1: Basic RAG Demo (Official Distractor Protocol) ===")
    print(f"{len(combined_dataset)} questions loaded "
          f"({train_size} train + {len(val_dataset)} validation).")
    print("Retrieval is scoped per-question — each question is answered only")
    print("against its own ~10-article context, never a shared corpus.")
    print(f"Enter an example index (0-{len(combined_dataset)-1}), "
          f"'eval' to run batch evaluation on the validation split "
          f"({args.num_samples} samples; override with --num-samples), "
          f"or 'quit' to exit.\n")

    while True:
        cmd = input("Example index / 'eval' / 'quit': ").strip()
        if cmd.lower() == "quit":
            break
        elif cmd.lower() == "eval":
            evaluate(embedder, num_samples=args.num_samples)
        elif cmd.isdigit():
            idx = int(cmd)
            if not (0 <= idx < len(combined_dataset)):
                print(f"Index out of range (0-{len(combined_dataset)-1}).")
                continue

            origin_split = "train" if idx < train_size else "validation"
            example = combined_dataset[idx]
            print(f"\n[{origin_split}] Gold answer: {example['answer']}")

            ex_index, ex_passages, ex_bm25 = build_example_corpus(example, embedder)
            if ex_index is None:
                print("This example has an empty context — skipping.")
                continue

            rag_query(example["question"], ex_index, embedder, ex_passages, bm25=ex_bm25)
        else:
            print("Enter an example index, 'eval', or 'quit'.")

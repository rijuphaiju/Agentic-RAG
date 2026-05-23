# Manseez Branch — Agentic RAG Pipeline (Stage 1–4)
**Contributor:** Manseez Bahadur Pradhan (780320)
**Project:** Reducing Hallucinations in Agentic RAG Systems using Adaptive Retrieval and Self-Verification
**Institution:** Khwopa Engineering College, Purbanchal University

---

## What This Branch Contains

This branch contains the complete implementation of all 4 stages of the Agentic RAG pipeline, developed and tested on **Windows with an NVIDIA RTX 4050 Laptop GPU**.

| File | Stage | Description |
|------|-------|-------------|
| `rag_pipeline.py` | Stage 1 | Basic RAG — retrieve + generate (baseline) |
| `verifier_gpu.py` | Stage 2 | DistilBERT faithfulness verifier (GPU accelerated) |
| `adaptive_retrieval.py` | Stage 3 | Query classifier + adaptive retrieval strategies |
| `agentic_loop.py` | Stage 4 | Self-correcting agentic decision loop |
| `requirements.txt` | — | Python dependencies |
| `sample_questions.txt` | — | Test questions with expected outputs |

---

## Work Done — Stage by Stage

---

### Stage 1: Basic RAG Pipeline (`rag_pipeline.py`)

**What it does:**
- Loads and indexes 150,000 Wikipedia passages from the HotpotQA dataset
- Converts passages into dense vector embeddings using `sentence-transformers`
- Builds a FAISS vector index for fast similarity search
- Takes a user query, retrieves top-10 most relevant passages
- Passes retrieved context to Ollama (llama3.2) to generate an answer
- No verification — this is the hallucination baseline

**Key components:**
- Embedding model: `all-MiniLM-L6-v2`
- Vector store: FAISS IndexFlatIP (cosine similarity)
- LLM: Ollama llama3.2 (runs locally, no API cost)
- Dataset: HotpotQA distractor split (train: 90,447 examples)

**Evaluation metric:**
- Exact Match (EM) score on HotpotQA validation set
- Run: `python rag_pipeline.py` → type `eval`

**Files generated:**
- `faiss_index.bin` — FAISS vector index (150k passages)
- `passages.pkl` — indexed passage texts and titles
- `stage1_results.json` — evaluation results

---

### Stage 2: Faithfulness Verifier (`verifier_gpu.py`)

**What it does:**
- Fine-tunes DistilBERT on automatically generated training data from HotpotQA
- Classifies (context, answer) pairs as `SUPPORTED`, `PARTIAL`, or `UNSUPPORTED`
- Operates independently from the LLM — eliminates shared-model bias
- Uses GPU acceleration with mixed precision (AMP) for faster training
- Implements early stopping to prevent overfitting

**Architecture:**
- Base model: `distilbert-base-uncased` (66M parameters)
- Classification head: Linear(768 → 256) → GELU → Dropout → Linear(256 → 3)
- Input format: `[CLS] answer [SEP] context [SEP]`
- Output: 3-class logits → SUPPORTED / PARTIAL / UNSUPPORTED

**Training data construction (from HotpotQA):**
```
SUPPORTED   (label 0) → supporting sentences + correct gold answer
UNSUPPORTED (label 2) → supporting sentences + random wrong answer
PARTIAL     (label 1) → distractor sentences + correct gold answer
```
- 30,000 HotpotQA examples → 90,000+ training triples (3 per example)
- No human annotation needed — HotpotQA supporting_facts used directly
- 90/10 train/val split

**Training configuration:**
- Optimizer: AdamW with separate LR for BERT body (2e-5) and head (2e-4)
- Scheduler: Linear warmup + decay
- Loss: CrossEntropyLoss with class weights [1.0, 1.2, 1.5] + label smoothing 0.05
- Batch size: 32 | Epochs: up to 8 with early stopping (patience=3)
- Mixed precision: enabled (GradScaler)

**Results achieved:**
| Metric | Target | Achieved |
|--------|--------|----------|
| Macro F1 | ≥ 0.70 | **0.9030** |
| Accuracy | — | **88.2%** |
| SUPPORTED F1 | — | **0.9062** |
| PARTIAL F1 | — | **0.8376** |
| UNSUPPORTED F1 | — | **0.9046** |

**Files generated:**
- `verifier_model.pt` — trained model weights + config
- `verifier_data_bert.pkl` — training data cache
- `verifier_history.json` — per-epoch loss and F1 scores
- `verifier_eval_results.json` — final evaluation metrics

---

### Stage 3: Adaptive Retrieval (`adaptive_retrieval.py`)

**What it does:**
- Classifies each query into SIMPLE, MULTI_HOP, or COMPARISON type
- Selects the most appropriate retrieval strategy based on query type
- Integrates Stage 2 verifier to show verification alongside answers

**Query complexity classifier:**

| Query Type | Detection Method | Example |
|------------|-----------------|---------|
| `COMPARISON` | Comparison words + 2 named entities | "Were X and Y of the same nationality?" |
| `MULTI_HOP` | Multi-hop trigger phrases | "Who directed the film that stars..." |
| `SIMPLE` | Default (no above signals) | "What year was X built?" |

**Three retrieval strategies:**

1. **SIMPLE** → Standard top-k dense retrieval (same as Stage 1)

2. **MULTI_HOP** → Iterative retrieval across up to 3 hops
   - Hop 1: retrieve for original query
   - Hop 2: reformulate query using top passage → retrieve again
   - Hop 3: further reformulation → retrieve again
   - Accumulates context across all hops

3. **COMPARISON** → Parallel retrieval for both entities
   - Extract 2 named entities from query
   - Retrieve passages for each entity separately
   - Also retrieve for full query to catch joint passages
   - Merge all results

**Files generated:**
- `stage3_results.json` — Stage 1 vs Stage 3 comparison

---

### Stage 4: Agentic Decision Loop (`agentic_loop.py`)

**What it does:**
- Integrates all previous stages into a single self-correcting pipeline
- Implements the full agentic loop from proposal Equation 2.9
- Re-retrieves with reformulated queries when verification fails
- Abstains rather than returning a hallucinated answer

**Agentic loop flow:**
```
For each iteration t (max 3):
  1. RETRIEVE  → adaptive retrieval (Stage 3)
  2. GENERATE  → Ollama LLM answer generation (Stage 1)
  3. VERIFY    → DistilBERT verifier (Stage 2)

  Decision:
  → SUPPORTED + confidence ≥ 0.6  : return answer ✅
  → PARTIAL / UNSUPPORTED          : reformulate query → next iteration ⚠️
  → Max iterations reached         : ABSTAIN ❌
```

**Query reformulation strategies:**
- Iteration 1: Extract named entities from original query
- Iteration 2: Use top retrieved passage title + query keywords
- Iteration 3: Rephrase as direct entity biography lookup

**Three possible outcomes:**

```
✅ SUPPORTED  — Answer verified, returned to user
⚠️  PARTIAL    — Triggers re-retrieval with reformulated query
❌  ABSTAINED  — Max iterations reached, system refuses to hallucinate
```

**Evaluation — Table 6.2 from proposal:**

Run `python agentic_loop.py --eval --samples 50` to generate:

| Metric | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|--------|---------|---------|---------|---------|
| Exact Match | baseline | — | — | — |
| Hallucination Rate | baseline | — | — | reduced |
| Abstention Rate | 0% | — | — | non-zero |

**Files generated:**
- `stage4_full_results.json` — full 4-stage comparison results

---

## Setup Instructions (Windows — CUDA GPU)

### Prerequisites
- Python 3.11 — download from https://python.org/downloads/release/python-3119/
- [Ollama](https://ollama.com) installed
- NVIDIA GPU with CUDA (RTX 3050 or better recommended)
- 10GB+ free disk space

### Step 1 — Clone and navigate
```powershell
git clone https://github.com/rijuphaiju/Agentic-RAG
cd "Agentic-RAG"
git checkout Manseez
```

### Step 2 — Create virtual environment
```powershell
py -3.11 -m venv venv311
venv311\Scripts\activate
```

### Step 3 — Install PyTorch with CUDA
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4 — Install remaining dependencies
```powershell
pip install -r requirements.txt
```

### Step 5 — Fix known version issues
```powershell
pip install sentence-transformers==2.7.0
pip install datasets==2.19.0
```

### Step 6 — Pull Ollama model
```powershell
ollama pull llama3.2
```

---

## Running the Project

### Every session — activate venv and start Ollama
```powershell
# Terminal 1 — keep open
ollama serve

# Terminal 2 — your working terminal
cd "D:\Projects\8th SEM\Agentic-RAG"
venv311\Scripts\activate
```

### Stage 1 — Basic RAG
```powershell
python rag_pipeline.py
# First run builds FAISS index (~15-20 mins)
# Subsequent runs load from disk instantly
```

### Stage 2 — Train Verifier (once only)
```powershell
python verifier_gpu.py --mode train   # ~20-30 mins on GPU
python verifier_gpu.py --mode test    # quick sanity check
python verifier_gpu.py --mode eval    # evaluate on HotpotQA
```

### Stage 3 — Adaptive Retrieval
```powershell
python adaptive_retrieval.py
```

### Stage 4 — Full Agentic Loop
```powershell
python agentic_loop.py                        # interactive demo
python agentic_loop.py --eval --samples 50    # full evaluation
```

---

## Files NOT in Repository (Auto-generated)

These are too large to push and are rebuilt automatically:

| File | Size | Rebuilt By |
|------|------|-----------|
| `faiss_index.bin` | ~600MB | `python rag_pipeline.py` |
| `passages.pkl` | ~500MB | `python rag_pipeline.py` |
| `verifier_model.pt` | ~250MB | `python verifier_gpu.py --mode train` |
| `verifier_data_bert.pkl` | ~200MB | `python verifier_gpu.py --mode train` |

---

## Known Issues and Fixes

| Issue | Fix |
|-------|-----|
| `sentence-transformers` crashes silently | `pip install sentence-transformers==2.7.0` |
| `datasets` version conflict | `pip install datasets==2.19.0` |
| `torch.cuda.is_available()` returns False | Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| Ollama connection error | Start Ollama: run `ollama serve` in separate terminal |
| FAISS index not found | Delete `.bin` and `.pkl` files and rerun `python rag_pipeline.py` |
| Silent script exit | Check Python version — must be 3.11, not 3.14 |

---

## Hardware Used for Development

- OS: Windows 11
- CPU: Intel Core i7 (13th gen)
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- RAM: 16GB
- Python: 3.11.9
- CUDA: 12.1

---

## Acknowledgements

Supervised by **Anish Baral**, Lecturer, Dept. of Computer and Electronics Engineering, Khwopa Engineering College.

Team: Manseez Bahadur Pradhan, Pranaya Basukala, Riju Phaiju, Salon Raut.

# HARA — Hallucination-Aware Retrieval Agent

**Purbanchal University | Khwopa Engineering College**
**Bachelor of Engineering in Computer Engineering — 8th Semester**

> A modular Retrieval-Augmented Generation system that reduces hallucinations through adaptive retrieval and a self-verification model trained from scratch.

---

## Team Members
| Name | Roll No | Branch |
|------|---------|--------|
| Manseez Bahadur Pradhan | 780320 | Manseez |
| Pranaya Basukala | 780327 | Pranaya |
| Riju Phaiju | 780330 | Riju |
| Salon Raut | 780337 | Salon |

**Supervisor:** Anish Baral, Lecturer — Dept. of Computer and Electronics Engineering

---

## Project Overview

This system is built in 4 stages:

| Stage | File | Description |
|-------|------|-------------|
| 1 | `rag_pipeline.py` | Basic RAG — retrieve + generate (baseline) |
| 2 | `verifier_gpu.py` | Faithfulness verifier — DistilBERT fine-tuned |
| 3 | `adaptive_retrieval.py` | Query complexity classifier + adaptive retrieval |
| 4 | `agentic_loop.py` | Self-correcting loop with abstention mechanism |

### How it works
```
User Query
    ↓
Query Classifier  →  SIMPLE / MULTI_HOP / COMPARISON
    ↓
Adaptive Retrieval  →  FAISS vector search (150k Wikipedia passages)
    ↓
LLM Generation  →  Ollama (llama3.2, runs locally)
    ↓
Faithfulness Verifier  →  SUPPORTED / PARTIAL / UNSUPPORTED
    ↓
Agentic Decision  →  Return Answer / Re-retrieve / Abstain
```

### Key Results
- Verifier Macro F1: **0.9030** (target was ≥ 0.70)
- Verifier Accuracy: **88.2%**
- System abstains rather than hallucinating when evidence is insufficient

---

## Setup Instructions

### Prerequisites
- Python 3.11 (required — PyTorch does not support Python 3.14)
- [Ollama](https://ollama.com) installed
- At least 8GB RAM
- 10GB free disk space

---

### Windows Setup (CUDA GPU)

```powershell
# 1. Clone the repository
git clone https://github.com/rijuphaiju/Agentic-RAG
cd Agentic-RAG
git checkout Manseez

# 2. Create virtual environment with Python 3.11
py -3.11 -m venv venv311
venv311\Scripts\activate

# 3. Install PyTorch with CUDA support (RTX GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Pull Ollama model
ollama pull llama3.2
```

---

### Mac M4 (Apple Silicon) Setup

```bash
# 1. Clone the repository
git clone https://github.com/rijuphaiju/Agentic-RAG
cd Agentic-RAG
git checkout Manseez

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch (Apple Silicon — MPS backend)
pip install torch torchvision torchaudio

# 4. Install remaining dependencies
pip install sentence-transformers==2.7.0
pip install datasets==2.19.0
pip install faiss-cpu
pip install transformers ollama tqdm numpy
pip install accelerate

# 5. Pull Ollama model
# Download Ollama from https://ollama.com first, then:
ollama pull llama3.2
```

> **Note for Mac M4:** The code automatically detects Apple Silicon and uses the MPS backend instead of CUDA. Training will be slower than an RTX GPU but faster than CPU.

---

### Linux Setup (CPU or GPU)

```bash
# 1. Clone the repository
git clone https://github.com/rijuphaiju/Agentic-RAG
cd Agentic-RAG
git checkout Manseez

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install PyTorch
# For CPU:
pip install torch torchvision torchaudio
# For CUDA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Pull Ollama model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

---

## Running the Project

### Important — Start Ollama First
```bash
# Run in a separate terminal and keep it open
ollama serve
```

---

### Stage 1 — Basic RAG (Baseline)
```bash
python rag_pipeline.py
```
First run builds the FAISS index (~15-20 mins). Subsequent runs load from disk instantly.

- Type a question to get an answer
- Type `eval` to run baseline evaluation
- Type `quit` to exit

---

### Stage 2 — Faithfulness Verifier

```bash
# Train the verifier (only needed once — ~20-30 mins on GPU)
python verifier_gpu.py --mode train

# Test on sample sentences
python verifier_gpu.py --mode test

# Evaluate on HotpotQA validation set
python verifier_gpu.py --mode eval
```

Expected result: Macro F1 ≥ 0.70

---

### Stage 3 — Adaptive Retrieval

```bash
python adaptive_retrieval.py
```

- Type a question — system classifies it and picks retrieval strategy
- Type `eval` to compare Stage 1 vs Stage 3
- Type `quit` to exit

---

### Stage 4 — Full Agentic Loop

```bash
# Interactive demo
python agentic_loop.py

# Full 4-stage evaluation (recommended: 50 samples)
python agentic_loop.py --eval --samples 50

# Larger evaluation (100 samples — more accurate results)
python agentic_loop.py --eval --samples 100
```

---

## Sample Questions to Test

### Comparison Questions
```
Were Scott Derrickson and Ed Wood of the same nationality?
Which magazine was started first, Arthur's Magazine or First for Women?
Who is older, Elvis Presley or Michael Jackson?
Which band was formed first, Radiohead or Coldplay?
```

### Multi-Hop Questions
```
Who directed the film Sinister?
Who wrote the novel that the film The Shining is based on?
Who founded the company that makes the iPhone?
What nationality is the director of the film Saw?
```

### Simple Questions
```
What year was the Eiffel Tower built?
Who wrote the play Hamlet?
When was Wikipedia founded?
Where was Albert Einstein born?
```

---

## Expected Output

**Answer verified in 1 iteration:**
```
Iteration 1/3
Verification: ✅ SUPPORTED (confidence: 0.94)
FINAL STATUS: SUPPORTED
FINAL ANSWER: Scott Derrickson directed the film Sinister.
```

**Self-correction after re-retrieval:**
```
Iteration 1/3 → ⚠️ PARTIAL → reformulate query
Iteration 2/3 → ✅ SUPPORTED
FINAL STATUS: SUPPORTED
```

**Principled abstention:**
```
Iteration 1/3 → ⚠️ PARTIAL
Iteration 2/3 → ⚠️ PARTIAL
Iteration 3/3 → ⚠️ PARTIAL
FINAL STATUS: ABSTAINED
(System chose silence over hallucination)
```

> **Note:** ABSTAINED results are correct behavior — the system is working as designed by refusing to return unverified answers.

---

## Files Generated After Setup

These files are NOT in the repository (too large) and will be generated automatically:

| File | Size | Generated By |
|------|------|-------------|
| `faiss_index.bin` | ~500MB | `rag_pipeline.py` (first run) |
| `passages.pkl` | ~500MB | `rag_pipeline.py` (first run) |
| `verifier_model.pt` | ~250MB | `verifier_gpu.py --mode train` |
| `verifier_data_bert.pkl` | ~200MB | `verifier_gpu.py --mode train` |

---

## Project Structure

```
Agentic-RAG/
├── rag_pipeline.py          ← Stage 1: Basic RAG
├── verifier_gpu.py          ← Stage 2: Faithfulness Verifier
├── adaptive_retrieval.py    ← Stage 3: Adaptive Retrieval
├── agentic_loop.py          ← Stage 4: Agentic Decision Loop
├── requirements.txt         ← Python dependencies
├── sample_questions.txt     ← Test questions with expected outputs
└── README.md                ← This file
```

---

## Dataset

This project uses [HotpotQA](https://hotpotqa.github.io/) — a multi-hop question answering dataset with 90,447 training examples from Wikipedia. The dataset is downloaded automatically on first run via HuggingFace Datasets.

---

## References

- Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Asai et al. (2024) — Self-RAG: Learning to Retrieve, Generate, and Critique
- Dhuliawala et al. (2024) — Chain-of-Verification Reduces Hallucination in LLMs
- Yang et al. (2018) — HotpotQA: A Dataset for Diverse, Explainable Multi-Hop QA
- Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers

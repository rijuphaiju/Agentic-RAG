# HARA — Hallucination-Aware Retrieval Agent

A multi-stage Retrieval-Augmented Generation (RAG) pipeline built on HotpotQA that progressively reduces hallucinations through retrieval quality improvements, faithfulness verification, adaptive query routing, and a self-correcting agentic loop.

**Capstone Project — Purbanchal University**

---

## Overview

Standard RAG systems retrieve documents and generate answers in a single pass, with no mechanism to detect or correct hallucinated output. This project addresses that gap through four incremental pipeline stages:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `Stage_1_RAG_Pipeline.py` | Basic RAG — FAISS dense retrieval + Ollama LLM generation |
| 2 | `Stage_2_RAG_Pipeline.py` | RAG + faithfulness verifier — DistilBERT classifier flags unsupported answers |
| 3 | `Stage_3_Adaptive_Retrieval.py` | Adaptive retrieval — query-type routing (SIMPLE / MULTI_HOP / COMPARISON) with BM25 hybrid scoring and CrossEncoder reranking |
| 4 | `Stage_4_Agentic_Loop.py` | Agentic loop — self-correcting loop that reformulates queries and re-retrieves when verification fails |

### Key Features

- **Hybrid retrieval** — BM25 keyword scoring fused with FAISS dense embeddings (alpha=0.5)
- **CrossEncoder reranking** — `ms-marco-MiniLM-L-6-v2` reranks candidate passages before generation
- **Query-type routing** — classifies each question as SIMPLE, MULTI_HOP, or COMPARISON and applies a tailored retrieval strategy
- **Multi-hop decomposition** — MULTI_HOP queries are split into sub-questions; the bridge answer is passed as explicit context to the LLM
- **DistilBERT verifier** — fine-tuned faithfulness classifier labels answers as SUPPORTED / PARTIAL / UNSUPPORTED
- **Self-consistency check** — Stage 4 uses a stochastic re-answer to validate PARTIAL verdicts before accepting
- **React chatbot UI** — dark-themed interface with stage selector, verification badges, and an evaluation results table
- **Full evaluation suite** — computes Exact Match, Precision, Recall, F1, Macro F1, Hallucination Rate, and Abstention Rate across all four stages

---

## Requirements

### System

- Python **3.11**
- CUDA-capable GPU (recommended — required for fast verifier training; CPU fallback is available)
- Node.js **18+** and npm (for the React frontend)
- [Ollama](https://ollama.com) installed and running locally

### Python Dependencies

```
torch
faiss-gpu        # Windows/Linux with CUDA
faiss-cpu        # Mac or CPU-only systems
sentence-transformers
transformers
datasets
fastapi
uvicorn
ollama
rank-bm25
numpy
tqdm
```

---

## Installation

### Windows

**1. Install Ollama**

Download and run the installer from [https://ollama.com](https://ollama.com).

**2. Clone the repository**

```powershell
git clone https://github.com/rijuphaiju/Agentic-RAG.git
cd Agentic-RAG
git checkout Manseez
```

**3. Create and activate a Python 3.11 virtual environment**

```powershell
python -m venv venv311
.\venv311\Scripts\Activate.ps1
```

> If activation is blocked by execution policy, run:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**4. Install Python dependencies**

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install faiss-gpu sentence-transformers transformers datasets
pip install fastapi uvicorn ollama rank-bm25 numpy tqdm
```

> If you do not have a CUDA GPU, replace `faiss-gpu` with `faiss-cpu` and install the CPU build of PyTorch from [https://pytorch.org](https://pytorch.org).

**5. Install frontend dependencies**

```powershell
cd frontend
npm install
cd ..
```

---

### Mac

**1. Install Ollama**

```bash
brew install ollama
```

Or download the Mac app from [https://ollama.com](https://ollama.com).

**2. Clone the repository**

```bash
git clone https://github.com/rijuphaiju/Agentic-RAG.git
cd Agentic-RAG
git checkout Manseez
```

**3. Create and activate a Python 3.11 virtual environment**

```bash
python3.11 -m venv venv311
source venv311/bin/activate
```

**4. Install Python dependencies**

```bash
pip install torch torchvision torchaudio
pip install faiss-cpu sentence-transformers transformers datasets
pip install fastapi uvicorn ollama rank-bm25 numpy tqdm
```

> Mac does not support CUDA. Use `faiss-cpu` only.

**5. Install frontend dependencies**

```bash
cd frontend
npm install
cd ..
```

---

## Running the Project

> **All steps below assume the virtual environment is activated.**
> Windows: `.\venv311\Scripts\Activate.ps1`
> Mac: `source venv311/bin/activate`

### First-time setup (Steps 1–3 only need to run once)

---

**Step 1 — Start Ollama** *(required before any other step — use a dedicated terminal)*

```bash
ollama serve
```

Then, in a second terminal, pull the LLM model:

```bash
ollama pull llama3.2
```

Leave the `ollama serve` terminal open for the entire session.

---

**Step 2 — Build the FAISS index** *(first time only — approximately 10–20 minutes)*

```bash
python Stage_1_RAG_Pipeline.py
```

This downloads the HotpotQA dataset, embeds all passages, and saves `faiss_index.bin` and `passages.pkl` to disk. Type `quit` to exit the interactive demo once the index is built.

---

**Step 3 — Train the faithfulness verifier** *(first time only — approximately 10–30 minutes on GPU)*

```bash
python Stage_2_Verifier_GPU.py --mode train
```

This fine-tunes a DistilBERT model and saves the verifier weights to `verifier_model/`.

---

### Running evaluation

**Step 4 — Evaluate all four pipeline stages on HotpotQA**

```bash
# Quick test — approximately 5–10 minutes
python Stage_6_Evaluation.py --samples 50

# Full evaluation — approximately 20–30 minutes
python Stage_6_Evaluation.py --samples 150
```

Results are printed as a formatted table and saved to `evaluation_results.json`.

---

### Running the chatbot UI

**Step 5 — Start the backend and frontend** *(requires two separate terminals)*

**Terminal 1 — FastAPI backend**

```bash
python Stage_5_API.py
```

**Terminal 2 — React frontend**

```bash
cd frontend
npm run dev
```

Open your browser and go to: **http://localhost:5173**

---

## Project Structure

```
HARA/
├── Stage_1_RAG_Pipeline.py        # FAISS index builder, dense retrieval, answer generation
├── Stage_1_RAG_Pipeline_GPU.py    # GPU-optimised variant of Stage 1
├── Stage_2_RAG_Pipeline.py        # Stage 1 + faithfulness verification
├── Stage_2_RAG_Pipeline_GPU.py    # GPU-optimised variant of Stage 2
├── Stage_2_Verifier.py            # DistilBERT verifier (CPU)
├── Stage_2_Verifier_GPU.py        # DistilBERT verifier training and inference (GPU)
├── Stage_3_Adaptive_Retrieval.py  # Query classification, BM25 hybrid, CrossEncoder reranking
├── Stage_4_Agentic_Loop.py        # Self-correcting agentic loop with query reformulation
├── Stage_5_API.py                 # FastAPI backend serving all pipeline stages
├── Stage_6_Evaluation.py          # Full evaluation: EM, F1, Hallucination Rate, Abstention Rate
├── frontend/                      # React + Vite chatbot UI
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ChatWindow.jsx
│   │   │   ├── ChatMessage.jsx
│   │   │   ├── ChatInput.jsx
│   │   │   ├── Sidebar.jsx
│   │   │   └── EvalTable.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
├── run_tests.py                   # Unit tests for pipeline components
└── evaluation_results.json        # Latest evaluation output
```

---

## Notes

- **Large binary files are excluded from the repository.** `faiss_index.bin` (~220 MB), `passages.pkl` (~79 MB), and verifier model weights are listed in `.gitignore`. They are rebuilt locally by running Steps 2 and 3 above.
- **Mac users** must use `faiss-cpu`. The GPU variants of Stage 1 and Stage 2 will fall back to CPU automatically.
- **Windows users** should always use the `venv311` environment. The parent project `venv` (if present) does not contain PyTorch or the required ML libraries.
- **Ollama must be running** (`ollama serve`) before starting the API server or running evaluation. Calls to the LLM will fail silently if Ollama is not active.
- The default Ollama model is `llama3.2`. To use a different model, change the `OLLAMA_MODEL` constant in `Stage_1_RAG_Pipeline.py`.

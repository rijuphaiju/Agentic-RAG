"""
Stage 2: Faithfulness Verifier -- DistilBERT Edition (GPU Accelerated)
=======================================================================
Project: HARA — Hallucination-Aware Retrieval Agent

Key upgrade over custom transformer:
  - DistilBERT encoder replaces custom tokenizer + transformer trained from scratch
  - DistilBERT already understands language -- fine-tuning converges in 3-5 epochs
  - Expected Macro F1: 0.70+ vs ~0.48 with custom model
  - Same --mode train / test / eval interface as before
  - verifier_model.pt format compatible with stage2_rag_pipeline_gpu.py
    (load_verifier() returns model + tokenizer together)

GPU notes:
  - DistilBERT fine-tuning on RTX 4050: ~8-12 min per epoch at batch_size=32
  - Mixed precision (AMP) enabled for ~1.5x speedup
  - Gradient checkpointing disabled (not needed at this scale)
"""

import argparse
import json
import os
import pickle
import random
import re
import string

import numpy as np
import torch
import torch.nn as nn
import faiss  # must load before datasets on Windows
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# DistilBERT tokenizer + model
from transformers import DistilBertTokenizerFast, DistilBertModel

# 
# CONFIG
# 
DEVICE         = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
# ── M4 training config — target: 1–2 hours ───────────────────────────────
# MAX_SEQ_LEN 192 vs 256: self-attention is O(n²), so 192 = 56% of 256's compute
# — biggest single speedup lever.  Context is ~800 chars ≈ 120 tokens; answers
# are ≤12 words; 192 covers both with headroom.
# TRAIN_SAMPLES 12000 × ~12 examples/sample = ~144k total — similar dataset size
# to the old 30k × 5 = 150k, but with much better label diversity.
# EPOCHS 5 + EARLY_STOP_PATIENCE 2: DistilBERT fine-tunes fast; with LR=3e-5
# it typically converges in 3–4 epochs.
MAX_SEQ_LEN    = 128   # 128 vs 192: attention is O(n²) → 2.25× faster; answers fit in 128
NUM_CLASSES    = 3
BATCH_SIZE     = 96 if DEVICE == "mps" else (32 if DEVICE == "cuda" else 8)
EPOCHS         = 4
LR             = 3e-5
TRAIN_SAMPLES  = 4000  # 4000 × ~13 examples = ~52k total; enough to learn the patterns
VERIFIER_PATH  = "verifier_model.pt"
DATA_PATH      = "verifier_data_bert.pkl"
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 2   # stop after 2 epochs with no improvement
DROPOUT        = 0.2

LABEL_MAP    = {0: "SUPPORTED", 1: "PARTIAL", 2: "UNSUPPORTED"}
LABEL_COLORS = {0: "[OK]", 1: "[!!]", 2: "[X]"}
BERT_MODEL   = "distilbert-base-uncased"

print(f"[Device] Using: {DEVICE.upper()}"
      + (f" -- {torch.cuda.get_device_name(0)}" if DEVICE == "cuda"
         else " -- Apple Silicon GPU (MPS)" if DEVICE == "mps"
         else " (CPU fallback)"))

# M4 Air optimisations
if DEVICE == "mps":
    torch.set_float32_matmul_precision("high")          # faster matmul on Apple Silicon
    if hasattr(torch.backends.mps, "enable_flash_sdp"):
        torch.backends.mps.enable_flash_sdp(True)       # Flash Attention on MPS (PyTorch 2.2+)


# 
# STEP 1: BUILD TRAINING DATA
# 
def _get_other_entity(question: str, gold: str) -> str | None:
    """Extract the comparison counterpart entity from a HotpotQA comparison question.

    Example: question = "Which band was founded first, Hole or The Wolfhounds?"
             gold     = "The Wolfhounds"
             returns  → "Hole"

    Used to generate UNSUPPORTED counterexamples for comparison questions so the
    verifier learns the reasoning boundary (not just surface overlap).
    """
    gold_lower = gold.lower()
    skip = {
        'which', 'who', 'were', 'was', 'what', 'when', 'where', 'how', 'did',
        'the', 'both', 'and', 'or', 'of', 'a', 'an', 'is', 'are', 'be',
    }
    # Find all capitalised phrases (1–4 words) in the question
    candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b', question)
    for c in candidates:
        if c.lower() in skip:
            continue
        if len(c) <= 2:
            continue
        if c.lower() in gold_lower or gold_lower in c.lower():
            continue
        return c
    return None


def build_training_data(num_samples=TRAIN_SAMPLES):
    """Build verifier training data from HotpotQA training split.

    Label definitions (what the verifier should learn):
    ─────────────────────────────────────────────────────
    SUPPORTED  — the context contains sufficient evidence to conclude the answer
                 is correct.  Includes verbatim matches AND logically inferred
                 conclusions (comparison results, multi-hop chains).

    PARTIAL    — the answer is partially grounded: either the context is incomplete
                 (distractor passages) OR the answer makes additional claims beyond
                 what the context supports.

    UNSUPPORTED — the answer contradicts the context or has no grounding in it.
                  Includes the wrong comparison entity with a confident assertion.

    Input format: "Q: {question} A: {answer}" vs context
    ─────────────────────────────────────────────────────
    Including the question lets the verifier understand what is being claimed, not
    just whether a string appears in the context.  This is what enables it to
    validate logical inference ("X was founded first" when context has both dates).
    """
    print("Loading HotpotQA training split...")
    dataset     = load_dataset("hotpot_qa", "distractor", split="train")
    examples    = list(dataset)
    all_answers = [ex["answer"] for ex in examples]

    data = []
    print(f"Building verifier training data from {num_samples} examples...")

    for ex in tqdm(examples[:num_samples]):
        gold      = ex["answer"]
        question  = ex["question"]
        qtype     = ex.get("type", "bridge")   # "bridge" or "comparison"
        titles    = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        sf_titles = set(ex["supporting_facts"]["title"])

        supporting, distractor = [], []
        for title, sents in zip(titles, sentences):
            block = " ".join(sents)
            (supporting if title in sf_titles else distractor).append(block)

        sup_ctx  = " ".join(supporting)[:800]
        dist_ctx = " ".join(distractor)[:800] if distractor else sup_ctx

        # Convenience: format answer with question prefix — this is the input format
        # the verifier will also receive at inference time via verify(..., question=q).
        def qa(ans: str) -> str:
            return f"Q: {question} A: {ans}"

        # ── SUPPORTED ─────────────────────────────────────────────────────────
        # Verbatim gold
        data.append({"context": sup_ctx, "answer": qa(gold), "label": 0})

        # Sentence-form paraphrases — what the LLM actually generates.
        # Previously these were mislabelled PARTIAL, teaching "sentence form = PARTIAL".
        for p in [
            f"The answer is {gold}.",
            f"It is {gold}.",
            f"Final Answer: {gold}",
        ]:
            data.append({"context": sup_ctx, "answer": qa(p), "label": 0})

        if len(gold.split()) <= 4:
            data.append({"context": sup_ctx, "answer": qa(f"{gold} is the answer."), "label": 0})
            data.append({"context": sup_ctx, "answer": qa(f"That would be {gold}."), "label": 0})

        # Comparison-specific SUPPORTED: inferred conclusions.
        # The verifier must learn that "{entity} won more / was founded first" is SUPPORTED
        # when the context contains evidence for both entities and gold is the correct one.
        if qtype == "comparison" and gold[:1].isupper() and len(gold.split()) <= 4:
            for tmpl in [
                f"{gold} was founded first.",
                f"{gold} was formed earlier.",
                f"{gold} is older.",
                f"{gold} won more.",
                f"{gold} has more.",
                f"{gold} came first.",
                f"{gold} is the answer because the evidence supports it.",
            ]:
                data.append({"context": sup_ctx, "answer": qa(tmpl), "label": 0})

        # ── UNSUPPORTED ────────────────────────────────────────────────────────
        wrong = random.choice(all_answers)
        while wrong.lower() == gold.lower():
            wrong = random.choice(all_answers)
        data.append({"context": sup_ctx, "answer": qa(wrong), "label": 2})
        data.append({"context": sup_ctx, "answer": qa(f"The answer is {wrong}."), "label": 2})

        # Comparison-specific UNSUPPORTED: the other entity with confident assertion.
        # This teaches the verifier the boundary for comparison reasoning:
        # "Hole was founded first" is UNSUPPORTED when Wolfhounds' date is earlier.
        if qtype == "comparison":
            other = _get_other_entity(question, gold)
            if other:
                for tmpl in [
                    other,
                    f"The answer is {other}.",
                    f"{other} was founded first.",
                    f"{other} was formed earlier.",
                    f"{other} won more.",
                ]:
                    data.append({"context": sup_ctx, "answer": qa(tmpl), "label": 2})

        # ── PARTIAL ────────────────────────────────────────────────────────────
        # Correct answer but incomplete/distractor context
        data.append({"context": dist_ctx, "answer": qa(gold), "label": 1})
        data.append({"context": dist_ctx, "answer": qa(f"The answer is {gold}."), "label": 1})

        # Truncated answer (first word only of multi-word gold)
        partial_ans = gold.split()[0]
        if partial_ans.lower() != gold.lower():
            data.append({"context": sup_ctx, "answer": qa(partial_ans), "label": 1})

        # Answer makes extra unverifiable claim alongside correct gold
        extra = random.choice(all_answers)
        while extra.lower() == gold.lower():
            extra = random.choice(all_answers)
        data.append({"context": sup_ctx, "answer": qa(f"{gold} and also {extra}."), "label": 1})

    random.shuffle(data)
    counts = {0: 0, 1: 0, 2: 0}
    for d in data:
        counts[d["label"]] += 1
    print(f"Dataset built: {len(data)} examples")
    print(f"  SUPPORTED={counts[0]} | PARTIAL={counts[1]} | UNSUPPORTED={counts[2]}")
    return data


# 
# STEP 2: DATASET
# 
class VerifierDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=MAX_SEQ_LEN):
        # Pre-tokenize entire split once on CPU.
        # IMPORTANT: do NOT move to MPS here.  The DataLoader accesses individual
        # rows via __getitem__ — if tensors live on MPS, every single row-index
        # triggers a full GPU synchronisation (one sync per sample per batch).
        # A batch of 64 items → 64 MPS syncs, which is why 4+ s/batch was observed.
        # Instead, keep data on CPU and do one bulk .to(device) per batch in run_epoch.
        print(f"  Pre-tokenizing {len(data):,} examples (one-time cost)...")
        encoded = tokenizer(
            [d["answer"]  for d in data],
            [d["context"] for d in data],
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Stay on CPU — moved to MPS in run_epoch per batch (one transfer per step)
        self.input_ids      = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels         = torch.tensor([d["label"] for d in data], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label":          self.labels[idx],
        }


# 
# STEP 3: MODEL
# 
class FaithfulnessVerifier(nn.Module):
    """
    DistilBERT encoder + classification head.
    Fine-tuned end-to-end on (context, answer) -> {SUPPORTED, PARTIAL, UNSUPPORTED}.
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.bert      = DistilBertModel.from_pretrained(BERT_MODEL)
        hidden_dim     = self.bert.config.hidden_size   # 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls    = out.last_hidden_state[:, 0, :]   # [CLS] token
        return self.classifier(cls)


# 
# STEP 4: METRICS
# 
def compute_macro_f1(labels, preds, num_classes=NUM_CLASSES):
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)
    return f1s, sum(f1s) / len(f1s)


# 
# STEP 5: TRAINING LOOP
# 
def run_epoch(model, loader, criterion, optimizer=None, scheduler=None,
              scaler=None, desc=""):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = torch.tensor(0.0, device=DEVICE)
    pred_chunks, label_chunks = [], []   # accumulate on-device, sync once per epoch

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            if is_train and scaler is not None:
                # CUDA path — AMP with GradScaler
                with autocast("cuda"):
                    logits = model(input_ids, attn_mask)
                    loss   = criterion(logits, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler:
                    scheduler.step()
            elif DEVICE == "mps":
                # MPS path — pure float32
                logits = model(input_ids, attn_mask)
                loss   = criterion(logits, labels)
                if is_train:
                    optimizer.zero_grad(set_to_none=True)  # faster than zeroing
                    loss.backward()
                    # clip_grad_norm_ calls .item() internally → MPS sync every step; skip it.
                    # LR=2e-5 is conservative enough for stable DistilBERT fine-tuning.
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
            else:
                # CPU path
                logits = model(input_ids, attn_mask)
                loss   = criterion(logits, labels)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()

            total_loss += loss.detach()                    # stays on device — no sync
            pred_chunks.append(logits.detach().argmax(dim=-1))
            label_chunks.append(labels)

    # Single GPU→CPU sync per epoch instead of 3× per step
    avg_loss          = (total_loss / len(loader)).item()
    all_preds         = torch.cat(pred_chunks).cpu().tolist()
    all_labels_cpu    = torch.cat(label_chunks).cpu().tolist()
    acc               = sum(p == l for p, l in zip(all_preds, all_labels_cpu)) / len(all_labels_cpu)
    per_f1, macro_f1  = compute_macro_f1(all_labels_cpu, all_preds)
    return avg_loss, acc, macro_f1, per_f1, all_preds, all_labels_cpu


def train(model, train_loader, val_loader):
    # ── Layer freezing for M4 MPS speed ──────────────────────────────────────
    # DistilBERT has 6 transformer layers (indices 0–5).
    # Freezing first 4 means backward only propagates through layers 4–5 + head.
    # This cuts backward compute by ~60% — the biggest remaining speedup lever on MPS.
    # The last 2 layers + classifier are enough to learn our new label patterns
    # (sentence forms, comparison conclusions) since the frozen layers already
    # encode the semantic representations from the pre-trained model.
    if DEVICE == "mps":
        FREEZE_LAYERS = 4
        for i, layer in enumerate(model.bert.transformer.layer):
            if i < FREEZE_LAYERS:
                for param in layer.parameters():
                    param.requires_grad = False
        frozen_params  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MPS] Frozen first {FREEZE_LAYERS} layers — "
              f"trainable: {trainable_params:,} / total: {frozen_params + trainable_params:,}")

    # PARTIAL gets the highest weight: it's the hardest class (inferred answers)
    # and the most consequential — a false UNSUPPORTED on a correct PARTIAL answer
    # causes unnecessary abstention in the agentic loop.
    weights   = torch.tensor([1.0, 1.8, 1.4]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    # Only pass trainable parameters to optimizer (frozen params excluded)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.bert.parameters()       if p.requires_grad], "lr": LR},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": LR * 10},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = total_steps // 10
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler         = GradScaler("cuda") if DEVICE == "cuda" else None  # MPS doesn't support GradScaler
    best_macro_f1  = 0.0
    patience_count = 0
    history        = []

    print(f"\n{'='*60}")
    print(f"Fine-tuning DistilBERT Faithfulness Verifier on {DEVICE.upper()}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LR} | AMP: {scaler is not None}")
    print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc, t_f1, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            desc=f"Epoch {epoch}/{EPOCHS} [Train]"
        )
        v_loss, v_acc, v_f1, per_f1, _, _ = run_epoch(
            model, val_loader, criterion,
            desc=f"Epoch {epoch}/{EPOCHS} [Val]  "
        )

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Train -> Loss: {t_loss:.4f}  Acc: {t_acc:.4f}  Macro F1: {t_f1:.4f}")
        print(f"  Val   -> Loss: {v_loss:.4f}  Acc: {v_acc:.4f}  Macro F1: {v_f1:.4f}")
        print(f"  Per-class F1 -> SUPPORTED: {per_f1[0]:.4f} | PARTIAL: {per_f1[1]:.4f} | UNSUPPORTED: {per_f1[2]:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": t_loss, "train_acc": t_acc, "train_macro_f1": t_f1,
            "val_loss":   v_loss, "val_acc":   v_acc, "val_macro_f1":   v_f1,
            "per_class_f1": {
                "SUPPORTED":   per_f1[0],
                "PARTIAL":     per_f1[1],
                "UNSUPPORTED": per_f1[2],
            }
        })

        if v_f1 > best_macro_f1:
            best_macro_f1  = v_f1
            patience_count = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "macro_f1":    v_f1,
                "model_type":  "distilbert",
                "config": {"bert_model": BERT_MODEL, "num_classes": NUM_CLASSES},
            }, VERIFIER_PATH)
            print(f"  [OK] Best model saved (Macro F1: {v_f1:.4f})\n")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{EARLY_STOP_PATIENCE})\n")
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best Macro F1: {best_macro_f1:.4f}")
                break

    with open("verifier_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best Macro F1: {best_macro_f1:.4f}")
    print(f"Model saved -> {VERIFIER_PATH}")
    return history


# 
# STEP 6: INFERENCE API
# 
def load_verifier(model_path=VERIFIER_PATH):
    checkpoint   = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model        = FaithfulnessVerifier().to(DEVICE)
    state_dict   = checkpoint["model_state"]
    # torch.compile() prefixes all keys with "_orig_mod." — strip it when loading
    # without compile so the model loads correctly regardless of how it was saved.
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    bert_tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL)
    print(f"Verifier loaded from {model_path} "
          f"(epoch {checkpoint['epoch']}, Macro F1: {checkpoint['macro_f1']:.4f})")
    return model, bert_tokenizer


def build_verify_context(passages: list, answer: str, top_n: int = 5) -> str:
    """Build a verification context string from a passage list.

    Passages that contain the answer string are moved to the front so they
    appear within the verifier's MAX_SEQ_LEN=256 token window.  Without this,
    the supporting passage is often ranked 6th+ (reranked by query similarity,
    not answer presence) and gets truncated out — causing PARTIAL even when the
    answer is 100% correct.
    """
    a_lower    = answer.lower().strip()
    containing = [p for p in passages if a_lower in p["text"].lower()]
    others     = [p for p in passages if a_lower not in p["text"].lower()]
    ordered    = (containing + others)[:top_n]
    return " ".join(p["text"] for p in ordered)


_EVASIVE_PATTERNS = re.compile(
    r'\b(not specified|not mentioned|no information|cannot determine|'
    r'cannot be determined|not clear|not provided|not available|'
    r'no direct connection|likely based elsewhere|based elsewhere|'
    r'it can be inferred|it is unclear|unclear from|insufficient|'
    r'not explicitly|does not specify|does not mention)\b',
    re.IGNORECASE,
)


_CLAUSE_BOUNDARY_RE = re.compile(
    r',\s+(?:and|but|while|with|also|as well as)\b|\s+(?:and|but|while)\s+',
    re.IGNORECASE,
)


def _truncate_at_clause_boundary(text: str, max_words: int) -> str:
    """Shortens `text` to at most `max_words`, preferring to cut at the last
    clause boundary (comma+conjunction, or a bare conjunction) within that
    budget rather than a strict word-count cut.

    A hard word-count cut risks leaving a dangling, incomplete trailing
    phrase — e.g. truncating "...for Women, with its publication beginning
    in 1844." at 12 words produces "...with its publication beginning",
    a fragment with no evaluable content. Verification code (both the
    verifier itself and the Stage 2 label-generation cascade) can misread
    such a fragment as a separate "extra claim" that fails a groundedness
    check purely because it's incomplete, not because it's a fabricated
    fact — manufacturing a spurious PARTIAL/UNSUPPORTED signal. Cutting at
    the nearest complete clause boundary instead avoids this.

    Falls back to the plain word-count cut when no clause boundary exists
    within the budget (a single long clause with no natural break) — this
    is a best-effort improvement, not a guarantee for every input.
    """
    words = text.split()
    if len(words) <= max_words:
        return text.strip()

    fallback = " ".join(words[:max_words])
    boundaries = [m.start() for m in _CLAUSE_BOUNDARY_RE.finditer(text)]
    valid_boundaries = [b for b in boundaries if len(text[:b].split()) <= max_words]
    if valid_boundaries:
        clause = text[:max(valid_boundaries)].strip().rstrip(",")
        if clause:
            return clause
    return fallback


def _distill_for_verify(answer: str, max_words: int = 12) -> str:
    """Shorten a verbose LLM answer to a compact phrase before verification.

    Normal case: distill to a clause-boundary-aware ~12-word phrase so the
    verifier sees a short phrase matching its training distribution (gold
    answers are 1-5 words), without leaving a dangling incomplete fragment
    behind (see _truncate_at_clause_boundary).

    Evasive/hedging case: when the LLM says "not specified", "cannot determine",
    "likely based elsewhere" etc., the first 12 words are often a grounded
    preamble ("The director of Big Stone Gap, Adriana...") while the actual
    wrong conclusion is in the middle/end.  Distilling to the preamble causes
    the verifier to say SUPPORTED for a factually wrong answer.
    Fix: detect evasive answers and return the full hedging phrase so the
    verifier sees "not specified" / "cannot determine" → UNSUPPORTED/PARTIAL.
    """
    # If the LLM used a "Final Answer:" tag, extract just that part
    m = re.search(r'(?:Final Answer|Answer)\s*:\s*(.+?)(?:\n|$)', answer, re.IGNORECASE)
    if m:
        answer = m.group(1).strip()

    # Detect hedging/evasive answers — the LLM is saying it doesn't know.
    # Return the conclusion sentence (last sentence) so the verifier sees the
    # actual claim, not just the grounded preamble.
    if _EVASIVE_PATTERNS.search(answer):
        sentences = re.split(r'(?<=[.!?])\s', answer.strip())
        # Find the sentence containing the evasive pattern and return it
        for sent in sentences:
            if _EVASIVE_PATTERNS.search(sent):
                return _truncate_at_clause_boundary(sent.strip(), max_words)

    # Normal case: take first sentence, cap at max_words (clause-boundary-aware)
    first = re.split(r'(?<=[.!?])\s', answer.strip(), maxsplit=1)[0]
    return _truncate_at_clause_boundary(first, max_words)


def verify(context, answer, model, tokenizer, question=None):
    """Single-sample inference.

    Args:
        context:   Retrieved passage text (sequence B).
        answer:    LLM-generated answer — distilled internally before verification.
        question:  The original user question (optional but strongly recommended).
                   When provided, input becomes "Q: {question} A: {answer}" which
                   matches the training format and enables comparison/multi-hop
                   entailment rather than pure lexical overlap.

    The question is critical for comparison questions: without it, the verifier
    cannot know that "The Wolfhounds were founded first" is a logical conclusion
    from two dates in context.  With it, the model sees the same (Q, A, context)
    triple it was trained on and can validate the inference.
    """
    answer_for_verify = _distill_for_verify(answer)
    # Prepend question when available — matches training data format
    seq_a = f"Q: {question} A: {answer_for_verify}" if question else answer_for_verify
    model.eval()
    encoded   = tokenizer(
        seq_a, context,
        max_length=MAX_SEQ_LEN, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attn_mask = encoded["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids, attn_mask)
        probs  = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        pred   = int(torch.tensor(probs).argmax())
    return {
        "label":      LABEL_MAP[pred],
        "icon":       LABEL_COLORS[pred],
        "confidence": probs[pred],
        "scores": {
            "SUPPORTED":   round(probs[0], 4),
            "PARTIAL":     round(probs[1], 4),
            "UNSUPPORTED": round(probs[2], 4),
        }
    }


def verify_batch(contexts, answers, model, tokenizer, batch_size=32):
    """GPU-optimised batched inference."""
    model.eval()
    all_results = []

    for start in range(0, len(contexts), batch_size):
        batch_ctx = contexts[start:start + batch_size]
        batch_ans = answers[start:start + batch_size]

        encoded   = tokenizer(
            batch_ans, batch_ctx,
            max_length=MAX_SEQ_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attn_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attn_mask)
            probs  = torch.softmax(logits, dim=-1).cpu().tolist()

        for p in probs:
            pred = int(np.argmax(p))
            all_results.append({
                "label":      LABEL_MAP[pred],
                "icon":       LABEL_COLORS[pred],
                "confidence": p[pred],
                "scores": {
                    "SUPPORTED":   round(p[0], 4),
                    "PARTIAL":     round(p[1], 4),
                    "UNSUPPORTED": round(p[2], 4),
                }
            })
    return all_results


# 
# STEP 7: EVALUATE ON HOTPOTQA
# 
def evaluate_on_hotpotqa(model, tokenizer, num_samples=500):
    print(f"\nEvaluating verifier on {num_samples} HotpotQA validation examples...")
    dataset     = load_dataset("hotpot_qa", "distractor", split="validation")
    examples    = list(dataset)
    all_answers = [ex["answer"] for ex in examples]

    eval_data = []
    for ex in examples[:num_samples // 3]:
        gold      = ex["answer"]
        titles    = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        sf_titles = set(ex["supporting_facts"]["title"])

        supporting, distractor = [], []
        for title, sents in zip(titles, sentences):
            block = " ".join(sents)
            (supporting if title in sf_titles else distractor).append(block)

        sup_ctx  = " ".join(supporting)[:800]
        dist_ctx = " ".join(distractor)[:800] if distractor else sup_ctx
        wrong    = random.choice(all_answers)

        eval_data.append({"context": sup_ctx,  "answer": gold,  "label": 0})
        eval_data.append({"context": sup_ctx,  "answer": wrong, "label": 2})
        eval_data.append({"context": dist_ctx, "answer": gold,  "label": 1})

    print(f"Running batched inference on {len(eval_data)} samples...")
    contexts = [d["context"] for d in eval_data]
    answers  = [d["answer"]  for d in eval_data]
    labels   = [d["label"]   for d in eval_data]

    results  = verify_batch(contexts, answers, model, tokenizer, batch_size=BATCH_SIZE)
    preds    = [list(LABEL_MAP.values()).index(r["label"]) for r in results]

    per_f1, macro_f1 = compute_macro_f1(labels, preds)
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)

    print(f"\n{'='*60}")
    print(f"Verifier Evaluation Results ({len(eval_data)} samples)")
    print(f"{'='*60}")
    print(f"Accuracy   : {acc:.4f} ({acc*100:.1f}%)")
    print(f"Macro F1   : {macro_f1:.4f}  (target: >= 0.70)")
    print(f"  SUPPORTED   F1: {per_f1[0]:.4f}")
    print(f"  PARTIAL     F1: {per_f1[1]:.4f}")
    print(f"  UNSUPPORTED F1: {per_f1[2]:.4f}")

    with open("verifier_eval_results.json", "w") as f:
        json.dump({
            "accuracy": acc, "macro_f1": macro_f1,
            "per_class_f1": {
                "SUPPORTED":   per_f1[0],
                "PARTIAL":     per_f1[1],
                "UNSUPPORTED": per_f1[2],
            }
        }, f, indent=2)
    print("Results saved -> verifier_eval_results.json")
    return macro_f1


# 
# MAIN
# 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "eval"], default="train")
    args = parser.parse_args()

    print(f"Device: {DEVICE.upper()}")

    if args.mode == "train":
        if os.path.exists(DATA_PATH):
            print("Loading existing training data...")
            with open(DATA_PATH, "rb") as f:
                data = pickle.load(f)
        else:
            data = build_training_data(TRAIN_SAMPLES)
            with open(DATA_PATH, "wb") as f:
                pickle.dump(data, f)

        print("Loading DistilBERT tokenizer...")
        bert_tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL)

        random.shuffle(data)
        split    = int(0.9 * len(data))
        train_ds = VerifierDataset(data[:split], bert_tokenizer)
        val_ds   = VerifierDataset(data[split:], bert_tokenizer)

        pin = DEVICE == "cuda"
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=pin)

        print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
        model = FaithfulnessVerifier().to(DEVICE)
        print(f"FaithfulnessVerifier (DistilBERT) -- {model.count_params():,} trainable parameters")
        train(model, train_loader, val_loader)

    elif args.mode == "test":
        if not os.path.exists(VERIFIER_PATH):
            print("No trained model found. Run: python verifier_gpu.py --mode train")
            return
        model, tokenizer = load_verifier(VERIFIER_PATH)

        test_cases = [
            {
                "context":  "Scott Derrickson is an American director, screenwriter and producer. Ed Wood was an American filmmaker, actor, and author.",
                "answer":   "Yes, both Scott Derrickson and Ed Wood are American.",
                "expected": "SUPPORTED"
            },
            {
                "context":  "Scott Derrickson is an American director, screenwriter and producer. Ed Wood was an American filmmaker, actor, and author.",
                "answer":   "Scott Derrickson is British and Ed Wood is French.",
                "expected": "UNSUPPORTED"
            },
            {
                "context":  "The Eiffel Tower is located in Paris. It was built in 1889.",
                "answer":   "Yes, both Scott Derrickson and Ed Wood are American.",
                "expected": "UNSUPPORTED"
            },
            {
                "context":  "Arthur's Magazine was an American literary periodical published in the 1800s.",
                "answer":   "Arthur's Magazine was published in the 1800s.",
                "expected": "SUPPORTED"
            },
        ]

        print(f"\n{'='*60}\nVerifier Test Cases\n{'='*60}")
        for i, tc in enumerate(test_cases, 1):
            result = verify(tc["context"], tc["answer"], model, tokenizer)
            status = "[CORRECT]" if result["label"] == tc["expected"] else "[WRONG]"
            print(f"\nTest {i}: {status}")
            print(f"  Context:  {tc['context'][:80]}...")
            print(f"  Answer:   {tc['answer']}")
            print(f"  Expected: {tc['expected']}")
            print(f"  Got:      {result['icon']} {result['label']} (confidence: {result['confidence']:.4f})")
            print(f"  Scores:   {result['scores']}")

    elif args.mode == "eval":
        if not os.path.exists(VERIFIER_PATH):
            print("No trained model found. Run: python verifier_gpu.py --mode train")
            return
        model, tokenizer = load_verifier(VERIFIER_PATH)
        evaluate_on_hotpotqa(model, tokenizer, num_samples=500)


if __name__ == "__main__":
    main()

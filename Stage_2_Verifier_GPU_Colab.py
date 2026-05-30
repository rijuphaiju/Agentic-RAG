"""
Stage 2: Faithfulness Verifier — Colab/CUDA Edition
=====================================================
Project: HARA — Hallucination-Aware Retrieval Agent

Paste this entire file into a Colab cell or upload as Stage_2_Verifier_GPU.py.
Set runtime to T4 GPU before running.

Run:
    !python Stage_2_Verifier_GPU.py --mode train

After training completes, download verifier_model.pt:
    from google.colab import files
    files.download("verifier_model.pt")
"""

# ── Install dependencies (run this cell first in Colab) ──────────────────────
# !pip install -q transformers datasets faiss-cpu sentence-transformers tqdm

import argparse
import json
import os
import pickle
import random
import re

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, DistilBertModel

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEQ_LEN         = 192    # full length — T4 handles it comfortably
NUM_CLASSES         = 3
BATCH_SIZE          = 64     # T4 16GB: safe at 64; increase to 96 if memory allows
EPOCHS              = 8
LR                  = 2e-5   # standard DistilBERT fine-tuning LR
TRAIN_SAMPLES       = 15000  # 15000 × ~13 examples = ~195k total
VERIFIER_PATH       = "verifier_model.pt"
DATA_PATH           = "verifier_data_bert.pkl"
LABEL_SMOOTHING     = 0.05
EARLY_STOP_PATIENCE = 3
DROPOUT             = 0.2

LABEL_MAP    = {0: "SUPPORTED", 1: "PARTIAL", 2: "UNSUPPORTED"}
LABEL_COLORS = {0: "[OK]", 1: "[!!]", 2: "[X]"}
BERT_MODEL   = "distilbert-base-uncased"

print(f"[Device] {DEVICE.upper()}"
      + (f" — {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else ""))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: BUILD TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────

def _get_other_entity(question: str, gold: str) -> str | None:
    """Extract the comparison counterpart entity from a HotpotQA comparison question.

    Example: question = "Which band was founded first, Hole or The Wolfhounds?"
             gold     = "The Wolfhounds"
             returns  -> "Hole"

    Used to generate UNSUPPORTED counterexamples for comparison questions so the
    verifier learns the reasoning boundary, not just surface lexical overlap.
    """
    gold_lower = gold.lower()
    skip = {
        'which', 'who', 'were', 'was', 'what', 'when', 'where', 'how', 'did',
        'the', 'both', 'and', 'or', 'of', 'a', 'an', 'is', 'are', 'be',
    }
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

    Label definitions:
    ------------------
    SUPPORTED   — context contains sufficient evidence to conclude the answer is
                  correct. Includes verbatim matches AND logically inferred
                  conclusions (comparison results, multi-hop chains).

    PARTIAL     — answer is partially grounded: context is incomplete (distractor
                  passages) OR answer makes extra claims beyond what context supports.

    UNSUPPORTED — answer contradicts context or has no grounding in it. Includes
                  the wrong comparison entity asserted confidently.

    Input format: "Q: {question} A: {answer}" vs context
    -----------------------------------------------------
    The question enables the verifier to understand what is being claimed, not
    just whether a string appears in context. This is critical for comparison
    questions: "The Wolfhounds were founded first" requires knowing what the
    question asks to validate it from two dates in context.
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
        qtype     = ex.get("type", "bridge")    # "bridge" or "comparison"
        titles    = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        sf_titles = set(ex["supporting_facts"]["title"])

        supporting, distractor = [], []
        for title, sents in zip(titles, sentences):
            block = " ".join(sents)
            (supporting if title in sf_titles else distractor).append(block)

        sup_ctx  = " ".join(supporting)[:800]
        dist_ctx = " ".join(distractor)[:800] if distractor else sup_ctx

        def qa(ans: str) -> str:
            return f"Q: {question} A: {ans}"

        # ── SUPPORTED ──────────────────────────────────────────────────────────
        # 1. Verbatim gold answer
        data.append({"context": sup_ctx, "answer": qa(gold), "label": 0})

        # 2. Sentence-form paraphrases of the gold answer.
        #    The LLM generates sentence forms ("The answer is X", "It is X",
        #    "Final Answer: X"). The old training data WRONGLY labelled these as
        #    PARTIAL against supporting context, teaching the verifier that
        #    "sentence form = PARTIAL". This is the root cause of PARTIAL bias.
        for p in [
            f"The answer is {gold}.",
            f"It is {gold}.",
            f"Final Answer: {gold}",
        ]:
            data.append({"context": sup_ctx, "answer": qa(p), "label": 0})

        if len(gold.split()) <= 4:
            data.append({"context": sup_ctx, "answer": qa(f"{gold} is the answer."), "label": 0})
            data.append({"context": sup_ctx, "answer": qa(f"That would be {gold}."), "label": 0})

        # 3. Comparison-specific SUPPORTED: inferred conclusions.
        #    "The Wolfhounds were founded first" is SUPPORTED when context shows
        #    Wolfhounds formed in 1985 and Hole formed in 1989.
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
        # 1. Random wrong answer (verbatim and sentence form)
        wrong = random.choice(all_answers)
        while wrong.lower() == gold.lower():
            wrong = random.choice(all_answers)
        data.append({"context": sup_ctx, "answer": qa(wrong), "label": 2})
        data.append({"context": sup_ctx, "answer": qa(f"The answer is {wrong}."), "label": 2})

        # 2. Comparison-specific UNSUPPORTED: wrong entity asserted confidently.
        #    "Hole was founded first" is UNSUPPORTED when Wolfhounds' date is earlier.
        #    This teaches the verifier the comparison reasoning boundary.
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
        # 1. Correct answer but incomplete/distractor context
        data.append({"context": dist_ctx, "answer": qa(gold), "label": 1})
        data.append({"context": dist_ctx, "answer": qa(f"The answer is {gold}."), "label": 1})

        # 2. Truncated answer (first word of multi-word gold)
        partial_ans = gold.split()[0]
        if partial_ans.lower() != gold.lower():
            data.append({"context": sup_ctx, "answer": qa(partial_ans), "label": 1})

        # 3. Correct answer + extra unverifiable claim (most common PARTIAL at inference)
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DATASET
# ─────────────────────────────────────────────────────────────────────────────
class VerifierDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=MAX_SEQ_LEN):
        print(f"  Pre-tokenizing {len(data):,} examples...")
        encoded = tokenizer(
            [d["answer"]  for d in data],
            [d["context"] for d in data],
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Keep on CPU — moved to GPU per batch in run_epoch (one transfer per step).
        # Do NOT move to GPU here: DataLoader accesses individual rows via __getitem__,
        # and per-row GPU indexing causes one sync per sample — killing throughput.
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: MODEL
# ─────────────────────────────────────────────────────────────────────────────
class FaithfulnessVerifier(nn.Module):
    """DistilBERT encoder + 2-layer classification head.
    Input: [CLS] Q:{question} A:{answer} [SEP] {context} [SEP]
    Output: logits over {SUPPORTED, PARTIAL, UNSUPPORTED}
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.bert       = DistilBertModel.from_pretrained(BERT_MODEL)
        hidden_dim      = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] representation
        return self.classifier(cls)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: METRICS
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, scheduler=None,
              scaler=None, desc=""):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss   = torch.tensor(0.0, device=DEVICE)
    pred_chunks  = []
    label_chunks = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            if is_train and scaler is not None:
                # CUDA + AMP path
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
            else:
                logits = model(input_ids, attn_mask)
                loss   = criterion(logits, labels)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()

            total_loss  += loss.detach()
            pred_chunks.append(logits.detach().argmax(dim=-1))
            label_chunks.append(labels)

    avg_loss       = (total_loss / len(loader)).item()
    all_preds      = torch.cat(pred_chunks).cpu().tolist()
    all_labels     = torch.cat(label_chunks).cpu().tolist()
    acc            = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    per_f1, macro  = compute_macro_f1(all_labels, all_preds)
    return avg_loss, acc, macro, per_f1, all_preds, all_labels


def train(model, train_loader, val_loader):
    # Class weights: PARTIAL highest — hardest class and most consequential
    # (false UNSUPPORTED on a correct PARTIAL causes unnecessary abstention)
    weights   = torch.tensor([1.0, 1.8, 1.4]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    # Differential LR: BERT body lower, classifier head higher
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(),       "lr": LR},
        {"params": model.classifier.parameters(), "lr": LR * 10},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = total_steps // 10

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler("cuda") if DEVICE == "cuda" else None

    best_macro_f1  = 0.0
    patience_count = 0
    history        = []

    print(f"\n{'='*60}")
    print(f"Fine-tuning DistilBERT Faithfulness Verifier on {DEVICE.upper()}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LR} | AMP: {scaler is not None}")
    print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc, t_f1, _, _, _       = run_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            desc=f"Epoch {epoch}/{EPOCHS} [Train]"
        )
        v_loss, v_acc, v_f1, per_f1, _, _  = run_epoch(
            model, val_loader, criterion,
            desc=f"Epoch {epoch}/{EPOCHS} [Val]  "
        )

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Train -> Loss: {t_loss:.4f}  Acc: {t_acc:.4f}  Macro F1: {t_f1:.4f}")
        print(f"  Val   -> Loss: {v_loss:.4f}  Acc: {v_acc:.4f}  Macro F1: {v_f1:.4f}")
        print(f"  Per-class F1 -> SUPPORTED: {per_f1[0]:.4f} | "
              f"PARTIAL: {per_f1[1]:.4f} | UNSUPPORTED: {per_f1[2]:.4f}")

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
                "config":      {"bert_model": BERT_MODEL, "num_classes": NUM_CLASSES},
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
    print(f"\nTraining complete. Best Macro F1: {best_macro_f1:.4f}")
    print(f"Model saved -> {VERIFIER_PATH}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: INFERENCE API  (used by all pipeline stages at inference time)
# ─────────────────────────────────────────────────────────────────────────────
def load_verifier(model_path=VERIFIER_PATH):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model      = FaithfulnessVerifier().to(DEVICE)
    state_dict = checkpoint["model_state"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL)
    print(f"Verifier loaded from {model_path} "
          f"(epoch {checkpoint['epoch']}, Macro F1: {checkpoint['macro_f1']:.4f})")
    return model, tokenizer


def build_verify_context(passages: list, answer: str, top_n: int = 5) -> str:
    """Build verification context prioritising passages that contain the answer.

    Without this, the answer-supporting passage is often ranked 6th+ by the
    CrossEncoder (which ranks by query similarity, not answer presence) and gets
    truncated out of the verifier's MAX_SEQ_LEN window, causing PARTIAL on
    correct answers.
    """
    a_lower    = answer.lower().strip()
    containing = [p for p in passages if a_lower in p["text"].lower()]
    others     = [p for p in passages if a_lower not in p["text"].lower()]
    ordered    = (containing + others)[:top_n]
    return " ".join(p["text"] for p in ordered)


def _distill_for_verify(answer: str, max_words: int = 12) -> str:
    """Shorten verbose LLM answer to a compact phrase for the verifier.

    The verifier input is "Q: {question} A: {distilled_answer}" — distilling
    keeps the answer within the training distribution and ensures the question
    gets enough token budget in MAX_SEQ_LEN=192.
    """
    m = re.search(r'(?:Final Answer|Answer)\s*:\s*(.+?)(?:\n|$)', answer, re.IGNORECASE)
    if m:
        answer = m.group(1).strip()
    first = re.split(r'(?<=[.!?])\s', answer.strip(), maxsplit=1)[0]
    words = first.split()
    return ' '.join(words[:max_words]) if len(words) > max_words else first.strip()


def verify(context, answer, model, tokenizer, question=None):
    """Single-sample faithfulness verification.

    Args:
        context:  Retrieved passage text.
        answer:   LLM-generated answer (distilled internally).
        question: Original user question. Required for comparison/multi-hop
                  entailment — without it the verifier reverts to lexical overlap.
    """
    answer_for_verify = _distill_for_verify(answer)
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
    """Batched inference for evaluation."""
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
        with torch.no_grad():
            logits = model(encoded["input_ids"].to(DEVICE),
                           encoded["attention_mask"].to(DEVICE))
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_on_hotpotqa(model, tokenizer, num_samples=500):
    print(f"\nEvaluating verifier on {num_samples} HotpotQA validation examples...")
    dataset     = load_dataset("hotpot_qa", "distractor", split="validation")
    examples    = list(dataset)
    all_answers = [ex["answer"] for ex in examples]

    eval_data = []
    for ex in examples[:num_samples // 3]:
        gold      = ex["answer"]
        question  = ex["question"]
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

        def qa(ans):
            return f"Q: {question} A: {ans}"

        eval_data.append({"context": sup_ctx,  "answer": qa(gold),  "label": 0})
        eval_data.append({"context": sup_ctx,  "answer": qa(wrong), "label": 2})
        eval_data.append({"context": dist_ctx, "answer": qa(gold),  "label": 1})

    contexts = [d["context"] for d in eval_data]
    answers  = [d["answer"]  for d in eval_data]
    labels   = [d["label"]   for d in eval_data]

    results  = verify_batch(contexts, answers, model, tokenizer, batch_size=BATCH_SIZE)
    preds    = [list(LABEL_MAP.values()).index(r["label"]) for r in results]

    per_f1, macro_f1 = compute_macro_f1(labels, preds)
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)

    print(f"\n{'='*60}")
    print(f"Verifier Evaluation — {len(eval_data)} samples")
    print(f"{'='*60}")
    print(f"Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    print(f"Macro F1  : {macro_f1:.4f}  (target ≥ 0.70)")
    print(f"  SUPPORTED   F1: {per_f1[0]:.4f}")
    print(f"  PARTIAL     F1: {per_f1[1]:.4f}")
    print(f"  UNSUPPORTED F1: {per_f1[2]:.4f}")

    with open("verifier_eval_results.json", "w") as f:
        json.dump({
            "accuracy": acc, "macro_f1": macro_f1,
            "per_class_f1": {"SUPPORTED": per_f1[0],
                             "PARTIAL":   per_f1[1],
                             "UNSUPPORTED": per_f1[2]}
        }, f, indent=2)
    return macro_f1


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "eval"], default="train")
    args = parser.parse_args()

    print(f"Device: {DEVICE.upper()}")

    if args.mode == "train":
        if os.path.exists(DATA_PATH):
            print("Loading cached training data...")
            with open(DATA_PATH, "rb") as f:
                data = pickle.load(f)
        else:
            data = build_training_data(TRAIN_SAMPLES)
            with open(DATA_PATH, "wb") as f:
                pickle.dump(data, f)

        tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL)
        random.shuffle(data)
        split      = int(0.9 * len(data))
        train_ds   = VerifierDataset(data[:split],  tokenizer)
        val_ds     = VerifierDataset(data[split:],  tokenizer)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=2, pin_memory=True)

        print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
        model = FaithfulnessVerifier().to(DEVICE)
        print(f"FaithfulnessVerifier (DistilBERT) — {model.count_params():,} trainable parameters")
        train(model, train_loader, val_loader)

        # Auto-download in Colab
        try:
            from google.colab import files
            print("\nDownloading verifier_model.pt...")
            files.download(VERIFIER_PATH)
        except ImportError:
            print(f"\nNot in Colab — model saved to: {VERIFIER_PATH}")

    elif args.mode == "eval":
        model, tokenizer = load_verifier(VERIFIER_PATH)
        evaluate_on_hotpotqa(model, tokenizer, num_samples=500)

    elif args.mode == "test":
        model, tokenizer = load_verifier(VERIFIER_PATH)
        test_cases = [
            {
                "question": "Which band was founded first, Hole or The Wolfhounds?",
                "context":  "Hole is an American rock band formed in Los Angeles in 1989. The Wolfhounds are a British indie band formed in 1985.",
                "answer":   "The Wolfhounds were founded first.",
                "expected": "SUPPORTED",
            },
            {
                "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
                "context":  "Scott Derrickson is an American director. Ed Wood was an American filmmaker.",
                "answer":   "Yes, both are American.",
                "expected": "SUPPORTED",
            },
            {
                "question": "Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?",
                "context":  "Henri Leconte never won a Grand Slam singles title. Jonathan Stark won two Grand Slam doubles titles.",
                "answer":   "Jonathan Stark won more Grand Slam titles.",
                "expected": "SUPPORTED",
            },
            {
                "question": "Where is Goodison Park located?",
                "context":  "Goodison Park is a football stadium located in Walton, Liverpool, England.",
                "answer":   "England",
                "expected": "SUPPORTED",
            },
            {
                "question": "Which band was founded first, Hole or The Wolfhounds?",
                "context":  "Hole is an American rock band formed in Los Angeles in 1989. The Wolfhounds are a British indie band formed in 1985.",
                "answer":   "Hole was founded first.",
                "expected": "UNSUPPORTED",
            },
        ]

        print(f"\n{'='*60}\nVerifier Test Cases\n{'='*60}")
        correct = 0
        for i, tc in enumerate(test_cases, 1):
            result = verify(tc["context"], tc["answer"], model, tokenizer,
                            question=tc["question"])
            ok = result["label"] == tc["expected"]
            correct += ok
            status = "[PASS]" if ok else "[FAIL]"
            print(f"\nTest {i}: {status}")
            print(f"  Q:        {tc['question']}")
            print(f"  Answer:   {tc['answer']}")
            print(f"  Expected: {tc['expected']}  Got: {result['label']} "
                  f"(support={result['scores']['SUPPORTED']:.3f})")
        print(f"\nResult: {correct}/{len(test_cases)} passed")


if __name__ == "__main__":
    main()

"""
Stage 2: Faithfulness Verifier -- DistilBERT Edition (GPU Accelerated)
=======================================================================
Project: Reducing Hallucinations in Agentic RAG Systems

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
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN    = 256
NUM_CLASSES    = 3
BATCH_SIZE     = 32 if DEVICE == "cuda" else 8   # DistilBERT needs more VRAM than custom model
EPOCHS         = 8
LR             = 2e-5   # standard fine-tuning LR for BERT-family models
TRAIN_SAMPLES  = 30000
VERIFIER_PATH  = "verifier_model.pt"
DATA_PATH      = "verifier_data_bert.pkl"        # separate file -- keeps v1 data intact
LABEL_SMOOTHING = 0.05                           # lower smoothing for pretrained model
EARLY_STOP_PATIENCE = 3
DROPOUT        = 0.2

LABEL_MAP    = {0: "SUPPORTED", 1: "PARTIAL", 2: "UNSUPPORTED"}
LABEL_COLORS = {0: "[OK]", 1: "[!!]", 2: "[X]"}
BERT_MODEL   = "distilbert-base-uncased"

print(f"[Device] Using: {DEVICE.upper()}"
      + (f" -- {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else " (CPU fallback)"))


# 
# STEP 1: BUILD TRAINING DATA
# 
def build_training_data(num_samples=TRAIN_SAMPLES):
    print("Loading HotpotQA training split...")
    dataset     = load_dataset("hotpot_qa", "distractor", split="train")
    examples    = list(dataset)
    all_answers = [ex["answer"] for ex in examples]

    data = []
    print(f"Building verifier training data from {num_samples} examples...")

    for ex in tqdm(examples[:num_samples]):
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

        # SUPPORTED: gold answer in supporting context
        data.append({"context": sup_ctx, "answer": gold, "label": 0})

        # UNSUPPORTED: random wrong answer in supporting context
        wrong = random.choice(all_answers)
        while wrong.lower() == gold.lower():
            wrong = random.choice(all_answers)
        data.append({"context": sup_ctx, "answer": wrong, "label": 2})

        # PARTIAL: gold answer but only distractor/partial context
        data.append({"context": dist_ctx, "answer": gold, "label": 1})

        # Extra PARTIAL: incomplete answer in supporting context
        if len(gold.split()) > 1:
            partial_ans = gold.split()[0]
            data.append({"context": sup_ctx, "answer": partial_ans, "label": 1})

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
        self.data      = data
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item    = self.data[idx]
        encoded = self.tokenizer(
            item["answer"],          # sentence A
            item["context"],         # sentence B (context is longer, goes second)
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label":          torch.tensor(item["label"], dtype=torch.long),
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

    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            if is_train and scaler is not None:
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

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss          = total_loss / len(loader)
    acc               = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    per_f1, macro_f1  = compute_macro_f1(all_labels, all_preds)
    return avg_loss, acc, macro_f1, per_f1, all_preds, all_labels


def train(model, train_loader, val_loader):
    weights   = torch.tensor([1.0, 1.2, 1.5]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    # Separate LR for BERT body vs classifier head -- standard fine-tuning trick
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(),       "lr": LR},
        {"params": model.classifier.parameters(), "lr": LR * 10},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = total_steps // 10
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler         = GradScaler("cuda") if DEVICE == "cuda" else None
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
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model      = FaithfulnessVerifier().to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    bert_tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL)
    print(f"Verifier loaded from {model_path} "
          f"(epoch {checkpoint['epoch']}, Macro F1: {checkpoint['macro_f1']:.4f})")
    return model, bert_tokenizer


def verify(context, answer, model, tokenizer):
    """Single-sample inference."""
    model.eval()
    encoded   = tokenizer(
        answer, context,
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

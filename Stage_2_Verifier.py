"""
Stage 2: Faithfulness Verifier — Transformer Trained from Scratch
=================================================================
Project: Reducing Hallucinations in Agentic RAG Systems
Proposal Section: 2.2.5, 2.4, 2.8

Architecture:
  - Encoder-only transformer (~10-15M parameters)
  - Input: [CLS] context [SEP] answer [SEP]
  - Output: SUPPORTED (0) / PARTIAL (1) / UNSUPPORTED (2)

Usage:
  python verifier.py --mode train    # train the verifier
  python verifier.py --mode test     # test on a few examples
  python verifier.py --mode eval     # evaluate on HotpotQA validation set
"""

import argparse
import json
import math
import os
import pickle
import random
import re
import string

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE     = 20000       # top 10k words — enough for this task
MAX_SEQ_LEN    = 256         # max tokens: [CLS] context [SEP] answer [SEP]
HIDDEN_DIM     = 384         # transformer hidden size
FF_DIM         = 768         # feed-forward inner dimension
NUM_HEADS      = 4           # attention heads (head_dim = 256/4 = 64)
NUM_LAYERS     = 4           # encoder layers
DROPOUT        = 0.1
NUM_CLASSES    = 3           # SUPPORTED=0, PARTIAL=1, UNSUPPORTED=2
BATCH_SIZE     = 32
EPOCHS         = 10
LR             = 3e-4
TRAIN_SAMPLES  = 30000       # HotpotQA examples to use for training
VERIFIER_PATH  = "verifier_model.pt"
TOKENIZER_PATH = "tokenizer.pkl"
DATA_PATH      = "verifier_data.pkl"

LABEL_MAP      = {0: "SUPPORTED", 1: "PARTIAL", 2: "UNSUPPORTED"}
LABEL_COLORS   = {0: "✅", 1: "⚠️ ", 2: "❌"}


# ─────────────────────────────────────────────
# STEP 1: SIMPLE WORD-LEVEL TOKENIZER
# ─────────────────────────────────────────────
class SimpleTokenizer:
    """
    Word-level tokenizer with fixed vocabulary.
    Encodes (context, answer) pairs as:
        [CLS] context_tokens [SEP] answer_tokens [SEP]
    — matching Equation 2.11 in the proposal.
    """
    PAD = 0
    UNK = 1
    CLS = 2
    SEP = 3

    def __init__(self):
        self.word2id = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.built   = False

    def _clean(self, text):
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        return text.split()

    def build_vocab(self, texts, max_size=VOCAB_SIZE):
        from collections import Counter
        counter = Counter()
        for text in tqdm(texts, desc="Building vocabulary"):
            counter.update(self._clean(text))
        for word, _ in counter.most_common(max_size - 4):
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)
        self.built = True
        print(f"Vocabulary built: {len(self.word2id)} tokens")

    def encode(self, context, answer, max_len=MAX_SEQ_LEN):
        """
        Returns (input_ids, segment_ids, attention_mask) — all length max_len.
        segment_ids: 0 = context side, 1 = answer side.
        """
        ctx = [self.word2id.get(w, self.UNK) for w in self._clean(context)]
        ans = [self.word2id.get(w, self.UNK) for w in self._clean(answer)]

        # Truncate context so everything fits
        max_ctx = max_len - len(ans) - 3   # 3 special tokens
        ctx = ctx[:max_ctx]
        ans = ans[:(max_len - 3)]          # safety truncate answer too

        ids  = [self.CLS] + ctx + [self.SEP] + ans + [self.SEP]
        seg  = [0] * (len(ctx) + 2) + [1] * (len(ans) + 1)

        # Pad to max_len
        pad  = max_len - len(ids)
        ids  += [self.PAD] * pad
        seg  += [self.PAD] * pad
        mask  = [1] * (max_len - pad) + [0] * pad

        return ids[:max_len], seg[:max_len], mask[:max_len]

    def save(self, path=TOKENIZER_PATH):
        with open(path, "wb") as f:
            pickle.dump(self.word2id, f)
        print(f"Tokenizer saved → {path}")

    def load(self, path=TOKENIZER_PATH):
        with open(path, "rb") as f:
            self.word2id = pickle.load(f)
        self.built = True
        print(f"Tokenizer loaded: {len(self.word2id)} tokens")


# ─────────────────────────────────────────────
# STEP 2: BUILD TRAINING DATA FROM HOTPOTQA
# ─────────────────────────────────────────────
def build_training_data(num_samples=TRAIN_SAMPLES):
    """
    Automatically creates labelled (context, answer, label) triples
    from HotpotQA's supporting_facts annotations.

    Labels:
      SUPPORTED   (0) → supporting sentences  + correct answer
      UNSUPPORTED (2) → supporting sentences  + wrong answer
      PARTIAL     (1) → distractor sentences  + correct answer

    No human annotation needed — HotpotQA's labels do it for free.
    This directly implements the dataset construction described in
    proposal Section 6.3.3.
    """
    print("Loading HotpotQA training split...")
    dataset  = load_dataset("hotpot_qa", "distractor", split="train")
    examples = list(dataset)
    all_answers = [ex["answer"] for ex in examples]

    data = []
    print(f"Building verifier training data from {num_samples} examples...")

    for ex in tqdm(examples[:num_samples]):
        gold        = ex["answer"]
        titles      = ex["context"]["title"]
        sentences   = ex["context"]["sentences"]
        sf_titles   = set(ex["supporting_facts"]["title"])

        # Separate supporting vs distractor passages
        supporting, distractor = [], []
        for title, sents in zip(titles, sentences):
            block = " ".join(sents)
            if title in sf_titles:
                supporting.append(block)
            else:
                distractor.append(block)

        sup_ctx  = " ".join(supporting)[:800]
        dist_ctx = " ".join(distractor)[:800] if distractor else sup_ctx

        # ── SUPPORTED: correct answer in supporting context ──
        data.append({"context": sup_ctx,  "answer": gold,  "label": 0})

        # ── UNSUPPORTED: wrong answer in supporting context ──
        wrong = random.choice(all_answers)
        while wrong.lower() == gold.lower():
            wrong = random.choice(all_answers)
        data.append({"context": sup_ctx,  "answer": wrong, "label": 2})

        # ── PARTIAL: correct answer but only distractor context ──
        data.append({"context": dist_ctx, "answer": gold,  "label": 1})

    random.shuffle(data)

    # Print label distribution
    counts = {0: 0, 1: 0, 2: 0}
    for d in data:
        counts[d["label"]] += 1
    print(f"Dataset built: {len(data)} examples")
    print(f"  SUPPORTED={counts[0]} | PARTIAL={counts[1]} | UNSUPPORTED={counts[2]}")
    return data


# ─────────────────────────────────────────────
# STEP 3: PYTORCH DATASET
# ─────────────────────────────────────────────
class VerifierDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data      = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ids, seg, mask = self.tokenizer.encode(item["context"], item["answer"])
        return {
            "input_ids":      torch.tensor(ids,           dtype=torch.long),
            "segment_ids":    torch.tensor(seg,           dtype=torch.long),
            "attention_mask": torch.tensor(mask,          dtype=torch.long),
            "label":          torch.tensor(item["label"], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# STEP 4: TRANSFORMER ENCODER FROM SCRATCH
# Implements Equations 2.1–2.6 from the proposal
# ─────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head attention.
    Equations 2.1–2.4 in the proposal.
    """
    def __init__(self, hidden_dim, num_heads, dropout=DROPOUT):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        # Project and split into heads
        Q = self.W_Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (Equation 2.2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply padding mask
        if mask is not None:
            scores = scores.masked_fill(
                mask[:, None, None, :] == 0, float("-inf")
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, V)                              # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)    # (B, T, D)
        return self.W_O(out)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with GELU activation.
    Equation 2.5 in the proposal.
    """
    def __init__(self, hidden_dim, ff_dim, dropout=DROPOUT):
        super().__init__()
        self.fc1     = nn.Linear(hidden_dim, ff_dim)
        self.act     = nn.GELU()
        self.drop    = nn.Dropout(dropout)
        self.fc2     = nn.Linear(ff_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer.
    Uses Pre-LayerNorm architecture (more stable training).
    Equation 2.6 in the proposal.
    """
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=DROPOUT):
        super().__init__()
        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.attn    = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ff      = FeedForward(hidden_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm + residual (more stable than post-norm)
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class FaithfulnessVerifier(nn.Module):
    """
    Encoder-only transformer faithfulness verifier trained from scratch.
    ~10-15M parameters as specified in proposal Section 2.2.5.

    Input format (Equation 2.11):
        [CLS] context_tokens [SEP] answer_tokens [SEP]

    Output:
        Logits over {SUPPORTED=0, PARTIAL=1, UNSUPPORTED=2}
        via the [CLS] token representation.
    """
    def __init__(
        self,
        vocab_size  = VOCAB_SIZE,
        hidden_dim  = HIDDEN_DIM,
        ff_dim      = FF_DIM,
        num_heads   = NUM_HEADS,
        num_layers  = NUM_LAYERS,
        max_seq_len = MAX_SEQ_LEN,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT,
    ):
        super().__init__()

        # ── Embeddings ──
        self.token_emb   = nn.Embedding(vocab_size,  hidden_dim, padding_idx=0)
        self.segment_emb = nn.Embedding(2,           hidden_dim)  # 0=context, 1=answer
        self.pos_emb     = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_norm    = nn.LayerNorm(hidden_dim)
        self.emb_drop    = nn.Dropout(dropout)

        # ── Encoder stack (4 layers, 4 heads, dim=256) ──
        self.encoder = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        # ── Classification head using [CLS] token ──
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()
        print(f"FaithfulnessVerifier ready — {self.count_params():,} trainable parameters")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, segment_ids, attention_mask):
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # Combine token + segment + positional embeddings
        x = self.token_emb(input_ids) + self.segment_emb(segment_ids) + self.pos_emb(pos)
        x = self.emb_norm(self.emb_drop(x))

        # Pass through encoder layers
        for layer in self.encoder:
            x = layer(x, attention_mask)

        x = self.final_norm(x)

        # Use [CLS] token (position 0) for classification
        cls_repr = x[:, 0, :]
        return self.classifier(cls_repr)


# ─────────────────────────────────────────────
# STEP 5: METRICS
# Equations 6.3–6.6 from the proposal
# ─────────────────────────────────────────────
def compute_macro_f1(labels, preds, num_classes=NUM_CLASSES):
    """
    Per-class Precision, Recall, F1 → unweighted Macro F1.
    Equations 6.3–6.6 in the proposal.
    """
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


# ─────────────────────────────────────────────
# STEP 6: TRAINING LOOP
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, scheduler=None, desc=""):
    """Single training or evaluation epoch."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            seg_ids   = batch["segment_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            logits = model(input_ids, seg_ids, attn_mask)
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

    avg_loss   = total_loss / len(loader)
    acc        = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    per_f1, macro_f1 = compute_macro_f1(all_labels, all_preds)
    return avg_loss, acc, macro_f1, per_f1, all_preds, all_labels


def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()   # Cross-Entropy Loss — Equation 2.10
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS, pct_start=0.1
    )

    best_macro_f1 = 0.0
    history = []

    print(f"\n{'='*60}")
    print(f"Training Faithfulness Verifier on {DEVICE.upper()}")
    print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc, t_f1, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            desc=f"Epoch {epoch}/{EPOCHS} [Train]"
        )
        v_loss, v_acc, v_f1, per_f1, _, _ = run_epoch(
            model, val_loader, criterion,
            desc=f"Epoch {epoch}/{EPOCHS} [Val]  "
        )

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Train → Loss: {t_loss:.4f}  Acc: {t_acc:.4f}  Macro F1: {t_f1:.4f}")
        print(f"  Val   → Loss: {v_loss:.4f}  Acc: {v_acc:.4f}  Macro F1: {v_f1:.4f}")
        print(f"  Per-class F1 → SUPPORTED: {per_f1[0]:.4f} | PARTIAL: {per_f1[1]:.4f} | UNSUPPORTED: {per_f1[2]:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": t_loss, "train_acc": t_acc, "train_macro_f1": t_f1,
            "val_loss":   v_loss, "val_acc":   v_acc, "val_macro_f1":   v_f1,
            "per_class_f1": {"SUPPORTED": per_f1[0], "PARTIAL": per_f1[1], "UNSUPPORTED": per_f1[2]}
        })

        if v_f1 > best_macro_f1:
            best_macro_f1 = v_f1
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "macro_f1": v_f1,
                "config": {
                    "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
                    "ff_dim": FF_DIM, "num_heads": NUM_HEADS,
                    "num_layers": NUM_LAYERS, "max_seq_len": MAX_SEQ_LEN,
                }
            }, VERIFIER_PATH)
            print(f"  ✅ Best model saved (Macro F1: {v_f1:.4f})\n")
        else:
            print()

    with open("verifier_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best Macro F1: {best_macro_f1:.4f}")
    print(f"Model saved → {VERIFIER_PATH}")
    return history


# ─────────────────────────────────────────────
# STEP 7: INFERENCE API
# ─────────────────────────────────────────────
def load_verifier(model_path=VERIFIER_PATH):
    """Load trained verifier from disk."""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    cfg = checkpoint["config"]
    model = FaithfulnessVerifier(
        vocab_size  = cfg["vocab_size"],
        hidden_dim  = cfg["hidden_dim"],
        ff_dim      = cfg["ff_dim"],
        num_heads   = cfg["num_heads"],
        num_layers  = cfg["num_layers"],
        max_seq_len = cfg["max_seq_len"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Verifier loaded from {model_path} (trained epoch {checkpoint['epoch']}, Macro F1: {checkpoint['macro_f1']:.4f})")
    return model


def verify(context, answer, model, tokenizer):
    """
    Verify a single (context, answer) pair.
    Returns label + confidence scores.

    This is the function called by the Agentic Loop in Stage 4.
    """
    model.eval()
    ids, seg, mask = tokenizer.encode(context, answer)

    input_ids = torch.tensor([ids],  dtype=torch.long).to(DEVICE)
    seg_ids   = torch.tensor([seg],  dtype=torch.long).to(DEVICE)
    attn_mask = torch.tensor([mask], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, seg_ids, attn_mask)
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


# ─────────────────────────────────────────────
# STEP 8: EVALUATE ON HOTPOTQA VALIDATION SET
# ─────────────────────────────────────────────
def evaluate_on_hotpotqa(model, tokenizer, num_samples=500):
    """
    Evaluate verifier on HotpotQA validation set.
    Builds ground truth labels same way as training data.
    Reports per-class F1 and Macro F1.
    """
    print(f"\nEvaluating verifier on {num_samples} HotpotQA validation examples...")
    dataset  = load_dataset("hotpot_qa", "distractor", split="validation")
    examples = list(dataset)
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

    all_preds, all_labels = [], []
    for item in tqdm(eval_data, desc="Evaluating"):
        result = verify(item["context"], item["answer"], model, tokenizer)
        pred   = list(LABEL_MAP.values()).index(result["label"])
        all_preds.append(pred)
        all_labels.append(item["label"])

    per_f1, macro_f1 = compute_macro_f1(all_labels, all_preds)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    print(f"\n{'='*60}")
    print(f"Verifier Evaluation Results ({len(eval_data)} samples)")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"Macro F1:  {macro_f1:.4f}  (target: ≥ 0.70)")
    print(f"  SUPPORTED   F1: {per_f1[0]:.4f}")
    print(f"  PARTIAL     F1: {per_f1[1]:.4f}")
    print(f"  UNSUPPORTED F1: {per_f1[2]:.4f}")

    # Save results
    with open("verifier_eval_results.json", "w") as f:
        json.dump({
            "accuracy": acc, "macro_f1": macro_f1,
            "per_class_f1": {
                "SUPPORTED": per_f1[0],
                "PARTIAL": per_f1[1],
                "UNSUPPORTED": per_f1[2]
            }
        }, f, indent=2)
    print("Results saved → verifier_eval_results.json")
    return macro_f1


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "eval"], default="train")
    args = parser.parse_args()

    print(f"Device: {DEVICE.upper()}")

    # ── TRAIN MODE ──
    if args.mode == "train":

        # 1. Build or load training data
        if os.path.exists(DATA_PATH):
            print("Loading existing training data...")
            with open(DATA_PATH, "rb") as f:
                data = pickle.load(f)
        else:
            data = build_training_data(TRAIN_SAMPLES)
            with open(DATA_PATH, "wb") as f:
                pickle.dump(data, f)

        # 2. Build or load tokenizer
        tokenizer = SimpleTokenizer()
        if os.path.exists(TOKENIZER_PATH):
            tokenizer.load(TOKENIZER_PATH)
        else:
            all_texts = [d["context"] + " " + d["answer"] for d in data]
            tokenizer.build_vocab(all_texts, max_size=VOCAB_SIZE)
            tokenizer.save(TOKENIZER_PATH)

        # 3. Train / val split (90/10)
        random.shuffle(data)
        split      = int(0.9 * len(data))
        train_ds   = VerifierDataset(data[:split], tokenizer)
        val_ds     = VerifierDataset(data[split:], tokenizer)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

        # 4. Build and train model
        model = FaithfulnessVerifier(vocab_size=len(tokenizer.word2id)).to(DEVICE)
        train(model, train_loader, val_loader)

    # ── TEST MODE (quick demo) ──
    elif args.mode == "test":
        if not os.path.exists(VERIFIER_PATH):
            print("No trained model found. Run: python verifier.py --mode train")
            return

        tokenizer = SimpleTokenizer()
        tokenizer.load(TOKENIZER_PATH)
        model = load_verifier(VERIFIER_PATH).to(DEVICE)

        test_cases = [
            {
                "context": "Scott Derrickson is an American director, screenwriter and producer. Ed Wood was an American filmmaker, actor, and author.",
                "answer":  "Yes, both Scott Derrickson and Ed Wood are American.",
                "expected": "SUPPORTED"
            },
            {
                "context": "Scott Derrickson is an American director, screenwriter and producer. Ed Wood was an American filmmaker, actor, and author.",
                "answer":  "Scott Derrickson is British and Ed Wood is French.",
                "expected": "UNSUPPORTED"
            },
            {
                "context": "The Eiffel Tower is located in Paris. It was built in 1889.",
                "answer":  "Yes, both Scott Derrickson and Ed Wood are American.",
                "expected": "UNSUPPORTED"
            },
            {
                "context": "Arthur's Magazine was an American literary periodical published in the 1800s.",
                "answer":  "Arthur's Magazine was published in the 1800s.",
                "expected": "SUPPORTED"
            },
        ]

        print(f"\n{'='*60}")
        print("Verifier Test Cases")
        print(f"{'='*60}")
        for i, tc in enumerate(test_cases, 1):
            result = verify(tc["context"], tc["answer"], model, tokenizer)
            status = "✅ CORRECT" if result["label"] == tc["expected"] else "❌ WRONG"
            print(f"\nTest {i}: {status}")
            print(f"  Context:  {tc['context'][:80]}...")
            print(f"  Answer:   {tc['answer']}")
            print(f"  Expected: {tc['expected']}")
            print(f"  Got:      {result['icon']} {result['label']} (confidence: {result['confidence']:.4f})")
            print(f"  Scores:   {result['scores']}")

    # ── EVAL MODE ──
    elif args.mode == "eval":
        if not os.path.exists(VERIFIER_PATH):
            print("No trained model found. Run: python verifier.py --mode train")
            return

        tokenizer = SimpleTokenizer()
        tokenizer.load(TOKENIZER_PATH)
        model = load_verifier(VERIFIER_PATH).to(DEVICE)
        evaluate_on_hotpotqa(model, tokenizer, num_samples=500)


if __name__ == "__main__":
    main()

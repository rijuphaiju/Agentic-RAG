"""
Stage 2: Fine-Tunes MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli
====================================================================
Project: HARA — Hallucination-Aware Retrieval Agent

Fine-tunes the pretrained NLI checkpoint's existing 3-way entailment/
neutral/contradiction head on this project's own (evidence, claim, label)
triples, derived from verifier_dataset_labeled.jsonl (produced by
Stage_2_Dataset.py build + label). The checkpoint's head is NOT
reinitialized — SUPPORTED/PARTIAL/UNSUPPORTED map onto the model's own
existing entailment/neutral/contradiction label ids, so fine-tuning nudges
the general-purpose NLI knowledge rather than replacing it.

Premise construction reuses Stage_2_Verifier.build_premise() — the exact
same function verify() calls at inference time — so the model is trained
on the same kind of (premise, hypothesis) pair it will actually see in
production, not a differently-shaped one.

Usage:
    python Stage_2_Verifier_Train.py --data-path verifier_dataset_labeled.jsonl \\
        --output-dir models/hara_deberta_v3_large_verifier --epochs 3

Saves a standard HF checkpoint directory (config.json, model weights,
tokenizer files) via save_pretrained() — Stage_2_Verifier.resolve_model_path()
picks it up automatically on the next run once it exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Stage_2_Verifier pulls in Stage_1_RAG_Pipeline, which imports faiss before
# sentence-transformers/torch — faiss must load first in-process or torch's
# multi-threaded CPU backend segfaults against faiss's OpenMP pool (see
# Stage_2_Verifier.py's import-order comment). Keep this import first.
from Stage_2_Verifier import (
    FINE_TUNED_MODEL_DIR,
    MAX_LENGTH,
    ZERO_SHOT_NLI_MODEL,
    build_premise,
    extract_claims,
    extract_entities,
    is_comparable_entity,
)

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

# Full fp32 fine-tuning of this 435M-param checkpoint needs params + grads +
# AdamW's fp32 momentum/variance ≈ 7GB minimum — more than a 6GB card has.
# Confirmed empirically: without LoRA, PyTorch's CUDA allocator over-
# subscribes into Windows' driver-level "shared GPU memory" (system RAM
# paging), and each training step goes from an expected ~1s to 65-77s (a
# ~70x slowdown), making full fine-tuning impractical on this GPU regardless
# of time budget. LoRA freezes the base model and trains small adapter
# matrices instead, cutting optimizer-state memory from gigabytes to
# megabytes. The adapter is merged back into the base weights before saving
# (see main(), model.merge_and_unload()), so the checkpoint on disk is a
# plain HF model directory — Stage_2_Verifier.py's inference path needs no
# changes and no peft dependency at inference time.
LORA_TARGET_MODULES = ["query_proj", "key_proj", "value_proj"]
LORA_MODULES_TO_SAVE = ["classifier"]

logger = logging.getLogger("stage2_verifier_train")

DEFAULT_DATA_PATH = "verifier_dataset_labeled.jsonl"
DEFAULT_PREMISE_CACHE = "verifier_train_premises_cache.jsonl"

_HARA_TO_NLI_LABEL = {
    "SUPPORTED": "entailment",
    "PARTIAL": "neutral",
    "UNSUPPORTED": "contradiction",
}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


def resolve_label2id(model) -> Dict[str, int]:
    """Maps SUPPORTED/PARTIAL/UNSUPPORTED to the checkpoint's OWN
    entailment/neutral/contradiction label ids (not a fixed assumption),
    so fine-tuning writes into the head the checkpoint already has."""
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}
    label2id: Dict[str, int] = {}
    for hara_label, nli_name in _HARA_TO_NLI_LABEL.items():
        matches = [i for i, l in id2label.items() if nli_name in l or l in nli_name]
        if not matches:
            raise ValueError(
                f"Could not find a '{nli_name}' label in the checkpoint's id2label={id2label}; "
                f"expected an entailment/neutral/contradiction 3-way head."
            )
        label2id[hara_label] = matches[0]
    return label2id


def load_labeled_records(data_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_premise_cache(cache_path: str) -> Dict[str, Tuple[str, str]]:
    cache: Dict[str, Tuple[str, str]] = {}
    if not os.path.exists(cache_path):
        return cache
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cache[row["candidate_id"]] = (row["premise"], row["hypothesis"])
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def build_premises(records: List[Dict[str, Any]], cache_path: str) -> Tuple[List[str], List[str]]:
    """Builds the (premise, hypothesis) pair for every record the SAME way
    verify() does at inference: extract_claims() first (including yes/no
    reformulation — "yes"/"no" alone has no propositional content an NLI
    model can judge), then build_premise() on the resulting primary claim.

    A prior version skipped extract_claims() and used the raw
    candidate_answer as the hypothesis directly. For comparison/yes-no
    questions (candidate_answer often literally "yes"/"no"), that meant
    training on a degenerate hypothesis ("no") against a premise built from
    zero extracted entities (build_premise() never took its multi-entity,
    concatenate-both-sides branch) — a fundamentally different, mismatched
    input shape from what verify() constructs at inference (a reformulated
    comparative claim like "X and Y were of the same nationality" against
    both entities' evidence concatenated). Confirmed empirically: the
    resulting fine-tuned model regressed on exactly this claim shape
    (correct comparison answers flipped from the zero-shot checkpoint's
    ENTAILED to CONTRADICTED) while simple single-entity claims improved.

    Resumable via a JSONL cache keyed by candidate_id, since the
    cross-encoder pass is the expensive step here.
    """
    cache = _load_premise_cache(cache_path)
    premises: List[str] = [None] * len(records)  # type: ignore[list-item]
    hypotheses: List[str] = [None] * len(records)  # type: ignore[list-item]
    to_compute = []

    for i, rec in enumerate(records):
        key = rec.get("candidate_id") or rec["question_id"]
        if key in cache:
            premises[i], hypotheses[i] = cache[key]
        else:
            to_compute.append((i, key, rec))

    if to_compute:
        logger.info(f"Building premises for {len(to_compute)} records ({len(cache)} already cached)...")
        with open(cache_path, "a", encoding="utf-8") as f:
            for i, key, rec in tqdm(to_compute, desc="Building premises"):
                claims = extract_claims(rec["candidate_answer"], question=rec.get("question", ""))
                hypothesis = claims[0].text
                entities = [e for e in extract_entities(hypothesis) if is_comparable_entity(e)]
                premise, _ = build_premise(hypothesis, entities, rec["context"])
                premises[i] = premise
                hypotheses[i] = hypothesis
                f.write(json.dumps({"candidate_id": key, "premise": premise, "hypothesis": hypothesis}) + "\n")

    return premises, hypotheses  # type: ignore[return-value]


class NLIPairDataset(Dataset):
    def __init__(self, premises: List[str], hypotheses: List[str], labels: List[int],
                 tokenizer, max_length: int = MAX_LENGTH):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        encoded = self.tokenizer(
            self.premises[idx], self.hypotheses[idx],
            truncation=True, max_length=self.max_length, padding="max_length",
        )
        encoded["labels"] = self.labels[idx]
        return {k: torch.tensor(v) for k, v in encoded.items()}


def run_train_epoch(model, loader, optimizer, device, autocast_dtype, grad_accum: int) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="train")):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            out = model(**batch)
            loss = out.loss / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += out.loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype) -> Dict[str, float]:
    model.eval()
    total_loss, correct, n, n_batches = 0.0, 0, 0, 0
    for batch in tqdm(loader, desc="eval"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            out = model(**batch)
        total_loss += out.loss.item()
        n_batches += 1
        preds = out.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        n += batch["labels"].size(0)
    return {"eval_loss": total_loss / max(n_batches, 1), "accuracy": correct / max(n, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the DeBERTa-v3-large NLI verifier.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--premise-cache", type=str, default=DEFAULT_PREMISE_CACHE)
    parser.add_argument("--output-dir", type=str, default=FINE_TUNED_MODEL_DIR)
    parser.add_argument("--base-model", type=str, default=ZERO_SHOT_NLI_MODEL)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size (tuned for ~6GB VRAM)")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-split-name", type=str, default="val")
    parser.add_argument("--train-split-name", type=str, default="train")
    parser.add_argument("--lora", dest="lora", action="store_true", default=True,
                         help="Train via a LoRA adapter instead of full fine-tuning (default: on — "
                              "necessary to fit this model's optimizer state in 6-8GB VRAM)")
    parser.add_argument("--no-lora", dest="lora", action="store_false",
                         help="Disable LoRA and fully fine-tune all weights (needs ~7GB+ VRAM headroom)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (higher = more capacity, more memory)")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"],
                         help="Compute precision for the forward/backward pass. Model weights are always "
                              "loaded as fp32 (see dtype=torch.float32 above) regardless of this setting — "
                              "that fp32 master-weight load is what actually matters for stability; fp16 "
                              "additionally needs GradScaler, which DeBERTa-v3's disentangled attention "
                              "conflicts with ('Attempting to unscale FP16 gradients'), so bf16 is the "
                              "recommended fast option and fp16 is best avoided")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    logger.info(f"Loading labeled dataset from {args.data_path}")
    records = load_labeled_records(args.data_path)
    logger.info(f"{len(records)} labeled records loaded")

    train_records = [r for r in records if r.get("split", args.train_split_name) == args.train_split_name]
    val_records = [r for r in records if r.get("split") == args.val_split_name]
    if not val_records:
        logger.warning("No val-split records found — evaluating on a 10%% slice of train instead.")
        cut = max(1, len(train_records) // 10)
        val_records, train_records = train_records[:cut], train_records[cut:]
    logger.info(f"Train: {len(train_records)} records, Val: {len(val_records)} records")

    logger.info(f"Loading base checkpoint: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # This checkpoint's published weights are fp16. from_pretrained() loads
    # them in that native dtype by default, which — without Trainer's
    # mixed-precision fp32-master-weight handling — means AdamW optimizes
    # fp16 parameters directly: gradients underflow/explode and the loss
    # diverges to NaN within a handful of steps (confirmed empirically,
    # reproducing identically under fp32, bf16, AND fp16 Trainer settings,
    # since none of those flags touch the *storage* dtype loaded here).
    # Forcing fp32 storage fixes it regardless of the Trainer precision mode.
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, dtype=torch.float32)
    label2id = resolve_label2id(model)
    logger.info(f"HARA label -> checkpoint label id: {label2id}")

    if args.lora:
        lora_config = LoraConfig(
            task_type="SEQ_CLS", r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.1,
            target_modules=LORA_TARGET_MODULES, modules_to_save=LORA_MODULES_TO_SAVE,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    logger.info("Building training premises (reuses Stage_2_Verifier.build_premise)...")
    train_premises, train_hypotheses = build_premises(train_records, args.premise_cache)
    val_premises, val_hypotheses = build_premises(val_records, args.premise_cache)

    train_dataset = NLIPairDataset(
        train_premises, train_hypotheses,
        [label2id[r["label"]] for r in train_records], tokenizer,
    )
    val_dataset = NLIPairDataset(
        val_premises, val_hypotheses,
        [label2id[r["label"]] for r in val_records], tokenizer,
    )

    # transformers.Trainer hung indefinitely on this machine before the
    # first optimizer step even completed (confirmed: a raw manual
    # forward/backward/optimizer-step loop with the identical model/data
    # runs in ~0.2-0.4s/step, so the hang is in Trainer/accelerate's setup,
    # not the underlying compute) — replaced with a plain PyTorch loop.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    precision = args.precision
    if precision == "bf16" and not (device.type == "cuda" and torch.cuda.is_bf16_supported()):
        precision = "fp32"
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    logger.info(f"Starting fine-tuning on {device} (autocast dtype={autocast_dtype})...")
    best_accuracy = -1.0
    best_adapter_state = None  # small (~9MB) CPU-resident adapter weights only
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, device, autocast_dtype, args.grad_accum)
        metrics = evaluate(model, val_loader, device, autocast_dtype)
        logger.info(f"epoch {epoch}: train_loss={train_loss:.4f} eval_loss={metrics['eval_loss']:.4f} "
                    f"accuracy={metrics['accuracy']:.4f}")

        if metrics["accuracy"] >= best_accuracy:
            best_accuracy = metrics["accuracy"]
            logger.info(f"New best accuracy so far: {best_accuracy:.4f}")
            if args.lora:
                # Merging (merge_and_unload / merge_adapter) mid-training is
                # what caused a confirmed ~20x slowdown (3.7 it/s -> 0.18
                # it/s): merge_and_unload() on a copy.deepcopy() allocated a
                # full second GPU-resident model copy, and merge_adapter()'s
                # in-place merge doesn't actually help either — it only
                # merges weights *numerically*, the module tree stays
                # LoRA-wrapped (base_layer/lora_A/lora_B keys), so it isn't
                # even a save-able plain checkpoint on its own. Instead: keep
                # only the small adapter state dict (~9MB, not the full 1.7GB
                # model) on CPU between epochs, and do the one expensive
                # merge+save just once, after training finishes entirely —
                # nothing else needs the GPU by then, so it can't regress
                # training throughput.
                best_adapter_state = {
                    k: v.detach().cpu().clone()
                    for k, v in get_peft_model_state_dict(model).items()
                }
            else:
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)

    if args.lora and best_adapter_state is not None:
        logger.info(f"Merging best adapter (accuracy={best_accuracy:.4f}) and saving final checkpoint...")
        set_peft_model_state_dict(model, best_adapter_state)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    logger.info(f"Done. Best val accuracy={best_accuracy:.4f}. "
                f"Stage_2_Verifier.resolve_model_path() will now use {args.output_dir}.")


if __name__ == "__main__":
    main()

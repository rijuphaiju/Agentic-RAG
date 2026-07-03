"""
Stage 2 V2: Pretrained NLI / Factual-Consistency Verifier
===========================================================
Project: HARA — Hallucination-Aware Retrieval Agent

The core replacement for Stage 2 V1's fine-tuned classifier. This module
does NOT train anything on HotpotQA or on this project's own labels — it
loads a public checkpoint already trained for natural-language inference
(MNLI + FEVER + ANLI, in the default case) and scores (evidence, claim)
pairs zero-shot. Its competence at entailment comes entirely from that
external pretraining, which is the whole point: no similarity-derived
label from this project's own labeling heuristics ever touches its weights.

Default model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli — a widely used,
publicly available three-way entailment/neutral/contradiction checkpoint
(~184M params), practical to run on a single GPU/MPS/CPU. Swap in a larger
variant (e.g. the "-ling-wanli" mixture, or a MiniCheck checkpoint) via the
`model_name` argument if the smaller model's accuracy on the audit sample
turns out to be insufficient — see the Stage 2 V2 architecture document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

# faiss (imported by Stage_1_RAG_Pipeline, which every Stage 3/4/5/6 caller
# already imports before this module) initializes its own native OpenMP
# thread pool. If torch's multi-threaded CPU backend initializes afterward
# in the same process, loading or running this module's transformers model
# segfaults deterministically — reproduced and confirmed via bisection.
# Forcing torch to single-threaded CPU execution avoids the conflict; the
# NLI model runs on MPS/CUDA when available (see DEVICE below), where this
# has no meaningful performance cost — only CPU-side tensor ops are
# affected, and this module has no CPU-heavy workload of its own.
torch.set_num_threads(1)

import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MAX_LENGTH = 256
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

_LABEL_MAP = {
    "entailment": "ENTAILED",
    "entailed": "ENTAILED",
    "neutral": "NEUTRAL",
    "contradiction": "CONTRADICTED",
    "contradicted": "CONTRADICTED",
}


@dataclass
class NLIResult:
    verdict: str            # ENTAILED | NEUTRAL | CONTRADICTED
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float

    @property
    def confidence(self) -> float:
        return max(self.entailment_prob, self.neutral_prob, self.contradiction_prob)


class NLIVerifier:
    """Loads a pretrained NLI/factual-consistency checkpoint once and scores
    (evidence, claim) pairs zero-shot — no Stage-2-specific training."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, device: str = DEVICE):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self._label_order = self._resolve_label_order()

    def _resolve_label_order(self) -> List[str]:
        id2label = self.model.config.id2label
        order = []
        for i in range(len(id2label)):
            raw = id2label[i].lower()
            order.append(_LABEL_MAP.get(raw, raw.upper()))
        return order

    def score_batch(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """pairs: list of (premise/evidence, hypothesis/claim). A pair with
        an empty premise (no evidence matched at all) short-circuits to a
        zero-confidence NEUTRAL without spending a model call on it."""
        results: List[Optional[NLIResult]] = [None] * len(pairs)
        scoreable_idx = [i for i, (premise, _) in enumerate(pairs) if premise.strip()]

        if scoreable_idx:
            premises = [pairs[i][0] for i in scoreable_idx]
            hypotheses = [pairs[i][1] for i in scoreable_idx]
            inputs = self.tokenizer(
                premises, hypotheses, return_tensors="pt",
                padding=True, truncation=True, max_length=MAX_LENGTH,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu()
            for pos, i in enumerate(scoreable_idx):
                scores = {label: float(probs[pos][j]) for j, label in enumerate(self._label_order)}
                verdict = max(scores, key=scores.get)
                results[i] = NLIResult(
                    verdict=verdict,
                    entailment_prob=scores.get("ENTAILED", 0.0),
                    neutral_prob=scores.get("NEUTRAL", 0.0),
                    contradiction_prob=scores.get("CONTRADICTED", 0.0),
                )

        for i in range(len(pairs)):
            if results[i] is None:
                results[i] = NLIResult(
                    verdict="NEUTRAL", entailment_prob=0.0,
                    neutral_prob=1.0, contradiction_prob=0.0,
                )
        return results

    def score(self, premise: str, hypothesis: str) -> NLIResult:
        return self.score_batch([(premise, hypothesis)])[0]


def load_nli_verifier(model_name: str = DEFAULT_NLI_MODEL) -> NLIVerifier:
    return NLIVerifier(model_name=model_name)

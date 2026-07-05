"""
Stage 2: Verifier Training Dataset — Build + Label (offline tooling)
=======================================================================
Project: HARA — Hallucination-Aware Retrieval Agent

Offline data pipeline that produces the fine-tuning data for
Stage_2_Verifier_Train.py. Two modes, run in sequence:

    python Stage_2_Dataset.py build --num-samples 90447 --output verifier_dataset.jsonl
    python Stage_2_Dataset.py label --input verifier_dataset.jsonl --output verifier_dataset_labeled.jsonl

BUILD generates ~8-12 diverse candidate answers per HotpotQA train question
(gold answer, real Stage 1 answer, deterministic hard-negative
transformations, capped LLM paraphrases) against that question's own
retrieved-and-reranked passages — retrieval happens exactly once per
question and is reused by every candidate generator. Assigns no final
label, only an `expected_label_hint` (construction-implied prior, metadata
only).

LABEL reads BUILD's output and assigns SUPPORTED / PARTIAL / UNSUPPORTED via
a deterministic-first decision cascade (exact/containment match -> token F1
-> embedding cosine similarity -> LLM fallback for correctness; verbatim
span -> entity/token coverage -> LLM fallback for grounding). The LLM is a
minority fallback only, never the primary labeling authority. Every
decision records which tier/signal resolved it (`label_metadata`), and a
deterministic train/val/test split is assigned by question_id hash.

Both modes are resumable: re-running the same command continues from the
last completed question/record rather than restarting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import ollama
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from Stage_1_RAG_Pipeline import (
    CHUNK_SIZE,
    EMBED_MODEL,
    OLLAMA_MODEL,
    RERANK_POOL,
    TOP_K,
    build_example_corpus,
    generate_answer,
    llm_judge_supported,
    normalize_answer,
    rerank_passages,
    retrieve_hybrid,
    token_f1,
)

logger = logging.getLogger("stage2_dataset")


def setup_logging(verbose: bool = False) -> None:
    """Configures console logging for this script only (does not touch the
    root logger's handlers set up by any imported module)."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


# ════════════════════════════════════════════════════════════════════════
# MODE: build — multi-candidate dataset generation (no labels)
# ════════════════════════════════════════════════════════════════════════

BUILD_DEFAULT_OUTPUT          = "verifier_dataset.jsonl"
BUILD_DEFAULT_NUM_SAMPLES     = 12000
BUILD_DEFAULT_MAX_CANDIDATES  = 12
BUILD_DEFAULT_LLM_PARAPHRASES = 1     # 0-2; concise paraphrase first, partial-paraphrase second

_YEAR_RE       = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER_RE     = re.compile(r"\b\d+\b")
_DATE_RE       = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}\b"
)
_ENTITY_RE     = re.compile(r"\b[A-Z][a-zA-Z'.]*(?:\s+[A-Z][a-zA-Z'.]*){0,3}\b")
_RELATION_WORDS = {
    "before": "after", "after": "before",
    "older": "younger", "younger": "older",
    "first": "last", "last": "first",
    "earlier": "later", "later": "earlier",
}
_ORG_SUFFIXES = (
    "University", "College", "Company", "Corporation", "Inc", "Ltd",
    "Party", "Organization", "Institute", "Association", "Church",
    "League", "Studio", "Records", "Films", "Productions", "Band", "Group",
    "Team", "Council", "Committee",
)
_LOCATION_SUFFIXES = (
    "City", "County", "Province", "State", "Island", "River", "Mountain",
    "Republic", "Kingdom", "Bay", "Lake", "Valley", "Coast",
)
_STOPWORDS_LEADING = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "He", "She",
    "They", "We", "You", "His", "Her", "Their", "In", "On", "At", "Final", "Answer",
}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_RANDOM_ENTITIES = [
    "Albert Einstein", "Napoleon Bonaparte", "Marie Curie", "Winston Churchill",
    "The Beatles", "Microsoft", "United Nations", "Charles Darwin",
    "Leonardo da Vinci", "Nelson Mandela",
]
_RANDOM_LOCATIONS = [
    "Paris", "Tokyo", "Cairo", "Toronto", "Buenos Aires",
    "Reykjavik", "Nairobi", "Jakarta", "Lisbon", "Helsinki",
]


def _extract_entities(text: str, exclude: Optional[Set[str]] = None) -> List[str]:
    """Heuristic proper-noun span extraction. Not NER-grade — good enough to
    source plausible, in-context hard negatives without an extra model."""
    exclude = exclude or set()
    found: List[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        for m in _ENTITY_RE.finditer(sentence):
            span = m.group(0).strip().rstrip(".")
            words = span.split()
            if not words:
                continue
            if words[0] in _STOPWORDS_LEADING or words[-1] in _STOPWORDS_LEADING:
                continue
            if len(span) < 3 or span in exclude:
                continue
            found.append(span)
    return list(dict.fromkeys(found))


def _classify_entity(entity: str) -> str:
    if any(entity.endswith(suf) or f" {suf}" in entity for suf in _ORG_SUFFIXES):
        return "organization"
    if any(entity.endswith(suf) or f" {suf}" in entity for suf in _LOCATION_SUFFIXES):
        return "location"
    if len(entity.split()) == 2:
        return "person"
    return "other"


@dataclass
class Candidate:
    candidate_answer: str
    source: str                              # gold | stage1 | stage3 | stage4 | <generator name>
    generator: str                           # registry key that produced this candidate
    difficulty: str                          # trivial | model | llm | easy | medium | hard
    transformation: Optional[str] = None     # same as generator for synthetic sources, else None
    expected_label_hint: Optional[str] = None  # metadata only — NOT a final label


@dataclass
class GenerationContext:
    """Everything a candidate generator needs, computed once per question."""
    question: str
    question_type: str
    gold_answer: str
    reranked_passages: List[Dict[str, Any]]
    context_titles: List[str]                # all reranked-passage titles (for distractor picks)
    context_text: str                        # joined reranked-passage text (regex scanning source)
    supporting_titles: Set[str]              # gold-supporting titles, to identify true distractors
    stage1_answer: Optional[str] = None


_REGISTRY: List[Dict[str, Any]] = []


def register_generator(name: str, difficulty: str):
    def _decorator(fn: Callable[[GenerationContext], Optional[Tuple[str, Optional[str]]]]):
        _REGISTRY.append({"name": name, "difficulty": difficulty, "fn": fn})
        return fn
    return _decorator


# ---- EASY: obviously-wrong, not even context-plausible ----

@register_generator("wrong_year", "easy")
def _gen_wrong_year(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    m = _YEAR_RE.search(ctx.gold_answer)
    if not m:
        return None
    year = int(m.group(0))
    offset = random.choice([-44, -37, -25, 18, 29, 41])
    wrong_year = year + offset
    candidate = ctx.gold_answer[:m.start()] + str(wrong_year) + ctx.gold_answer[m.end():]
    return candidate, "UNSUPPORTED"


@register_generator("wrong_number", "easy")
def _gen_wrong_number(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    m = _NUMBER_RE.search(ctx.gold_answer)
    if not m or _YEAR_RE.fullmatch(m.group(0)):
        return None
    num = int(m.group(0))
    wrong_num = num + random.choice([-9, -5, 4, 7, 11])
    if wrong_num < 0:
        wrong_num = num + 3
    candidate = ctx.gold_answer[:m.start()] + str(wrong_num) + ctx.gold_answer[m.end():]
    return candidate, "UNSUPPORTED"


@register_generator("random_entity", "easy")
def _gen_random_entity(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    pool = [e for e in _RANDOM_ENTITIES if e.lower() not in ctx.gold_answer.lower()]
    if not pool:
        return None
    return random.choice(pool), "UNSUPPORTED"


@register_generator("random_location", "easy")
def _gen_random_location(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    pool = [e for e in _RANDOM_LOCATIONS if e.lower() not in ctx.gold_answer.lower()]
    if not pool:
        return None
    return random.choice(pool), "UNSUPPORTED"


# ---- MEDIUM: sourced from this question's own context, plausible ----

@register_generator("entity_replacement_context", "medium")
def _gen_entity_replacement(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    entities = _extract_entities(ctx.context_text, exclude={ctx.gold_answer})
    candidates = [e for e in entities if e.lower() not in ctx.gold_answer.lower()
                  and ctx.gold_answer.lower() not in e.lower()]
    if not candidates:
        return None
    return random.choice(candidates), "UNSUPPORTED"


@register_generator("number_date_replacement_context", "medium")
def _gen_number_date_replacement(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    has_num = _YEAR_RE.search(ctx.gold_answer) or _NUMBER_RE.search(ctx.gold_answer)
    if not has_num:
        return None
    context_years = [y for y in _YEAR_RE.findall(ctx.context_text) if y not in ctx.gold_answer]
    context_dates = [d for d in _DATE_RE.findall(ctx.context_text) if d not in ctx.gold_answer]
    pool = context_dates + context_years
    if not pool:
        return None
    replacement = random.choice(pool)
    m = _YEAR_RE.search(ctx.gold_answer) or _NUMBER_RE.search(ctx.gold_answer)
    candidate = ctx.gold_answer[:m.start()] + replacement + ctx.gold_answer[m.end():]
    return candidate, "UNSUPPORTED"


@register_generator("remove_important_entity", "medium")
def _gen_remove_entity(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    words = ctx.gold_answer.split()
    if len(words) < 2:
        return None
    entities = _extract_entities(ctx.gold_answer)
    if not entities:
        return None
    target = entities[0]
    candidate = ctx.gold_answer.replace(target, "", 1).strip()
    candidate = re.sub(r"\s{2,}", " ", candidate).strip(" ,.")
    if not candidate or candidate.lower() == ctx.gold_answer.lower():
        return None
    return candidate, "UNSUPPORTED"


@register_generator("truncation", "medium")
def _gen_truncation(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    words = ctx.gold_answer.split()
    if len(words) < 2:
        return None
    return words[0], "PARTIAL"


# ---- HARD: same-type, context-sourced, maximally plausible ----

def _typed_context_entities(ctx: GenerationContext, want_type: str) -> List[str]:
    entities = _extract_entities(ctx.context_text, exclude={ctx.gold_answer})
    return [e for e in entities
            if _classify_entity(e) == want_type
            and e.lower() not in ctx.gold_answer.lower()
            and ctx.gold_answer.lower() not in e.lower()]


@register_generator("plausible_organization_swap", "hard")
def _gen_org_swap(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    pool = _typed_context_entities(ctx, "organization")
    if not pool:
        return None
    return random.choice(pool), "UNSUPPORTED"


@register_generator("plausible_person_swap", "hard")
def _gen_person_swap(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    pool = _typed_context_entities(ctx, "person")
    if not pool:
        return None
    return random.choice(pool), "UNSUPPORTED"


@register_generator("plausible_location_swap", "hard")
def _gen_location_swap(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    pool = _typed_context_entities(ctx, "location")
    if not pool:
        return None
    return random.choice(pool), "UNSUPPORTED"


@register_generator("relation_corruption", "hard")
def _gen_relation_corruption(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    lower = ctx.gold_answer.strip().lower()
    if lower in ("yes", "no"):
        return ("No" if lower == "yes" else "Yes"), "UNSUPPORTED"
    for word, antonym in _RELATION_WORDS.items():
        if re.search(rf"\b{word}\b", ctx.gold_answer, re.IGNORECASE):
            candidate = re.sub(rf"\b{word}\b", antonym, ctx.gold_answer, flags=re.IGNORECASE)
            return candidate, "UNSUPPORTED"
    return None


@register_generator("unsupported_additional_clause", "hard")
def _gen_unsupported_clause(ctx: GenerationContext) -> Optional[Tuple[str, str]]:
    distractor_titles = [t for t in ctx.context_titles if t not in ctx.supporting_titles]
    if not distractor_titles:
        return None
    distractor = random.choice(distractor_titles)
    candidate = f"{ctx.gold_answer}, related to {distractor}"
    return candidate, "PARTIAL"


def _llm_paraphrase_concise(question: str, gold_answer: str, model: str = OLLAMA_MODEL) -> Optional[str]:
    prompt = (
        f"Question: {question}\n"
        f"Correct answer: {gold_answer}\n\n"
        f"Rewrite the correct answer using different wording, keeping the exact "
        f"same meaning. Reply with only the rewritten answer, as few words as "
        f"possible, no explanation.\n\nRewritten answer:"
    )
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 20},
        )
        text = response["message"]["content"].strip().strip('"')
        return text or None
    except Exception:
        logger.exception("LLM concise paraphrase failed")
        return None


def _llm_paraphrase_partial(question: str, gold_answer: str, model: str = OLLAMA_MODEL) -> Optional[str]:
    prompt = (
        f"Question: {question}\n"
        f"Correct answer: {gold_answer}\n\n"
        f"State the correct answer, then add one additional specific detail "
        f"that is plausible but not confirmed by the question. One short "
        f"sentence only, no explanation.\n\nAnswer:"
    )
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 40},
        )
        text = response["message"]["content"].strip().strip('"')
        return text or None
    except Exception:
        logger.exception("LLM partial paraphrase failed")
        return None


# Post-processing for real Stage 1/3/4 model answers before they're stored as
# a candidate_answer: shortens a verbose LLM answer to a compact, clause-
# boundary-aware phrase matching the gold-answer distribution (1-5 words),
# except when the LLM hedged ("not specified", "cannot determine") — in that
# case the hedging sentence itself is kept, since the first N words of a
# hedge are usually a grounded preamble that would otherwise mislabel a
# genuinely wrong/evasive answer as SUPPORTED.
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
    clause boundary within that budget over a raw word-count cut, so a
    truncated dangling fragment doesn't manufacture a spurious PARTIAL/
    UNSUPPORTED signal downstream."""
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


def distill_for_verify(answer: str, max_words: int = 12) -> str:
    """Shorten a verbose LLM answer to a compact phrase before it's stored
    as a training candidate — see module comment above _EVASIVE_PATTERNS."""
    m = re.search(r'(?:Final Answer|Answer)\s*:\s*(.+?)(?:\n|$)', answer, re.IGNORECASE)
    if m:
        answer = m.group(1).strip()

    if _EVASIVE_PATTERNS.search(answer):
        sentences = re.split(r'(?<=[.!?])\s', answer.strip())
        for sent in sentences:
            if _EVASIVE_PATTERNS.search(sent):
                return _truncate_at_clause_boundary(sent.strip(), max_words)

    first = re.split(r'(?<=[.!?])\s', answer.strip(), maxsplit=1)[0]
    return _truncate_at_clause_boundary(first, max_words)


@dataclass
class BuildStats:
    questions_processed: int = 0
    questions_skipped_empty: int = 0
    questions_skipped_error: int = 0
    total_candidates: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)
    hint_counts: Dict[str, int] = field(default_factory=dict)
    duplicates_dropped: int = 0
    example_by_source: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    def record(self, cand: Candidate, question_id: str) -> None:
        self.total_candidates += 1
        self.source_counts[cand.source] = self.source_counts.get(cand.source, 0) + 1
        hint = cand.expected_label_hint or "none"
        self.hint_counts[hint] = self.hint_counts.get(hint, 0) + 1
        if cand.source not in self.example_by_source:
            self.example_by_source[cand.source] = {
                "question_id": question_id,
                "candidate_answer": cand.candidate_answer,
                "generator": cand.generator,
                "expected_label_hint": cand.expected_label_hint,
            }

    def report(self) -> None:
        elapsed = time.time() - self.start_time
        avg_per_q = (self.total_candidates / self.questions_processed) if self.questions_processed else 0.0
        logger.info("─" * 60)
        logger.info("STAGE 2 DATASET GENERATION — SUMMARY")
        logger.info("─" * 60)
        logger.info(f"Questions processed        : {self.questions_processed}")
        logger.info(f"Questions skipped (empty)   : {self.questions_skipped_empty}")
        logger.info(f"Questions skipped (error)   : {self.questions_skipped_error}")
        logger.info(f"Total candidates written    : {self.total_candidates}")
        logger.info(f"Avg candidates / question   : {avg_per_q:.2f}")
        logger.info(f"Duplicate candidates dropped: {self.duplicates_dropped}")
        logger.info(f"Runtime                    : {elapsed / 60:.1f} min")
        logger.info("Source distribution:")
        for src, count in sorted(self.source_counts.items(), key=lambda kv: -kv[1]):
            logger.info(f"  {src:<32} {count}")
        logger.info("Expected-label-hint distribution (metadata only, not final labels):")
        for hint, count in sorted(self.hint_counts.items(), key=lambda kv: -kv[1]):
            logger.info(f"  {hint:<32} {count}")
        logger.info("Example candidate per source:")
        for src, ex in self.example_by_source.items():
            logger.info(f"  [{src}] {ex['candidate_answer']!r} (generator={ex['generator']}, hint={ex['expected_label_hint']})")


def load_processed_ids(output_path: str) -> Set[str]:
    """Scans an existing JSONL output file (if any) and returns the set of
    question_ids already fully written, so a re-run resumes instead of
    restarting."""
    processed: Set[str] = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                processed.add(record["question_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def _strip_passage(passage: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"title": passage["title"], "text": passage["text"]}
    if "rerank_score" in passage:
        out["rerank_score"] = round(float(passage["rerank_score"]), 4)
    return out


def generate_candidates(
    ctx: GenerationContext,
    max_candidates: int,
    llm_paraphrases: int,
    include_stage3: bool,
    include_stage4: bool,
    stage3_fn=None,
    stage4_fn=None,
) -> List[Candidate]:
    """Runs every applicable registered generator plus the fixed sources
    (gold/stage1/optional stage3/stage4/LLM paraphrases), then deduplicates
    by normalized answer text."""
    raw: List[Candidate] = []

    raw.append(Candidate(ctx.gold_answer, source="gold", generator="gold",
                          difficulty="trivial", expected_label_hint="SUPPORTED"))

    if ctx.stage1_answer:
        raw.append(Candidate(ctx.stage1_answer, source="stage1", generator="stage1",
                              difficulty="model", expected_label_hint=None))

    for entry in _REGISTRY:
        try:
            result = entry["fn"](ctx)
        except Exception:
            logger.exception(f"Generator {entry['name']} failed — skipping")
            continue
        if result is None:
            continue
        answer_text, hint = result
        if not answer_text or not answer_text.strip():
            continue
        raw.append(Candidate(
            answer_text.strip(), source=entry["name"], generator=entry["name"],
            difficulty=entry["difficulty"], transformation=entry["name"],
            expected_label_hint=hint,
        ))

    for i in range(max(0, min(llm_paraphrases, 2))):
        if i == 0:
            text = _llm_paraphrase_concise(ctx.question, ctx.gold_answer)
            name = "paraphrase_concise"
            hint = "SUPPORTED"
        else:
            text = _llm_paraphrase_partial(ctx.question, ctx.gold_answer)
            name = "paraphrase_partial"
            hint = "PARTIAL"
        if text:
            raw.append(Candidate(text, source=name, generator=name,
                                  difficulty="llm", expected_label_hint=hint))

    if include_stage3 and stage3_fn is not None:
        try:
            answer = stage3_fn()
            if answer:
                raw.append(Candidate(answer, source="stage3", generator="stage3",
                                      difficulty="model", expected_label_hint=None))
        except Exception:
            logger.exception("Stage 3 candidate generation failed")

    if include_stage4 and stage4_fn is not None:
        try:
            answer = stage4_fn()
            if answer:
                raw.append(Candidate(answer, source="stage4", generator="stage4",
                                      difficulty="model", expected_label_hint=None))
        except Exception:
            logger.exception("Stage 4 candidate generation failed")

    seen: Set[str] = set()
    deduped: List[Candidate] = []
    for cand in raw:
        key = normalize_answer(cand.candidate_answer)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(cand)

    if len(deduped) > max_candidates:
        protected = [c for c in deduped if c.source in ("gold", "stage1")]
        rest = [c for c in deduped if c.source not in ("gold", "stage1")]
        random.shuffle(rest)
        deduped = protected + rest[: max(0, max_candidates - len(protected))]

    return deduped


def build_records_for_question(
    example: Dict[str, Any],
    embedder: SentenceTransformer,
    chunk_size: int,
    max_candidates: int,
    llm_paraphrases: int,
    include_stage3: bool,
    include_stage4: bool,
    stage3_state=None,
    stage4_state=None,
) -> Optional[List[Dict[str, Any]]]:
    """Builds the per-question corpus and retrieves evidence exactly once,
    generates every candidate against that same evidence, deduplicates, and
    returns one JSON-ready record per surviving candidate."""
    question    = example["question"]
    gold_answer = example["answer"]

    ex_index, ex_passages, ex_bm25 = build_example_corpus(example, embedder, chunk_size=chunk_size)
    if ex_index is None:
        return None

    recall_pool = min(max(RERANK_POOL, 20), len(ex_passages))
    pool = retrieve_hybrid(
        question, ex_index, embedder, ex_passages,
        top_k=recall_pool, bm25=ex_bm25,
    )
    reranked = rerank_passages(question, pool, top_k=TOP_K)
    context_passages = [_strip_passage(p) for p in reranked]
    context_text = "\n".join(p["text"] for p in context_passages)
    context_titles = [p["title"] for p in context_passages]
    supporting_titles = set(example["supporting_facts"]["title"])
    question_type = example.get("type", "bridge")

    stage1_answer = generate_answer(question, reranked)

    gen_ctx = GenerationContext(
        question=question,
        question_type=question_type,
        gold_answer=gold_answer,
        reranked_passages=context_passages,
        context_titles=context_titles,
        context_text=context_text,
        supporting_titles=supporting_titles,
        stage1_answer=stage1_answer,
    )

    stage3_fn = None
    stage4_fn = None
    if include_stage3 and stage3_state is not None:
        adaptive_rag_query, verifier_model, verifier_tokenizer = stage3_state
        stage3_fn = lambda: adaptive_rag_query(
            question, ex_index, embedder, ex_passages,
            verifier_model=verifier_model, verifier_tokenizer=verifier_tokenizer,
            verbose=False, query_type_override=question_type,
            level_override=example.get("level"), bm25=ex_bm25,
        ).get("answer")
    if include_stage4 and stage4_state is not None:
        agentic_query, verifier_model, verifier_tokenizer = stage4_state

        def _stage4_best_effort():
            result = agentic_query(
                question, ex_index, embedder, ex_passages,
                verifier_model, verifier_tokenizer,
                verbose=False, query_type_override=question_type,
                level_override=example.get("level"), bm25=ex_bm25,
            )
            if result.get("status") in ("SUPPORTED", "BEST_EFFORT"):
                return result.get("answer")
            return None
        stage4_fn = _stage4_best_effort

    candidates = generate_candidates(
        gen_ctx, max_candidates=max_candidates, llm_paraphrases=llm_paraphrases,
        include_stage3=include_stage3, include_stage4=include_stage4,
        stage3_fn=stage3_fn, stage4_fn=stage4_fn,
    )

    supporting_facts = [
        {"title": t, "sent_id": s}
        for t, s in zip(
            example["supporting_facts"]["title"],
            example["supporting_facts"]["sent_id"],
        )
    ]

    records = []
    for cand in candidates:
        answer_text = cand.candidate_answer
        if cand.source in ("stage1", "stage3", "stage4"):
            answer_text = distill_for_verify(answer_text)
        records.append({
            "question_id": example["id"],
            "candidate_id": f'{example["id"]}::{cand.source}',
            "question": question,
            "question_type": question_type,
            "level": example.get("level", "unknown"),
            "gold_answer": gold_answer,
            "supporting_facts": supporting_facts,
            "context": context_passages,
            "candidate_answer": answer_text,
            "source": cand.source,
            "generator": cand.generator,
            "difficulty": cand.difficulty,
            "transformation": cand.transformation,
            "expected_label_hint": cand.expected_label_hint,
            "corpus_size": len(ex_passages),
        })
    return records


def run_build(args) -> None:
    random.seed(args.seed)
    logger.info("Stage 2 multi-candidate dataset generation starting")
    logger.info(f"Target: {args.num_samples} questions -> {args.output}")
    logger.info(f"Max candidates/question: {args.max_candidates}, LLM paraphrases: {args.llm_paraphrases}, "
                f"Stage3: {args.include_stage3}, Stage4: {args.include_stage4}")

    processed_ids = load_processed_ids(args.output)
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} questions already present in {args.output}")

    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    stage3_state = stage4_state = None
    if args.include_stage3 or args.include_stage4:
        from Stage_2_Verifier import load_verifier, VERIFIER_PATH
        verifier_model, verifier_tokenizer = load_verifier(args.verifier_path or VERIFIER_PATH)
        if args.include_stage3:
            from Stage_3_Adaptive_Retrieval import adaptive_rag_query
            stage3_state = (adaptive_rag_query, verifier_model, verifier_tokenizer)
        if args.include_stage4:
            from Stage_4_Agentic_Loop import agentic_query
            stage4_state = (agentic_query, verifier_model, verifier_tokenizer)

    logger.info("Loading HotpotQA train split (distractor)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="train")

    stats = BuildStats()
    done = len(processed_ids)

    pbar = tqdm(total=args.num_samples, initial=done, desc="Building verifier dataset")
    try:
        with open(args.output, "a", encoding="utf-8") as f:
            for example in dataset:
                if done >= args.num_samples:
                    break
                qid = example["id"]
                if qid in processed_ids:
                    continue
                try:
                    records = build_records_for_question(
                        example, embedder, chunk_size=args.chunk_size,
                        max_candidates=args.max_candidates,
                        llm_paraphrases=args.llm_paraphrases,
                        include_stage3=args.include_stage3,
                        include_stage4=args.include_stage4,
                        stage3_state=stage3_state, stage4_state=stage4_state,
                    )
                except Exception:
                    logger.exception(f"Error processing question {qid} — skipping")
                    stats.questions_skipped_error += 1
                    continue
                if records is None:
                    stats.questions_skipped_empty += 1
                    continue
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                for rec in records:
                    stats.record(Candidate(
                        rec["candidate_answer"], rec["source"], rec["generator"],
                        rec["difficulty"], rec["transformation"], rec["expected_label_hint"],
                    ), qid)
                stats.questions_processed += 1
                done += 1
                pbar.update(1)
    except KeyboardInterrupt:
        logger.warning(
            f"Interrupted by user. {done} questions saved to {args.output}. "
            f"Re-run the same command to resume from where it left off."
        )
    finally:
        pbar.close()

    stats.report()


# ════════════════════════════════════════════════════════════════════════
# MODE: label — SUPPORTED/PARTIAL/UNSUPPORTED assignment
# ════════════════════════════════════════════════════════════════════════

LABEL_DEFAULT_INPUT       = "verifier_dataset.jsonl"
LABEL_DEFAULT_OUTPUT      = "verifier_dataset_labeled.jsonl"
LABEL_DEFAULT_AUDIT_CSV   = "verifier_dataset_audit_sample.csv"
LABEL_DEFAULT_AUDIT_SIZE  = 200

# Thresholds are starting points — calibrate against the manual audit sample
# before trusting them at full scale.
F1_HIGH        = 0.70   # token F1 >= this -> confidently correct
F1_LOW         = 0.30   # token F1 <= this (with low embedding sim too) -> confidently incorrect
EMBED_SIM_HIGH = 0.85   # cosine sim >= this -> confidently correct (paraphrase)
EMBED_SIM_LOW  = 0.55   # cosine sim <= this -> confidently incorrect
COVERAGE_HIGH  = 0.80   # entity/token coverage >= this -> confidently grounded
COVERAGE_LOW   = 0.30   # entity/token coverage <= this -> confidently ungrounded

_SALIENT_TOKEN_RE = re.compile(r'\b[A-Z][a-zA-Z]{2,}\b|\b\d{3,4}\b')
_CLAUSE_SPLIT_RE = re.compile(
    r'(?:(?<=[.!?])\s+|\s+and also\s+|,\s+and also\s+|\s+as well as\s+|,\s+and\s+|\s+and\s+|,\s+)',
    re.IGNORECASE,
)
_LEADING_CONJ_RE = re.compile(r'^(?:and|also|but|while|with|as well as)\s+', re.IGNORECASE)
MIN_EXTRA_CLAIM_WORDS = 3   # below this (after stripping a leading conjunction), treat a split
                            # fragment as truncation noise, not a genuine extra claim


def _record_key(record: Dict[str, Any]) -> str:
    """Unique key for one labelable unit — per-candidate (via `candidate_id`)
    since the multi-candidate dataset writes several rows per question_id."""
    return record.get("candidate_id") or record["question_id"]


def load_labeled_keys(output_path: str) -> Set[str]:
    keys: Set[str] = set()
    if not os.path.exists(output_path):
        return keys
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(_record_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def assign_split(question_id: str, val_pct: int = 10, test_pct: int = 10) -> str:
    """Deterministic train/val/test split by question_id (hash, not a random
    shuffle) — reproducible across reruns."""
    bucket = int(hashlib.md5(question_id.encode()).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


def extract_salient_tokens(text: str) -> Set[str]:
    return {t.lower() for t in _SALIENT_TOKEN_RE.findall(text)}


def _cosine_sim(embedder: SentenceTransformer, a: str, b: str) -> float:
    """Fixed pretrained-embedding cosine similarity — a static metric, not a
    generative LLM judgment."""
    vecs = embedder.encode([a, b], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vecs)
    return float(np.dot(vecs[0], vecs[1]))


def _llm_equivalence_check(question: str, claim: str, gold_answer: str) -> bool:
    """Decision 1, Tier 4 — minority fallback only."""
    prompt = (
        f"Question: {question}\n"
        f"Candidate answer: {claim}\n"
        f"Reference answer: {gold_answer}\n\n"
        f"Is the candidate answer semantically equivalent to the reference "
        f"answer as a response to this question? Reply with only YES or NO."
    )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 5},
        )
        return response["message"]["content"].strip().upper().startswith("YES")
    except Exception:
        logger.exception("LLM equivalence check failed — defaulting to False")
        return False


def _llm_decompose_label(answer: str) -> List[str]:
    """Decision 3, minority fallback — invoked only when deterministic
    clause splitting can't confidently decide on a long, unstructured answer."""
    prompt = (
        f"List the distinct factual claims made in the following sentence, "
        f"one per line, as short phrases. If it makes only one claim, "
        f"return just that one line.\n\nSentence: {answer}\n\nClaims:"
    )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 80},
        )
        lines = [l.strip("-• \t") for l in response["message"]["content"].splitlines() if l.strip()]
        return lines if lines else [answer]
    except Exception:
        logger.exception("LLM decomposition failed — treating answer as a single claim")
        return [answer]


def assess_correctness(
    claim: str, gold_answer: str, question: str, embedder: SentenceTransformer,
) -> Tuple[bool, int, str]:
    """Is `claim` correct relative to `gold_answer`? Returns (is_correct, tier, signal)."""
    norm_claim = normalize_answer(claim)
    norm_gold = normalize_answer(gold_answer)

    if not norm_gold:
        return True, 1, "empty_gold"

    if norm_claim == norm_gold:
        return True, 1, "exact_match"
    if norm_gold in norm_claim or norm_claim in norm_gold:
        return True, 1, "containment"

    f1 = token_f1(claim, gold_answer)
    if f1 >= F1_HIGH:
        return True, 2, f"token_f1={f1:.2f}"

    sim = _cosine_sim(embedder, claim, gold_answer)
    if sim >= EMBED_SIM_HIGH:
        return True, 3, f"embedding_sim={sim:.2f}"
    if sim <= EMBED_SIM_LOW and f1 <= F1_LOW:
        return False, 3, f"embedding_sim={sim:.2f},token_f1={f1:.2f}"

    is_equiv = _llm_equivalence_check(question, claim, gold_answer)
    return is_equiv, 4, "llm_equivalence"


def assess_grounding(
    claim: str,
    reranked_passages: List[Dict[str, Any]],
    supporting_titles: Set[str],
    question: str,
) -> Tuple[bool, int, str]:
    """Is `claim` evidenced by the passages actually given to the LLM?
    Returns (is_grounded, tier, signal)."""
    if not normalize_answer(claim):
        return False, 1, "empty_claim"

    norm_claim = normalize_answer(claim)
    context_concat = " ".join(p["text"] for p in reranked_passages)

    matching_titles = {
        p["title"] for p in reranked_passages
        if norm_claim and norm_claim in normalize_answer(p["text"])
    }
    if matching_titles:
        if matching_titles & supporting_titles:
            return True, 1, "verbatim_span_supporting_title"
        return True, 1, "verbatim_span_other_title"

    claim_tokens = extract_salient_tokens(claim)
    if claim_tokens:
        context_tokens = extract_salient_tokens(context_concat)
        coverage = len(claim_tokens & context_tokens) / len(claim_tokens)
        if coverage >= COVERAGE_HIGH:
            return True, 2, f"entity_coverage={coverage:.2f}"
        if coverage <= COVERAGE_LOW:
            return False, 2, f"entity_coverage={coverage:.2f}"
    else:
        overlap = token_f1(claim, context_concat)
        if overlap >= COVERAGE_HIGH:
            return True, 2, f"token_overlap={overlap:.2f}"
        if overlap <= COVERAGE_LOW:
            return False, 2, f"token_overlap={overlap:.2f}"

    is_grounded = llm_judge_supported(question, claim, reranked_passages)
    return is_grounded, 3, "llm_judge_supported"


def _is_meaningful_claim(text: str) -> bool:
    """Filters out truncation artifacts / dangling fragments from being
    counted as genuine extra claims."""
    stripped = _LEADING_CONJ_RE.sub("", text).strip()
    return len(stripped.split()) >= MIN_EXTRA_CLAIM_WORDS


def decompose_claims(processed_answer: str, gold_answer: str) -> Tuple[str, List[str], int, str]:
    """Splits `processed_answer` into a core claim (best correctness match
    against gold_answer) and zero or more extra claims. Returns (core_claim,
    extra_claims, tier, signal)."""
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(processed_answer) if p.strip()]
    if not parts:
        return processed_answer.strip(), [], 1, "no_split_needed"

    if len(parts) == 1:
        only = parts[0]
        norm_only, norm_gold = normalize_answer(only), normalize_answer(gold_answer)
        looks_like_single_claim = (
            len(only.split()) <= 15
            or (norm_gold and (norm_gold in norm_only or norm_only in norm_gold))
        )
        if looks_like_single_claim:
            return only, [], 1, "single_clause"
        decomposed = _llm_decompose_label(only)
        if len(decomposed) > 1:
            extras = [c for c in decomposed[1:] if _is_meaningful_claim(c)]
            return decomposed[0], extras, 2, "llm_decomposition"
        return only, [], 1, "single_clause"

    def _match_score(clause: str) -> float:
        norm_c, norm_g = normalize_answer(clause), normalize_answer(gold_answer)
        if norm_g and (norm_g in norm_c or norm_c in norm_g):
            return 1.0
        return token_f1(clause, gold_answer)

    best_idx = max(range(len(parts)), key=lambda i: _match_score(parts[i]))
    core = parts[best_idx]
    extras = [p for i, p in enumerate(parts) if i != best_idx and _is_meaningful_claim(p)]
    return core, extras, 1, "clause_split"


def label_record(record: Dict[str, Any], embedder: SentenceTransformer) -> Dict[str, Any]:
    """Runs the full Decision 1/2/3 cascade on one dataset record and returns
    the record augmented with `label`, `label_metadata`, and `split`."""
    question = record["question"]
    gold_answer = record["gold_answer"]
    processed_answer = record.get("candidate_answer", record.get("processed_answer"))
    reranked_passages = record.get("context", record.get("reranked_passages"))
    supporting_titles = {sf["title"] for sf in record["supporting_facts"]}

    core_claim, extra_claims, decomp_tier, decomp_signal = decompose_claims(processed_answer, gold_answer)

    core_correct, correct_tier, correct_signal = assess_correctness(
        core_claim, gold_answer, question, embedder
    )
    core_grounded, grounded_tier, grounded_signal = assess_grounding(
        core_claim, reranked_passages, supporting_titles, question
    )

    extra_claim_results = []
    for extra in extra_claims:
        grounded, tier, signal = assess_grounding(extra, reranked_passages, supporting_titles, question)
        extra_claim_results.append({"text": extra, "grounded": grounded, "tier": tier, "signal": signal})

    if not (core_correct and core_grounded):
        label = "UNSUPPORTED"
    elif all(c["grounded"] for c in extra_claim_results):
        label = "SUPPORTED"
    else:
        label = "PARTIAL"

    out = dict(record)
    out["label"] = label
    out["label_metadata"] = {
        "core_claim": core_claim,
        "core_correct": core_correct,
        "core_correct_tier": correct_tier,
        "core_correct_signal": correct_signal,
        "core_grounded": core_grounded,
        "core_grounded_tier": grounded_tier,
        "core_grounded_signal": grounded_signal,
        "decomposition_tier": decomp_tier,
        "decomposition_signal": decomp_signal,
        "extra_claims": extra_claim_results,
    }
    out["split"] = assign_split(record["question_id"])
    return out


def export_audit_sample(labeled_path: str, csv_path: str, sample_size: int, seed: int = 42) -> None:
    """Writes a random sample of labeled records to a flat CSV for manual
    review — the calibration set for the deterministic thresholds above."""
    records = []
    with open(labeled_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    random.Random(seed).shuffle(records)
    sample = records[:sample_size]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id", "source", "question", "gold_answer", "candidate_answer",
            "label", "core_correct", "core_correct_signal",
            "core_grounded", "core_grounded_signal", "num_extra_claims",
            "human_verdict_agrees",  # left blank for manual annotation
        ])
        for r in sample:
            md = r["label_metadata"]
            candidate_answer = r.get("candidate_answer", r.get("processed_answer"))
            writer.writerow([
                r["question_id"], r.get("source", "stage1"), r["question"], r["gold_answer"], candidate_answer,
                r["label"], md["core_correct"], md["core_correct_signal"],
                md["core_grounded"], md["core_grounded_signal"], len(md["extra_claims"]),
                "",
            ])
    logger.info(f"Audit sample ({len(sample)} records) written to {csv_path}")


def run_label(args) -> None:
    logger.info(f"Stage 2 label generation starting: {args.input} -> {args.output}")

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    processed_keys = load_labeled_keys(args.output)
    if processed_keys:
        logger.info(f"Resuming: {len(processed_keys)} records already labeled in {args.output}")

    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    with open(args.input, "r", encoding="utf-8") as f:
        input_records = [json.loads(line) for line in f if line.strip()]
    logger.info(f"{len(input_records)} records loaded from {args.input}")

    label_counts: Counter = Counter()
    tier_counts: Counter = Counter()
    llm_fallback_count = 0
    done = len(processed_keys)

    pbar = tqdm(total=len(input_records), initial=done, desc="Labeling")
    try:
        with open(args.output, "a", encoding="utf-8") as out_f:
            for record in input_records:
                key = _record_key(record)
                if key in processed_keys:
                    continue
                qid = record["question_id"]
                try:
                    labeled = label_record(record, embedder)
                except Exception:
                    logger.exception(f"Error labeling question {qid} — skipping")
                    continue

                out_f.write(json.dumps(labeled) + "\n")
                out_f.flush()

                label_counts[labeled["label"]] += 1
                md = labeled["label_metadata"]
                tier_counts[f"correct_tier_{md['core_correct_tier']}"] += 1
                tier_counts[f"grounded_tier_{md['core_grounded_tier']}"] += 1
                if md["core_correct_tier"] == 4 or md["core_grounded_tier"] == 3:
                    llm_fallback_count += 1

                done += 1
                pbar.update(1)
    except KeyboardInterrupt:
        logger.warning(f"Interrupted. {done} records labeled so far in {args.output}. Re-run to resume.")
    finally:
        pbar.close()

    logger.info(f"Done. {done} records labeled -> {args.output}")
    logger.info(f"Label distribution: {dict(label_counts)}")
    logger.info(f"Tier usage: {dict(tier_counts)}")
    if done:
        logger.info(f"LLM fallback rate: {llm_fallback_count}/{done} ({100*llm_fallback_count/done:.1f}%)")

    export_audit_sample(args.output, args.audit_csv, args.audit_size)


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: build and label the verifier fine-tuning dataset."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    build_p = subparsers.add_parser("build", help="Generate multi-candidate dataset (no labels)")
    build_p.add_argument("--num-samples", type=int, default=BUILD_DEFAULT_NUM_SAMPLES,
                          help=f"Number of HotpotQA train questions to process (default: {BUILD_DEFAULT_NUM_SAMPLES})")
    build_p.add_argument("--output", type=str, default=BUILD_DEFAULT_OUTPUT,
                          help=f"JSONL output path; re-running with the same path resumes (default: {BUILD_DEFAULT_OUTPUT})")
    build_p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                          help="Sentences per passage chunk, passed to build_example_corpus")
    build_p.add_argument("--max-candidates", type=int, default=BUILD_DEFAULT_MAX_CANDIDATES,
                          help=f"Cap on candidates written per question (default: {BUILD_DEFAULT_MAX_CANDIDATES})")
    build_p.add_argument("--llm-paraphrases", type=int, default=BUILD_DEFAULT_LLM_PARAPHRASES, choices=[0, 1, 2],
                          help="Number of LLM paraphrase candidates per question, 0-2 (default: 1)")
    build_p.add_argument("--include-stage3", action="store_true",
                          help="Also generate a Stage 3 candidate per question (several extra LLM calls; off by default)")
    build_p.add_argument("--include-stage4", action="store_true",
                          help="Also generate a Stage 4 BEST_EFFORT candidate per question (most expensive; off by default)")
    build_p.add_argument("--verifier-path", type=str, default=None,
                          help="Verifier checkpoint path, required only if --include-stage3/--include-stage4 is set")
    build_p.add_argument("--seed", type=int, default=42, help="Random seed for transformation sampling")
    build_p.add_argument("--verbose", action="store_true", help="Enable debug-level logging")

    label_p = subparsers.add_parser("label", help="Assign SUPPORTED/PARTIAL/UNSUPPORTED labels")
    label_p.add_argument("--input", type=str, default=LABEL_DEFAULT_INPUT)
    label_p.add_argument("--output", type=str, default=LABEL_DEFAULT_OUTPUT)
    label_p.add_argument("--audit-csv", type=str, default=LABEL_DEFAULT_AUDIT_CSV)
    label_p.add_argument("--audit-size", type=int, default=LABEL_DEFAULT_AUDIT_SIZE)
    label_p.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.mode == "build":
        run_build(args)
    else:
        run_label(args)


if __name__ == "__main__":
    main()

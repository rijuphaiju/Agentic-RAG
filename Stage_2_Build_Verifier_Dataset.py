"""
Stage 2: Multi-Candidate Verifier Dataset Generator
====================================================
Project: HARA — Hallucination-Aware Retrieval Agent

Replaces the old "one Stage 1 answer per question" dataset generator. The
verifier must learn to judge ANY candidate answer against retrieved evidence
regardless of which stage produced it — so this script generates roughly
8-12 diverse candidate answers per HotpotQA train question instead of one:
gold answer, real Stage 1 answer, a battery of cheap deterministic
transformations (easy/medium/hard hard-negatives, truncations, corruptions),
and a small, capped number of LLM paraphrases.

This script assigns NO final label. Each candidate optionally carries an
`expected_label_hint` — a construction-implied prior (e.g. a number-corrupted
answer is *probably* UNSUPPORTED) — but that is metadata only. The downstream
labeling pipeline (Stage_2_Label_Generator.py) remains the single source of
truth for SUPPORTED/PARTIAL/UNSUPPORTED, using its existing deterministic
correctness/grounding cascade with an LLM fallback for genuinely ambiguous
cases. That cascade doesn't care about provenance, so it needs no rewrite to
label these candidates — it already takes (question, context, candidate,
gold, supporting_facts) and works for any source.

Pipeline per question:
    build_example_corpus() -> retrieve_hybrid() -> rerank_passages()  [ONCE]
    -> generate candidates from every applicable registered generator
    -> deduplicate by normalized answer text
    -> write one JSON record per surviving candidate

Retrieval happens exactly once per question and is reused by every
candidate generator (deterministic transforms and LLM paraphrases all
operate on the same reranked passages; no repeated retrieval). Stage 1's
real answer costs one Ollama call (as before). LLM paraphrases are capped at
1-2 calls per question by default. Stage 3/Stage 4 candidates are OFF by
default (each costs several additional LLM-involving steps) and can be
enabled with --include-stage3 / --include-stage4 for smaller, targeted runs.

Only the HotpotQA TRAIN split is read here. The validation split is never
touched, so it stays uncontaminated for downstream evaluation.

Usage:
    python Stage_2_Build_Verifier_Dataset.py --num-samples 12000 --output verifier_dataset.jsonl
    (re-running the same command resumes from the last completed question)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ollama

from Stage_1_RAG_Pipeline import (
    build_example_corpus,
    retrieve_hybrid,
    rerank_passages,
    generate_answer,
    normalize_answer,
    EMBED_MODEL,
    RERANK_POOL,
    TOP_K,
    CHUNK_SIZE,
    OLLAMA_MODEL,
)
from Stage_2_Verifier_GPU import _distill_for_verify

logger = logging.getLogger("stage2_dataset_builder")

DEFAULT_OUTPUT               = "verifier_dataset.jsonl"
DEFAULT_NUM_SAMPLES          = 12000
DEFAULT_MAX_CANDIDATES       = 12
DEFAULT_LLM_PARAPHRASES      = 1     # 0-2; concise paraphrase first, partial-paraphrase second


def setup_logging(verbose: bool = False) -> None:
    """Configures console logging for this script only (does not touch the
    root logger's handlers set up by any imported module)."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


# ─────────────────────────────────────────────
# CANDIDATE DATACLASS
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# DETERMINISTIC EXTRACTION HELPERS
# (heuristic, regex-based — no NER model dependency)
# ─────────────────────────────────────────────

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
    "before": "after",
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
    source plausible, in-context hard negatives without an extra model.

    Extraction runs per-sentence so a span can never merge the last
    capitalized word of one sentence with the first capitalized word of the
    next (e.g. "Scholastic. It").
    """
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
    return list(dict.fromkeys(found))  # de-dup, preserve order


def _classify_entity(entity: str) -> str:
    if any(entity.endswith(suf) or f" {suf}" in entity for suf in _ORG_SUFFIXES):
        return "organization"
    if any(entity.endswith(suf) or f" {suf}" in entity for suf in _LOCATION_SUFFIXES):
        return "location"
    if len(entity.split()) == 2:
        return "person"
    return "other"


# ─────────────────────────────────────────────
# CANDIDATE GENERATOR REGISTRY
# Each generator: (GenerationContext) -> Optional[Tuple[answer_text, expected_label_hint]]
# Registering a new generator = adding one entry here; orchestration never
# needs to change.
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# LLM PARAPHRASE GENERATION (capped, minority use)
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────

@dataclass
class Stats:
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


# ─────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────

def load_processed_ids(output_path: str) -> Set[str]:
    """Scans an existing JSONL output file (if any) and returns the set of
    question_ids already fully written, so a re-run resumes instead of
    restarting. A question is "done" once any of its candidate rows appear —
    all candidates for a question are written together in one batch, so
    partial-question writes cannot occur except on a hard interrupt, which
    is tolerated by skipping malformed trailing lines.
    """
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
    by normalized answer text. Caps the final count at max_candidates while
    always keeping gold and Stage 1 (the two guaranteed, free/cheap sources).
    """
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
    returns one JSON-ready record per surviving candidate. Returns None if
    the example's context is empty.
    """
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
            answer_text = _distill_for_verify(answer_text)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: build a multi-candidate verifier training dataset (no labels)."
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                         help=f"Number of HotpotQA train questions to process (default: {DEFAULT_NUM_SAMPLES})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                         help=f"JSONL output path; re-running with the same path resumes (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                         help="Sentences per passage chunk, passed to build_example_corpus")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES,
                         help=f"Cap on candidates written per question (default: {DEFAULT_MAX_CANDIDATES})")
    parser.add_argument("--llm-paraphrases", type=int, default=DEFAULT_LLM_PARAPHRASES, choices=[0, 1, 2],
                         help="Number of LLM paraphrase candidates per question, 0-2 (default: 1)")
    parser.add_argument("--include-stage3", action="store_true",
                         help="Also generate a Stage 3 candidate per question (several extra LLM calls; off by default)")
    parser.add_argument("--include-stage4", action="store_true",
                         help="Also generate a Stage 4 BEST_EFFORT candidate per question (most expensive; off by default)")
    parser.add_argument("--verifier-path", type=str, default=None,
                         help="Verifier checkpoint path, required only if --include-stage3/--include-stage4 is set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for transformation sampling")
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
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
        from Stage_2_Verifier_GPU import load_verifier, VERIFIER_PATH
        verifier_model, verifier_tokenizer = load_verifier(args.verifier_path or VERIFIER_PATH)
        if args.include_stage3:
            from Stage_3_Adaptive_Retrieval import adaptive_rag_query
            stage3_state = (adaptive_rag_query, verifier_model, verifier_tokenizer)
        if args.include_stage4:
            from Stage_4_Agentic_Loop import agentic_query
            stage4_state = (agentic_query, verifier_model, verifier_tokenizer)

    logger.info("Loading HotpotQA train split (distractor)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="train")

    stats = Stats()
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


if __name__ == "__main__":
    main()

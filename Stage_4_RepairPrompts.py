"""
Stage 4: Diagnosis-Specific Repair Prompts
============================================
Project: HARA — Hallucination-Aware Retrieval Agent

Stage 2 V2's failure_reason previously only steered WHICH action Stage 4
took (EXPAND/REWRITE/COMPARE/...); the actual regeneration prompt was
always the same generic question regardless of what failed. This module
turns each failure_reason into a specific, targeted instruction appended to
the query before regeneration — so a YEAR-type mismatch tells the model
exactly that, a WRONG_ENTITY failure tells it to re-identify the entity from
evidence, a PARTIAL answer tells it which claims already held up and asks
only for what's missing, etc.

Appended, never prepended: Stage 1's generate_answer() dispatches its
internal prompt template by pattern-matching the LEADING words of the query
(e.g. `q_lower.startswith("who ")`, `re.match(r'^(?:when |what year...)`).
Appending the repair instruction after the original question preserves
those checks untouched — Stage 1 itself is never modified.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _claim_summaries(claims: List[Dict[str, Any]]) -> tuple:
    supported = [c["claim"] for c in claims if c.get("label") == "SUPPORTED"]
    unsupported = [c["claim"] for c in claims if c.get("label") != "SUPPORTED"]
    return supported, unsupported


def build_repair_instruction(diagnosis: Dict[str, Any]) -> Optional[str]:
    """Returns a natural-language instruction to append to the query for
    regeneration, tailored to `diagnosis["failure_reason"]` — or None when
    there's no specific instruction to add (the caller then falls back to
    plain regeneration against the newly retrieved evidence)."""
    reason = diagnosis.get("failure_reason")
    intent = diagnosis.get("question_intent") or {}
    expected_type = intent.get("expected_type", "UNKNOWN")
    claims = diagnosis.get("claims") or []

    if reason == "QUESTION_NOT_ANSWERED":
        return (
            "Your previous answer did not actually answer this question. "
            "Re-read the question carefully and answer exactly what is being "
            "asked, using only the retrieved evidence."
        )

    if reason == "ANSWER_TYPE_MISMATCH" and expected_type not in ("UNKNOWN", None):
        return (
            f"The question expects an answer of type {expected_type}. "
            f"Your previous answer was not a {expected_type}. "
            f"Use only the retrieved evidence and return only the {expected_type.lower()}."
        )

    if reason == "WRONG_ENTITY":
        return (
            "Your previous answer named the wrong entity. Carefully identify "
            "the correct entity from the retrieved evidence that actually "
            "matches what the question asks."
        )

    if reason == "WRONG_YEAR":
        return (
            "Your previous answer stated the wrong year. Use only the year "
            "explicitly stated in the retrieved evidence."
        )

    if reason == "WRONG_NUMBER":
        return (
            "Your previous answer stated the wrong number. Use only the "
            "specific number explicitly stated in the retrieved evidence."
        )

    if reason in ("MISSING_REQUIRED_ENTITY", "MISSING_REQUIRED_NUMBER", "MISSING_REQUIRED_DATE"):
        missing_kind = {"MISSING_REQUIRED_ENTITY": "entity", "MISSING_REQUIRED_NUMBER": "number",
                        "MISSING_REQUIRED_DATE": "date"}[reason]
        return (
            f"Your previous answer is missing a required {missing_kind}. "
            f"Include the specific {missing_kind} from the retrieved evidence."
        )

    if reason == "PARTIAL_COMPARISON":
        return (
            "This question requires comparing two entities. Explicitly compare "
            "both entities using the retrieved evidence, then state which one "
            "satisfies the question — do not hedge with 'both' or 'unclear'."
        )

    if reason == "CONTRADICTED_BY_EVIDENCE":
        return (
            "Your previous answer was contradicted by the retrieved evidence. "
            "Re-check the evidence carefully and correct the answer, or state "
            "plainly that it cannot be determined if the evidence truly disagrees."
        )

    if reason == "MULTIPLE_CONFLICTING_PASSAGES":
        return (
            "Multiple retrieved passages disagree or discuss different entities. "
            "Determine which passage is actually about the entity this question "
            "asks about before answering."
        )

    if reason in ("NO_RELEVANT_PASSAGE", "INSUFFICIENT_EVIDENCE"):
        return (
            "The retrieved evidence does not clearly contain the answer. Use "
            "only the retrieved evidence — if it genuinely lacks the answer, "
            "say so rather than guessing."
        )

    if diagnosis.get("label") == "PARTIAL" and claims:
        supported, unsupported = _claim_summaries(claims)
        if supported and unsupported:
            return (
                f"Your previous answer correctly established: {'; '.join(supported)}. "
                f"However, this part was not confirmed by the evidence: "
                f"{'; '.join(unsupported)}. Provide only the missing information, "
                f"using the retrieved evidence."
            )

    return None


def augment_query_for_repair(query: str, diagnosis: Dict[str, Any]) -> str:
    """Appends the repair instruction (if any) to `query` for regeneration.
    Never prepends — see module docstring for why that matters."""
    instruction = build_repair_instruction(diagnosis)
    if not instruction:
        return query
    return f"{query}\n\n{instruction}"

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import SETTINGS, get_logger
from app.modules.llm import call_llm_json
from app.modules.parent_context import ExpandedContext

_log = get_logger("validation")


@dataclass
class ValidationResult:
    """Structured outcome of validating an answer against its evidence."""
    sufficient_evidence: bool
    context_relevant: bool
    has_contradictions: bool
    confidence: float          # 0-1
    notes: str

    @property
    def passed(self) -> bool:
        """Whether the answer is acceptable as-is (no retry needed)."""
        return self.sufficient_evidence and self.context_relevant and not self.has_contradictions \
            and self.confidence >= 0.5

    @classmethod
    def skipped(cls, reason: str) -> "ValidationResult":
        return cls(
            sufficient_evidence=True, context_relevant=True,
            has_contradictions=False, confidence=0.75, notes=reason,
        )


def validate_answer(
    query: str,
    answer: str,
    expanded_contexts: list[ExpandedContext],
) -> ValidationResult:
    evidence_summary = "\n".join(f"- {ec.core.text[:200]}" for ec in expanded_contexts)

    prompt = f"""Evaluate whether this ANSWER is properly grounded in the EVIDENCE for
the given QUESTION.

QUESTION: {query}

EVIDENCE (excerpt previews):
{evidence_summary}

ANSWER: {answer}

Reply with ONLY a JSON object:
{{"sufficient_evidence": <true/false, does the evidence contain enough to fully
   answer the question>,
  "context_relevant": <true/false, is the evidence actually about the question's topic>,
  "has_contradictions": <true/false, does the answer contradict or go beyond the evidence>,
  "confidence": <0-1 float, overall confidence the answer is correct and grounded>,
  "notes": "<one sentence explaining the judgment>"}}"""

    result = call_llm_json(
        prompt,
        system_prompt="You are a strict, skeptical fact-checking evaluator.",
        max_tokens=SETTINGS.groq_max_tokens_validate,
    )
    if not result:
        return ValidationResult(
            sufficient_evidence=False, context_relevant=False,
            has_contradictions=True, confidence=0.0,
            notes="Validator call failed; treating as unverified.",
        )

    return ValidationResult(
        sufficient_evidence=bool(result.get("sufficient_evidence", False)),
        context_relevant=bool(result.get("context_relevant", False)),
        has_contradictions=bool(result.get("has_contradictions", True)),
        confidence=float(result.get("confidence", 0.0) or 0.0),
        notes=result.get("notes", ""),
    )

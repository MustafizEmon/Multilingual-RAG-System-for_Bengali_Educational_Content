from __future__ import annotations

from dataclasses import dataclass
import functools
import time

from app.core.config import get_logger
from app.modules.llm import call_llm_json
from app.modules.parent_context import ExpandedContext

_log = get_logger("eval_utils")


# --- Retrieval metrics --------------------------------------------------------------
def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


@dataclass
class RetrievalEvalCase:
    """One labeled evaluation example for retrieval metrics."""
    query: str
    relevant_chunk_ids: set[str]


def evaluate_retrieval(cases: list[RetrievalEvalCase], retrieve_fn, k: int = 10) -> dict:
    recalls, rrs = [], []
    for case in cases:
        retrieved_ids = retrieve_fn(case.query)
        recalls.append(recall_at_k(retrieved_ids, case.relevant_chunk_ids, k))
        rrs.append(mean_reciprocal_rank(retrieved_ids, case.relevant_chunk_ids))

    return {
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": sum(rrs) / len(rrs) if rrs else 0.0,
        "n_cases": len(cases),
    }


# --- LLM-judged generation-quality metrics -------------------------------------------
def _llm_score(prompt: str, system_prompt: str) -> float:
    """Shared helper: ask the LLM for a single 0-1 JSON score."""
    result = call_llm_json(prompt, system_prompt=system_prompt)
    try:
        return max(0.0, min(1.0, float(result.get("score", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def faithfulness_score(answer: str, expanded_contexts: list[ExpandedContext]) -> float:

    evidence = "\n".join(f"- {ec.core.text[:200]}" for ec in expanded_contexts)
    prompt = f"""EVIDENCE:\n{evidence}\n\nANSWER:\n{answer}\n\nWhat fraction of the ANSWER's
claims are directly supported by the EVIDENCE? Reply with ONLY:
{{"score": <0-1 float>}}"""
    return _llm_score(prompt, "You are a strict faithfulness/hallucination judge.")


def answer_relevance_score(query: str, answer: str) -> float:

    prompt = f"""QUESTION:\n{query}\n\nANSWER:\n{answer}\n\nHow directly and completely
does the ANSWER address the QUESTION? Reply with ONLY:
{{"score": <0-1 float>}}"""
    return _llm_score(prompt, "You are a strict answer-relevance judge.")


def context_precision_score(query: str, expanded_contexts: list[ExpandedContext]) -> float:
    if not expanded_contexts:
        return 0.0
    chunks_preview = "\n".join(f"{i+1}. {ec.core.text[:150]}" for i, ec in enumerate(expanded_contexts))
    prompt = f"""QUESTION:\n{query}\n\nRETRIEVED CHUNKS:\n{chunks_preview}\n\nWhat fraction of
these chunks are actually relevant to answering the question? Reply with ONLY:
{{"score": <0-1 float>}}"""
    return _llm_score(prompt, "You are a strict context-precision judge.")


def answer_correctness_score(question: str, generated_answer: str, expected_answer: str) -> float:
    """Score whether a generated answer conveys the same core answer as a known-correct one.

    Purpose:
        Ground-truth comparison for offline evaluation, as opposed to
        faithfulness/context_precision which only check consistency with
        retrieved evidence, not correctness against a human-supplied answer.
    Arguments:
        question: The original question.
        generated_answer: The RAG pipeline's generated answer.
        expected_answer: A short, human-supplied correct answer (ground truth).
    Returns:
        Correctness score in [0, 1] (1 = fully correct/equivalent; partial
        credit for partially correct or incomplete answers).
    Exceptions:
        None (returns 0.0 on judge failure).
    """
    prompt = f"""QUESTION: {question}

EXPECTED (correct) ANSWER: {expected_answer}

GENERATED ANSWER: {generated_answer}

Does the GENERATED ANSWER convey the same correct information as the EXPECTED
ANSWER, even if worded differently or more elaborately? Reply with ONLY:
{{"score": <0-1 float, 1.0 = fully correct and equivalent, 0.5 = partially
  correct, 0.0 = incorrect or contradicts the expected answer>}}"""
    return _llm_score(prompt, "You are a strict answer-correctness judge comparing against a known correct answer.")


# --- Latency ---------------------------------------------------------------------
def measure_latency(fn):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            wrapper.last_latency_seconds = time.time() - start
            _log.info("%s took %.3fs", fn.__name__, wrapper.last_latency_seconds)
    wrapper.last_latency_seconds = 0.0
    return wrapper

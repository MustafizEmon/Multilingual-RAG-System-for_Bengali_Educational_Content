from __future__ import annotations
import time
from dataclasses import dataclass

from app.core.config import SETTINGS, get_logger
from app.modules.answer_generation import generate_answer
from app.modules.hybrid_retrieval import RetrievedCandidate, hybrid_retrieve
from app.modules.parent_context import ExpandedContext, expand_with_parent_context
from app.modules.prompt import build_prompt
from app.modules.query_understanding import (
    QueryClassification,
    QueryExpansion,
    analyze_query,
    detect_query_language,
    generate_hyde_passage,
    normalize_query,
)
from app.modules.reranker import RERANKER
from app.modules.session import SESSION_STORE, SessionMode, resolve_session_context
from app.modules.validation import ValidationResult, validate_answer

_log = get_logger("pipeline")


@dataclass
class RAGResponse:
    """Everything about one pipeline run, useful for the API layer, evaluation, and debugging."""
    query: str
    session_id: str
    session_mode: SessionMode
    language: str
    classification: QueryClassification
    expansion: QueryExpansion
    hyde_used: bool
    candidates: list[RetrievedCandidate]
    expanded_contexts: list[ExpandedContext]
    answer: str
    validation: ValidationResult
    llm_validated: bool     # True only if validate_answer actually ran (an LLM call)
    retried: bool
    latency_seconds: float
    history: list[list[str]]   # prior [question, answer] pairs, BEFORE this turn


class RAGPipeline:

    def __init__(self, fused_candidate_k: int = SETTINGS.fused_candidate_k):
        self.fused_candidate_k = fused_candidate_k

    def _retrieve_and_answer(
        self,
        norm_query: str,
        classification: QueryClassification,
        expansion: QueryExpansion,
        hyde_passage: str | None,
        session_context: str | None,
        candidate_k: int,
        rerank_k: int,
        neighbor_k: int,
    ) -> tuple[list[RetrievedCandidate], list[ExpandedContext], str]:
        """One retrieve -> rerank -> expand -> prompt -> generate pass."""
        candidates = hybrid_retrieve(norm_query, expansion, hyde_passage, top_k=candidate_k)
        reranked = RERANKER.rerank(norm_query, candidates, top_k=rerank_k)
        expanded = expand_with_parent_context(reranked, max_neighbors=neighbor_k)
        prompt_text = build_prompt(norm_query, classification, expanded, session_context)
        answer = generate_answer(prompt_text)
        return candidates, expanded, answer

    def run(
        self,
        raw_query: str,
        session_id: str,
        session_mode: SessionMode = SessionMode.FRESH,
    ) -> RAGResponse:

        start = time.time()

        # Snapshot history BEFORE this turn is added, so the response's
        # `history` field reflects "everything said before this question" —
        # matching what a client needs to render a running conversation.
        history_before = (
            SESSION_STORE.get_history_pairs(session_id)
            if session_mode == SessionMode.SESSION
            else []
        )

        language = detect_query_language(raw_query)
        norm_query = normalize_query(raw_query)
        classification, expansion = analyze_query(norm_query)   # 1 call, not 2
        is_high_risk = classification.is_high_risk

        hyde_passage = generate_hyde_passage(norm_query, classification)  # None (no call) unless high-risk
        session_context = resolve_session_context(session_id, session_mode)

        rerank_k = SETTINGS.rerank_top_k_complex if is_high_risk else SETTINGS.rerank_top_k_simple
        neighbor_k = SETTINGS.max_neighbors_complex if is_high_risk else SETTINGS.max_neighbors_simple

        candidates, expanded, answer = self._retrieve_and_answer(
            norm_query, classification, expansion, hyde_passage, session_context,
            candidate_k=self.fused_candidate_k, rerank_k=rerank_k, neighbor_k=neighbor_k,
        )

        llm_validated = False
        if is_high_risk:
            validation = validate_answer(norm_query, answer, expanded)
            llm_validated = True
        elif not expanded:
            # Safety net that costs nothing: retrieval found no evidence at all,
            # regardless of question type — that's always worth a retry.
            validation = ValidationResult(
                sufficient_evidence=False, context_relevant=False,
                has_contradictions=True, confidence=0.0,
                notes="No evidence was retrieved for this query.",
            )
        else:
            validation = ValidationResult.skipped(
                f"Low-risk question type ('{classification.question_type}'); "
                "validation skipped to conserve API usage."
            )

        retried = False
        if not validation.passed:
            _log.info("Validation failed (%s) — retrying once with widened, full-depth retrieval",
                       validation.notes)
            retried = True
            candidates, expanded, answer = self._retrieve_and_answer(
                norm_query, classification, expansion, hyde_passage, session_context,
                candidate_k=self.fused_candidate_k + 10,
                rerank_k=SETTINGS.rerank_top_k_complex,
                neighbor_k=SETTINGS.max_neighbors_complex,
            )
            if is_high_risk:
                validation = validate_answer(norm_query, answer, expanded)
                llm_validated = True
            else:
                validation = ValidationResult.skipped(
                    "Answer regenerated with widened retrieval after an empty first "
                    "attempt; not re-validated to conserve API usage."
                )

        if session_mode == SessionMode.SESSION:
            SESSION_STORE.add_turn(session_id, raw_query, answer)

        elapsed = time.time() - start
        _log.info("Pipeline run complete in %.2fs (session=%s, high_risk=%s, retried=%s, "
                   "llm_validated=%s, confidence=%.2f)",
                   elapsed, session_id, is_high_risk, retried, llm_validated, validation.confidence)

        return RAGResponse(
            query=raw_query,
            session_id=session_id,
            session_mode=session_mode,
            language=language,
            classification=classification,
            expansion=expansion,
            hyde_used=hyde_passage is not None,
            candidates=candidates,
            expanded_contexts=expanded,
            answer=answer,
            validation=validation,
            llm_validated=llm_validated,
            retried=retried,
            latency_seconds=elapsed,
            history=history_before,
        )


PIPELINE = RAGPipeline()

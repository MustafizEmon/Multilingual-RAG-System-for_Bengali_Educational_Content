from __future__ import annotations
from fastapi import APIRouter, HTTPException

from app.api.deps import new_session_id
from app.core.config import get_logger
from app.modules.pipeline import PIPELINE
from app.modules.session import SessionMode
from app.schemas.query import QueryRequest, QueryResponse, SourceExcerpt, ValidationInfo

_log = get_logger("api.query")

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question against the ingested documents",
)
async def query(request: QueryRequest) -> QueryResponse:

    session_id = request.session_id or new_session_id()
    session_mode = SessionMode(request.session_mode.value)

    try:
        response = PIPELINE.run(request.query, session_id=session_id, session_mode=session_mode)
    except RuntimeError as exc:
        # Groq/API-layer failures (rate limits, auth, transient outages) after
        # the pipeline's own retries are exhausted — a real 503, not a bug.
        _log.error("Pipeline run failed for session %s: %s", session_id, exc)
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {exc}") from exc
    except Exception as exc:
        _log.exception("Unexpected pipeline error for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    sources = [
        SourceExcerpt(
            document_name=ec.core.metadata.get("document_name", "?"),
            page=int(ec.core.metadata.get("page", 0) or 0),
            chapter=ec.core.metadata.get("chapter") or None,
            heading=ec.core.metadata.get("heading") or None,
            excerpt=(ec.core.text[:300] + ("…" if len(ec.core.text) > 300 else "")),
            neighbor_count=len(ec.neighbors),
        )
        for ec in response.expanded_contexts
    ]

    return QueryResponse(
        question=response.query,
        answer=response.answer,
        session_id=response.session_id,
        session_mode=response.session_mode.value,
        language=response.language,
        question_type=response.classification.question_type,
        is_high_risk=response.classification.is_high_risk,
        hyde_used=response.hyde_used,
        sources=sources,
        validation=ValidationInfo(
            sufficient_evidence=response.validation.sufficient_evidence,
            context_relevant=response.validation.context_relevant,
            has_contradictions=response.validation.has_contradictions,
            confidence=response.validation.confidence,
            notes=response.validation.notes,
            llm_validated=response.llm_validated,
        ),
        retried=response.retried,
        latency_seconds=response.latency_seconds,
        history=response.history,
    )

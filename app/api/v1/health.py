from __future__ import annotations

from fastapi import APIRouter

from app.core.config import SETTINGS
from app.modules.document_registry import DOCUMENT_REGISTRY
from app.modules.embeddings import EMBEDDER
from app.modules.reranker import RERANKER
from app.modules.session import SESSION_STORE
from app.schemas.health import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse, summary="Service health/readiness snapshot")
async def health() -> HealthResponse:

    return HealthResponse(
        status="ok",
        embedding_model=SETTINGS.embedding_model_name,
        embedding_model_loaded=EMBEDDER._model is not None,
        reranker_model=SETTINGS.reranker_model_name,
        reranker_model_loaded=RERANKER._model is not None,
        groq_model=SETTINGS.groq_model,
        indexed_documents=len(DOCUMENT_REGISTRY.list_all()),
        active_sessions=len(SESSION_STORE),
    )

from __future__ import annotations
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Service health/readiness snapshot."""

    status: str
    embedding_model: str
    embedding_model_loaded: bool
    reranker_model: str
    reranker_model_loaded: bool
    groq_model: str
    indexed_documents: int
    active_sessions: int

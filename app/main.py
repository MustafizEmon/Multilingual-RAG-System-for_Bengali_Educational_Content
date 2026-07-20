from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import SETTINGS, get_logger
from app.modules.embeddings import EMBEDDER
from app.modules.reranker import RERANKER

_log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("Starting up — preloading embedding and reranker models...")
    EMBEDDER.load()
    RERANKER.load()
    _log.info("Models ready. Groq model: %s", SETTINGS.groq_model)
    yield
    _log.info("Shutting down.")


app = FastAPI(
    title="Multilingual Educational RAG API — Bangla + English",
    description=(
        "Production-quality Retrieval-Augmented Generation service for Bangla "
        "educational content (literature, history, science, sociology, and general "
        "education), with English/mixed query support. Upload any Bangla PDF (any "
        "subject or genre) via /api/v1/documents/upload, then ask questions via "
        "/api/v1/query. See the README for full architecture and setup."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Note: CORS is wide open by default (configurable via API_CORS_ORIGINS) so a future
# React (or any other) frontend can call this API directly during development.
# Tighten this to your real frontend origin(s) before deploying publicly.
_origins = (
    ["*"] if SETTINGS.api_cors_origins.strip() == "*"
    else [o.strip() for o in SETTINGS.api_cors_origins.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Root"], summary="Service info")
async def root() -> dict:
    """Basic service pointer — see /docs for the full interactive API reference."""
    return {
        "service": "Multilingual Educational RAG API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }

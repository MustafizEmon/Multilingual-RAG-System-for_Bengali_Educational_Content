from __future__ import annotations
from fastapi import APIRouter
from app.api.v1 import documents, health, query, sessions

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(query.router)
api_router.include_router(sessions.router)

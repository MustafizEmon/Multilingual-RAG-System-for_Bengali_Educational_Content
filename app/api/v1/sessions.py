from __future__ import annotations
from fastapi import APIRouter

from app.core.config import get_logger
from app.modules.session import SESSION_STORE
from app.schemas.sessions import SessionClearResponse, SessionHistoryResponse

_log = get_logger("api.sessions")
router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.get(
    "/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="View a conversation's history",
)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    pairs = SESSION_STORE.get_history_pairs(session_id)
    return SessionHistoryResponse(session_id=session_id, history=pairs, turn_count=len(pairs))


@router.delete(
    "/{session_id}",
    response_model=SessionClearResponse,
    summary="Clear a conversation's history",
)
async def clear_session(session_id: str) -> SessionClearResponse:
    """Discard all held turns for a session_id, starting that conversation fresh."""
    cleared = SESSION_STORE.clear(session_id)
    return SessionClearResponse(session_id=session_id, cleared=cleared)

from __future__ import annotations
from pydantic import BaseModel, Field

class SessionHistoryResponse(BaseModel):
    """Response for GET /api/v1/sessions/{session_id}/history."""
    session_id: str
    history: list[list[str]] = Field(
        description='List of [question, answer] pairs, oldest first, up to the last 5 turns.'
    )
    turn_count: int


class SessionClearResponse(BaseModel):
    """Response for DELETE /api/v1/sessions/{session_id}."""
    session_id: str
    cleared: bool = Field(description="True if the session existed and was cleared.")

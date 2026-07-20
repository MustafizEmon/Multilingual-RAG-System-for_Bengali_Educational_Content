from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field

class SessionModeEnum(str, Enum):
    """Mirrors app.modules.session.SessionMode for the API boundary."""
    fresh = "fresh"
    session = "session"

class QueryRequest(BaseModel):
    """Incoming question for POST /api/v1/query."""
    query: str = Field(
        ...,
        min_length=1,
        description="The question, in Bangla, English, or mixed. The knowledge base "
                    "is Bangla educational content, but you may ask in any language.",
        examples=["‘লা প্যারুর’ গল্পটির রচয়িতা কে?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Conversation id. Omit to start a new conversation — the server "
                    "generates one and returns it in the response so you can reuse it "
                    "on subsequent calls to keep the same history.",
    )
    session_mode: SessionModeEnum = Field(
        default=SessionModeEnum.fresh,
        description="'fresh' ignores any prior turns for this session_id. 'session' "
                    "includes up to the last 5 Q&A turns as conversation context.",
    )


class SourceExcerpt(BaseModel):
    """One piece of evidence the answer was grounded in."""
    document_name: str
    page: int
    chapter: str | None = None
    heading: str | None = None
    excerpt: str = Field(description="Truncated preview of the chunk's text (first ~300 characters).")
    neighbor_count: int = Field(description="Number of adjacent chunks also included as context.")


class ValidationInfo(BaseModel):
    """Grounding-check outcome attached to the answer."""
    sufficient_evidence: bool
    context_relevant: bool
    has_contradictions: bool
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str
    llm_validated: bool = Field(
        description="True only if an LLM validation call actually ran. False means "
                    "validation was skipped for this (low-risk) question type to "
                    "conserve API usage — 'passed' below is a default, not a real check."
    )


class QueryResponse(BaseModel):
    """Full response for POST /api/v1/query — everything the pipeline knows about this answer."""
    question: str
    answer: str
    session_id: str
    session_mode: SessionModeEnum

    language: str = Field(description="Detected query language: 'bn', 'en', or 'mixed'.")
    question_type: str = Field(description="Classified question type, e.g. 'definition', 'reasoning'.")
    is_high_risk: bool = Field(
        description="Whether this question was flagged as needing the full (costlier) "
                    "pipeline path: HyDE, wider context, and LLM-judged validation."
    )
    hyde_used: bool

    sources: list[SourceExcerpt] = Field(description="Evidence excerpts the answer was grounded in.")
    validation: ValidationInfo
    retried: bool = Field(description="Whether one corrective retry (widened retrieval) occurred.")
    latency_seconds: float

    history: list[list[str]] = Field(
        description="Prior [question, answer] pairs for this session, BEFORE this turn "
                    "(empty in 'fresh' mode, or for a brand-new session_id). Example: "
                    '[["‘লা প্যারুর’ গল্পটির রচয়িতা কে?", "মপাসাঁ।"]]',
    )

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    """Single-turn QA request."""
    question: str = Field(..., description="User question")
    
class ChatRequest(BaseModel):
    """Multi-turn conversation request."""
    question: str = Field(..., description="User question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")

class Citation(BaseModel):
    """Source citation information."""
    chunk_id: str
    page: int = 0
    source: str = ""
    section: str = ""
    text_preview: Optional[str] = None

class AnswerResponse(BaseModel):
    """Response with answer and metadata."""
    answer: str
    confidence: float
    sources: List[Citation] = []
    session_id: Optional[str] = None
    model_used: Optional[str] = None
    generation_time: float = 0.0
    tokens_used: int = 0
    
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime
    vector_store_status: str
    llm_status: str
    memory_status: str

class MetricsResponse(BaseModel):
    """Metrics response."""
    total_queries: int
    average_latency: float
    retrieval_stats: Dict[str, Any]
    memory_usage: Dict[str, Any]
    system_stats: Dict[str, Any]

class DocumentIngestRequest(BaseModel):
    """Document ingestion request."""
    pdf_path: str = Field(..., description="Path to PDF file")
    story_extraction: bool = Field(True, description="Extract story content")
    chunking_strategy: str = Field("semantic", description="Chunking strategy")
    
class DocumentIngestResponse(BaseModel):
    """Document ingestion response."""
    status: str
    documents_added: int
    chunks_created: int
    metadata: Dict[str, Any]
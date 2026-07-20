from __future__ import annotations
from pydantic import BaseModel, Field

class DocumentUploadResponse(BaseModel):
    """Response for POST /api/v1/documents/upload."""
    document_name: str
    subject: str = Field(description="Auto-detected subject/domain, e.g. 'History', 'Bangla Literature'.")
    n_pages: int
    n_chunks: int
    message: str

class DocumentInfo(BaseModel):
    """One entry in the GET /api/v1/documents listing."""
    document_name: str
    subject: str
    n_pages: int
    n_chunks: int

class DocumentListResponse(BaseModel):
    """Response for GET /api/v1/documents."""
    documents: list[DocumentInfo]
    total: int

class RawTextResponse(BaseModel):
    """Response for GET /api/v1/documents/{document_name}/raw-text."""
    document_name: str
    raw_text: str = Field(description="Every page's cleaned OCR text, concatenated, "
                                        "exactly as handed to the chunker.")

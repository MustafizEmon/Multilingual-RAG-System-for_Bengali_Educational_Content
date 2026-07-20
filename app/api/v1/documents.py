from __future__ import annotations
from fastapi import APIRouter, HTTPException, UploadFile

from app.api.deps import save_upload_file
from app.core.config import SETTINGS, get_logger
from app.modules.document_registry import DOCUMENT_REGISTRY
from app.modules.ingestion import ingest_document
from app.schemas.documents import (
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    RawTextResponse,
)

_log = get_logger("api.documents")
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload and ingest a PDF (any subject, any type — scanned or text)",
)
async def upload_document(file: UploadFile) -> DocumentUploadResponse:

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")
    saved_path = save_upload_file(file)
    try:
        result = ingest_document(saved_path)
    except ValueError as exc:
        # No text could be extracted — a legitimate 422 (unprocessable input),
        # not a server error.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("Ingestion failed for %s", saved_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return DocumentUploadResponse(
        document_name=result.document_name,
        subject=result.subject,
        n_pages=result.n_pages,
        n_chunks=result.n_chunks,
        message=f"Ingested '{result.document_name}' successfully "
                f"({result.n_pages} pages, {result.n_chunks} chunks, subject: {result.subject}).",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List every document ingested so far",
)
async def list_documents() -> DocumentListResponse:
    """Return every document currently indexed, with its detected subject and chunk count."""
    records = DOCUMENT_REGISTRY.list_all()
    return DocumentListResponse(
        documents=[
            DocumentInfo(document_name=r.document_name, subject=r.subject,
                         n_pages=r.n_pages, n_chunks=r.n_chunks)
            for r in records
        ],
        total=len(records),
    )


@router.get(
    "/{document_name}/raw-text",
    response_model=RawTextResponse,
    summary="View a document's raw pre-chunking text dump",
)
async def get_raw_text(document_name: str) -> RawTextResponse:
    path = SETTINGS.raw_text_dump_dir / f"{document_name}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No raw text dump found for '{document_name}'.")
    return RawTextResponse(document_name=document_name, raw_text=path.read_text(encoding="utf-8"))

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_logger
from app.modules.chunker import build_chunks, structure_page_text
from app.modules.document_registry import DOCUMENT_REGISTRY, DocumentRecord
from app.modules.llm import call_llm_json
from app.modules.ocr import ingest_pdf
from app.modules.retriever import DENSE_STORE, SPARSE_STORE
from app.modules.utils import free_memory

_log = get_logger("ingestion")


def detect_document_subject(sample_text: str) -> str:
    if not sample_text.strip():
        return "General Education"

    result = call_llm_json(
        f"""Read this excerpt from an educational document (Bangla or English) and
identify its subject/domain in 1-4 words (e.g. "History", "Bangla Literature",
"Poetry", "Science", "Sociology", "Mathematics", "General Education").

Excerpt:
{sample_text[:2000]}

Reply with ONLY: {{"subject": "<short subject label>"}}""",
        system_prompt="You are a precise educational content classifier.",
    )
    return result.get("subject") or "General Education"


@dataclass
class IngestionResult:
    """Outcome of ingesting one document, returned to the API caller."""
    document_name: str
    subject: str
    n_pages: int
    n_chunks: int


def ingest_document(pdf_path: Path, document_name: str | None = None) -> IngestionResult:
    document_name = document_name or pdf_path.stem
    _log.info("Starting ingestion for '%s' (%s)", document_name, pdf_path)

    page_records = ingest_pdf(pdf_path, document_name=document_name)
    if not page_records:
        raise ValueError(f"No text could be extracted from '{document_name}'.")

    doc_paragraphs = []
    for rec in page_records:
        doc_paragraphs.extend(structure_page_text(rec.document_name, rec.page_number, rec.text))

    sample = " ".join(p.text for p in doc_paragraphs[:5])
    subject = detect_document_subject(sample)
    _log.info("Detected subject for '%s': %s", document_name, subject)

    doc_chunks = build_chunks(document_name=document_name, paragraphs=doc_paragraphs, subject=subject)

    DENSE_STORE.add_chunks(doc_chunks)
    SPARSE_STORE.add_chunks(doc_chunks)

    result = IngestionResult(
        document_name=document_name, subject=subject,
        n_pages=len(page_records), n_chunks=len(doc_chunks),
    )
    DOCUMENT_REGISTRY.register(DocumentRecord(
        document_name=result.document_name, subject=result.subject,
        n_pages=result.n_pages, n_chunks=result.n_chunks,
    ))

    free_memory(page_records, doc_paragraphs, doc_chunks, sample)
    _log.info("Ingestion complete for '%s': %d pages, %d chunks", document_name,
               result.n_pages, result.n_chunks)
    return result

from __future__ import annotations

from dataclasses import dataclass, field
import re
import uuid

from app.core.config import SETTINGS, get_logger
from app.modules.utils import count_tokens, detect_script


@dataclass
class Paragraph:
    """A single reconstructed paragraph within a page, with structural hints."""
    text: str
    page_number: int
    chapter: str | None = None
    heading: str | None = None


@dataclass
class ChunkMetadata:
    """Full metadata attached to every chunk, per the project's required schema."""
    chunk_id: str
    document_name: str
    page: int
    chapter: str | None
    heading: str | None
    subject: str | None
    language: str                     # "bn" | "en" | "mixed"
    prev_chunk_id: str | None
    next_chunk_id: str | None
    chunk_summary: str | None = None
    keywords: list[str] = field(default_factory=list)
    named_entities: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """A retrieval-unit of text plus its full metadata."""
    text: str
    metadata: ChunkMetadata


# --- Lightweight structural heuristics ------------------------------------------
# Headings in OCR'd Bangla educational books are typically short, standalone
# lines without terminal punctuation, often numbered ("অধ্যায় ১", "চতুর্থ অধ্যায়")
# or in Title Case for English material. These heuristics avoid needing a
# trained layout model while still capturing most chapter/heading boundaries.
_CHAPTER_RE = re.compile(
    r"^(chapter|অধ্যায়|পরিচ্ছেদ)\s*[:\-]?\s*([0-9০-৯]+|[ivxIVX]+)?", re.IGNORECASE
)
_HEADING_MAX_WORDS = 8
_HEADING_MAX_CHARS = 60


def _looks_like_heading(line: str) -> bool:
    """Heuristically decide whether a standalone line is a heading, not prose."""
    stripped = line.strip()
    if not stripped or len(stripped) > _HEADING_MAX_CHARS:
        return False
    if stripped.endswith(("।", ".", ",")):
        return False
    return len(stripped.split()) <= _HEADING_MAX_WORDS


def structure_page_text(document_name: str, page_number: int, page_text: str) -> list[Paragraph]:

    current_chapter: str | None = None
    current_heading: str | None = None
    paragraphs: list[Paragraph] = []

    blocks = page_text.split("\n\n")
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        first_line = block.splitlines()[0].strip()
        chapter_match = _CHAPTER_RE.match(first_line)
        if chapter_match:
            current_chapter = first_line
            current_heading = None
            remainder = block[len(first_line):].strip()
            if not remainder:
                continue
            block = remainder

        if _looks_like_heading(block):
            current_heading = block
            continue

        paragraphs.append(Paragraph(
            text=block,
            page_number=page_number,
            chapter=current_chapter,
            heading=current_heading,
        ))

    return paragraphs


_log = get_logger("chunker")


def _split_oversized_paragraph(paragraph_text: str, max_tokens: int) -> list[str]:

    import re
    sentences = re.split(r"(?<=[।.!?])\s+", paragraph_text)
    pieces, current = [], ""
    for sent in sentences:
        candidate = f"{current} {sent}".strip()
        if count_tokens(candidate) > max_tokens and current:
            pieces.append(current)
            current = sent
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _overlap_text(text: str, overlap_ratio: float) -> str:
    words = text.split()
    n_overlap = max(1, int(len(words) * overlap_ratio))
    return " ".join(words[-n_overlap:])


def build_chunks(
    document_name: str,
    paragraphs: list[Paragraph],
    subject: str | None = None,
    min_tokens: int = SETTINGS.min_chunk_tokens,
    max_tokens: int = SETTINGS.max_chunk_tokens,
    overlap_ratio: float = SETTINGS.chunk_overlap_ratio,
) -> list[Chunk]:

    chunks: list[Chunk] = []
    buffer_text = ""
    buffer_page = None
    buffer_chapter = None
    buffer_heading = None

    def _flush(next_seed: str = "") -> None:
        nonlocal buffer_text, buffer_page, buffer_chapter, buffer_heading
        if not buffer_text.strip():
            return
        language = detect_script(buffer_text)
        meta = ChunkMetadata(
            chunk_id=str(uuid.uuid4()),
            document_name=document_name,
            page=buffer_page or 0,
            chapter=buffer_chapter,
            heading=buffer_heading,
            subject=subject,
            language=language,
            prev_chunk_id=chunks[-1].metadata.chunk_id if chunks else None,
            next_chunk_id=None,  # backfilled after all chunks are built
        )
        chunks.append(Chunk(text=buffer_text.strip(), metadata=meta))
        buffer_text = next_seed

    for para in paragraphs:
        pieces = (
            _split_oversized_paragraph(para.text, max_tokens)
            if count_tokens(para.text) > max_tokens
            else [para.text]
        )
        for piece in pieces:
            candidate = f"{buffer_text}\n\n{piece}".strip() if buffer_text else piece
            candidate_tokens = count_tokens(candidate)

            if candidate_tokens <= max_tokens:
                buffer_text = candidate
                buffer_page = buffer_page or para.page_number
                buffer_chapter = buffer_chapter or para.chapter
                buffer_heading = buffer_heading or para.heading
                if candidate_tokens >= min_tokens:
                    # Close now if the *next* paragraph would push us over budget;
                    # otherwise keep accumulating. We peek by closing eagerly once
                    # min_tokens is reached and let the overlap carry continuity.
                    overlap_seed = _overlap_text(buffer_text, overlap_ratio)
                    _flush(next_seed=overlap_seed)
                    buffer_page = para.page_number
                    buffer_chapter = para.chapter
                    buffer_heading = para.heading
            else:
                # Piece doesn't fit in the current buffer: close current buffer,
                # start a new one seeded with overlap + this piece.
                overlap_seed = _overlap_text(buffer_text, overlap_ratio) if buffer_text else ""
                _flush(next_seed=overlap_seed)
                buffer_text = f"{buffer_text}\n\n{piece}".strip() if buffer_text else piece
                buffer_page = para.page_number
                buffer_chapter = para.chapter
                buffer_heading = para.heading

    _flush()  # flush remainder

    # Backfill next_chunk_id links now that the full sequence is known.
    for i in range(len(chunks) - 1):
        chunks[i].metadata.next_chunk_id = chunks[i + 1].metadata.chunk_id

    _log.info("Built %d chunks for document '%s'", len(chunks), document_name)
    return chunks

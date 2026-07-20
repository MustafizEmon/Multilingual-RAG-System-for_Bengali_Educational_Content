from __future__ import annotations

from functools import lru_cache
import gc
import re
import unicodedata
from app.core.config import get_logger

_log = get_logger("utils")

# --- Bangla punctuation / normalization tables --------------------------------------
# Bangla text scraped from OCR frequently contains ASCII punctuation stand-ins and
# inconsistent Unicode normalization forms (NFC vs NFD/decomposed matras).
_BANGLA_PUNCT_MAP = {
    "|": "।",     # ASCII pipe often mis-OCR'd in place of Bangla full stop (dari)
    "..": "।",
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
}

_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")  # word broken across a line by a hyphen


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_bangla_punctuation(text: str) -> str:
    for bad, good in _BANGLA_PUNCT_MAP.items():
        text = text.replace(bad, good)
    return text


def fix_hyphenation(text: str) -> str:
    return _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


def reconstruct_paragraphs(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    rebuilt = []
    for para in paragraphs:
        # Join soft-wrapped lines inside a paragraph with a single space.
        joined = " ".join(line.strip() for line in para.splitlines() if line.strip())
        if joined:
            rebuilt.append(joined)
    return "\n\n".join(rebuilt)


def clean_page_text(raw_text: str) -> str:
    text = normalize_unicode(raw_text)
    text = fix_hyphenation(text)
    text = normalize_bangla_punctuation(text)
    text = reconstruct_paragraphs(text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# --- Token counting -------------------------------------------------------------
# avoided pulling in the full bge-m3 tokenizer for a *count-only* operation
# (expensive to load repeatedly). tiktoken's cl100k is not exact for Bangla but is
# fast, dependency-light, and consistent enough to budget chunk sizes reliably.
@lru_cache(maxsize=1)
def _get_tiktoken_encoder():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    try:
        enc = _get_tiktoken_encoder()
        return len(enc.encode(text))
    except Exception:
        _log.warning("tiktoken unavailable, falling back to whitespace token count")
        return len(text.split())


def free_memory(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()


def detect_script(text: str) -> str:
    bangla_chars = sum(1 for ch in text if "\u0980" <= ch <= "\u09FF")
    latin_chars = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = bangla_chars + latin_chars
    if total == 0:
        return "unknown"
    bangla_ratio = bangla_chars / total
    if bangla_ratio > 0.85:
        return "bn"
    if bangla_ratio < 0.15:
        return "en"
    return "mixed"

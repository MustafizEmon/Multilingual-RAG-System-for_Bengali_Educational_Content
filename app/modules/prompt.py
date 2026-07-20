from __future__ import annotations

from app.core.config import SETTINGS
from app.modules.parent_context import ExpandedContext
from app.modules.query_understanding import QueryClassification

SYSTEM_PROMPT = """You are a grounded and specific educational assistant for Bangla and \
English learning materials (literature, history, science, sociology, and general \
education). Your job is to answer user questions based on that context and by analysing \
the whole chunk, even if a question requires connecting multiple excerpts \
(multi-hop reasoning), you reason step by step and cite which excerpt(s) each \
part of your answer relies on. You must use **semantic reasoning**, draw \
**logical inferences**, and identify **relationships or implications** within \
the content to construct an answer. Connect all the dots and plots between the context chunks.\
Answer in the same language as the question (Bangla, English, or mixed, matching \
the user). Make small answer."""


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def _format_context_block(expanded: ExpandedContext, index: int) -> str:
    """Render one expanded-context item (core chunk + neighbors) as prompt text."""
    meta = expanded.core.metadata
    header_bits = [
        f"Excerpt {index}",
        f"document={meta.get('document_name', '?')}",
        f"page={meta.get('page', '?')}",
    ]
    if meta.get("chapter"):
        header_bits.append(f"chapter={meta['chapter']}")
    if meta.get("heading"):
        header_bits.append(f"heading={meta['heading']}")
    header = " | ".join(header_bits)

    core_text = _truncate_words(expanded.core.text, SETTINGS.excerpt_max_words)
    body = [f"[{header}]\n{core_text}"]
    for n in expanded.neighbors:
        neighbor_text = _truncate_words(n["text"], SETTINGS.neighbor_max_words)
        body.append(f"[{header} | {n['relation']}-context]\n{neighbor_text}")
    return "\n\n".join(body)


def build_prompt(
    query: str,
    classification: QueryClassification,
    expanded_contexts: list[ExpandedContext],
    session_context: str | None = None,
) -> str:

    evidence_blocks = "\n\n---\n\n".join(
        _format_context_block(ec, i + 1) for i, ec in enumerate(expanded_contexts)
    )

    style_hint = {
        "mcq": "Answer as a single best option with a one-line justification.",
        "essay": "Write a well-organized multi-paragraph answer.",
        "comparison": "Structure the answer around clear point-by-point comparisons.",
        "timeline": "Present events in chronological order.",
    }.get(classification.question_type, "Answer directly and concisely, then elaborate if needed.")

    parts = []
    if session_context:
        parts.append(f"Recent conversation (for context only, evidence still governs facts):\n{session_context}")

    parts.append(f"Question type: {classification.question_type}. {style_hint}")
    parts.append(f"Question: {query}")
    parts.append(f"Evidence excerpts:\n\n{evidence_blocks}")
    parts.append(
        "Instructions: Base your answer strictly on the evidence excerpts above. "
        "If the excerpts do not contain enough information, state that explicitly "
        "rather than filling gaps with assumptions. For reasoning/inference/comparison "
        "questions, briefly note which excerpt(s) support each step of your reasoning."
    )
    return "\n\n".join(parts)

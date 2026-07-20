from __future__ import annotations

from dataclasses import dataclass, field

from langdetect import LangDetectException, detect

from app.core.config import SETTINGS, get_logger
from app.modules.llm import call_llm, call_llm_json
from app.modules.utils import clean_page_text, detect_script

_log = get_logger("query_understanding")

QUESTION_TYPES = [
    "fact", "definition", "summary", "comparison", "reasoning", "inference",
    "timeline", "author", "character", "theme", "science", "history",
    "literature", "mcq", "essay",
]

# Question types where multi-hop reasoning errors are likely enough to justify
# the extra API cost of HyDE generation and LLM-judged answer validation
# (Section 18 gates both of these on this same set). Everything else takes the
# cheap path: no HyDE, no LLM validation call, smaller reranked context.
HIGH_RISK_QUESTION_TYPES = {"reasoning", "inference", "comparison", "timeline"}


def detect_query_language(query: str) -> str:

    script_guess = detect_script(query)
    if script_guess in ("bn", "en"):
        return script_guess
    try:
        langdetect_guess = detect(query)
        if langdetect_guess == "bn":
            return "bn"
        if langdetect_guess == "en" and script_guess != "mixed":
            return "en"
    except LangDetectException:
        pass
    return "mixed" if script_guess == "mixed" else "en"


def normalize_query(query: str) -> str:
    return clean_page_text(query).strip()


@dataclass
class QueryClassification:
    """Result of classifying a query into an educational question type."""
    question_type: str
    confidence: float
    reasoning_required: bool

    @property
    def is_high_risk(self) -> bool:
        """Whether this query warrants the full (costlier) pipeline path."""
        return self.reasoning_required or self.question_type in HIGH_RISK_QUESTION_TYPES


@dataclass
class QueryExpansion:
    """Expanded representation of a query used to widen/aim retrieval."""
    expanded_query: str
    keywords: list[str] = field(default_factory=list)
    alternative_phrasings: list[str] = field(default_factory=list)
    subject_hints: list[str] = field(default_factory=list)
    named_entities: list[str] = field(default_factory=list)


def analyze_query(query: str) -> tuple[QueryClassification, QueryExpansion]:

    prompt = f"""Analyze the following educational question (Bangla, English, or mixed).
It may be about literature, stories, poetry, novels, history, sociology, science,
or general education.

Question: "{query}"

Reply with ONLY a JSON object with exactly these fields:
{{"question_type": "<one of: {", ".join(QUESTION_TYPES)}>",
  "confidence": <0-1 float>,
  "reasoning_required": <true/false — does answering need connecting multiple
   non-adjacent pieces of information>,
  "expanded_query": "<a fuller restatement covering likely relevant sub-topics>",
  "keywords": [<3-8 important keywords, in the question's original language(s)>],
  "alternative_phrasings": [<1-2 short alternative ways to ask the same question>],
  "subject_hints": [<likely subject areas, e.g. "Bangla Literature", "History">],
  "named_entities": [<any people, places, works, dates mentioned or implied>]}}"""

    result = call_llm_json(
        prompt,
        system_prompt="You are a precise multilingual educational question analyzer.",
        max_tokens=SETTINGS.groq_max_tokens_analysis,
    )

    question_type = result.get("question_type", "fact")
    if question_type not in QUESTION_TYPES:
        question_type = "fact"

    classification = QueryClassification(
        question_type=question_type,
        confidence=float(result.get("confidence", 0.0) or 0.0),
        reasoning_required=bool(result.get("reasoning_required", False)),
    )
    expansion = QueryExpansion(
        expanded_query=result.get("expanded_query") or query,
        keywords=result.get("keywords") or [],
        alternative_phrasings=result.get("alternative_phrasings") or [],
        subject_hints=result.get("subject_hints") or [],
        named_entities=result.get("named_entities") or [],
    )
    return classification, expansion


def generate_hyde_passage(query: str, classification: QueryClassification) -> str | None:
    if not classification.is_high_risk:
        return None

    prompt = f"""Write a short (3-5 sentence) hypothetical passage that WOULD plausibly
answer this educational question, as if excerpted from a textbook. This is only used
to improve semantic search — it does not need to be factually verified.

Question: "{query}\""""
    try:
        return call_llm(prompt, system_prompt="You write concise, plausible textbook-style passages.",
                         temperature=0.4, max_tokens=SETTINGS.groq_max_tokens_hyde)
    except Exception as exc:
        _log.warning("HyDE generation failed: %s", exc)
        return None

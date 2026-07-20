from __future__ import annotations

from app.core.config import SETTINGS, get_logger
from app.modules.llm import call_llm
from app.modules.prompt import SYSTEM_PROMPT

_log = get_logger("answer_generation")


def generate_answer(prompt: str, model: str = SETTINGS.groq_model) -> str:
    _log.info("Generating final answer (model=%s)", model)
    return call_llm(
        prompt,
        system_prompt=SYSTEM_PROMPT,
        model=model,
        temperature=SETTINGS.groq_temperature,
        max_tokens=SETTINGS.groq_max_tokens,
    )

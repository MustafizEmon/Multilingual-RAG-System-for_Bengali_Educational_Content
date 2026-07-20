from __future__ import annotations

import json
import time
from groq import Groq
from app.core.config import SETTINGS, get_logger, load_groq_api_key

_log = get_logger("llm")
_client: Groq | None = None


def _get_client() -> Groq:

    global _client
    if _client is None:
        api_key = load_groq_api_key()
        _client = Groq(api_key=api_key)
    return _client


def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    model: str = SETTINGS.groq_model,
    temperature: float = SETTINGS.groq_temperature,
    max_tokens: int = SETTINGS.groq_max_tokens,
    json_mode: bool = False,
    max_retries: int = 2,
) -> str:

    client = _get_client()

    if json_mode:
        # Groq's API rejects response_format={"type": "json_object"} unless the
        # word "json" literally appears somewhere in the messages. Guarantee
        # this centrally so every caller (classifier, expansion, validator,
        # subject detection, eval scorers, ...) is safe without having to
        # remember to word its own prompt a certain way.
        haystack = f"{system_prompt or ''} {prompt}".lower()
        if "json" not in haystack:
            json_instruction = "Respond with a valid JSON object only."
            system_prompt = f"{system_prompt} {json_instruction}" if system_prompt else json_instruction

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            # A json_validate_failed / "max completion tokens" error means the model's
            # output was truncated before it could close the JSON object -- retrying with
            # the same max_tokens will just fail again the same way, so grow the budget
            # instead of only backing off. Real transient errors still get the backoff.
            is_length_issue = json_mode and (
                "json_validate_failed" in str(exc) or "max completion tokens" in str(exc)
            )
            if is_length_issue:
                kwargs["max_tokens"] = int(kwargs["max_tokens"] * 1.5)
                _log.warning(
                    "Groq call failed (attempt %d/%d): truncated JSON output; "
                    "retrying with max_tokens=%d: %s",
                    attempt, max_retries + 1, kwargs["max_tokens"], exc,
                )
            else:
                _log.warning("Groq call failed (attempt %d/%d): %s", attempt, max_retries + 1, exc)
                time.sleep(min(2 ** attempt, 8))

    raise RuntimeError(f"Groq inference failed after {max_retries + 1} attempts: {last_exc}")


def call_llm_json(prompt: str, system_prompt: str | None = None, **kwargs) -> dict:
    raw = call_llm(prompt, system_prompt=system_prompt, json_mode=True, **kwargs)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        _log.error("LLM did not return valid JSON: %r", raw[:200])
        return {}

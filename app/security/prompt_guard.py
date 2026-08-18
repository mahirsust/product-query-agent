"""Prompt-injection detection backed by Meta's Prompt Guard 2 classifier.

Deliberately independent of the agent: this module knows only how to score a string, so it can be
reused to screen any untrusted input. `app/agent/middleware/guardrails.py` is the current caller.

The model is a lightweight (86M parameter) binary classifier rather than a chat model. It is
served over Groq's OpenAI-compatible endpoint but returns a bare probability as its message
content, so it is called directly with httpx instead of through a chat abstraction.
"""

import logging
from dataclasses import dataclass

import httpx

from app.config import settings

logger = logging.getLogger("app.security.prompt_guard")

_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"

_client: httpx.AsyncClient | None = None


@dataclass(frozen=True)
class GuardVerdict:
    """Outcome of screening one piece of text.

    `available` is False when the classifier could not be consulted at all (disabled, no API key,
    network failure). Callers must distinguish that from a confident "not an injection", because
    it means the input is simply unscreened.
    """

    is_injection: bool
    score: float
    available: bool

    @classmethod
    def unavailable(cls) -> "GuardVerdict":
        """A verdict meaning the input was never screened, not that it was found clean."""
        return cls(is_injection=False, score=0.0, available=False)


def _get_client() -> httpx.AsyncClient:
    """Lazily created, reused client so connections are pooled across requests."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=settings.prompt_guard_timeout_seconds)
    return _client


async def classify(text: str) -> GuardVerdict:
    """Score `text` for prompt-injection intent.

    Fails open: any error returns an unavailable verdict rather than raising, so a classifier
    outage degrades screening instead of taking the whole request down. The caller still has the
    local regex checks, and the failure is logged.
    """
    if not settings.prompt_guard_enabled or not settings.groq_api_key.get_secret_value():
        return GuardVerdict.unavailable()

    try:
        response = await _get_client().post(
            _COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key.get_secret_value()}"},
            json={
                "model": settings.prompt_guard_model,
                "messages": [{"role": "user", "content": text}],
            },
        )
        response.raise_for_status()
        score = float(response.json()["choices"][0]["message"]["content"])
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        logger.warning("prompt guard unavailable; input was not screened", exc_info=True)
        return GuardVerdict.unavailable()

    return GuardVerdict(
        is_injection=score >= settings.prompt_guard_threshold,
        score=score,
        available=True,
    )

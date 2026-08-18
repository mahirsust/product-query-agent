"""Redis cache of final answers, keyed per user."""

import hashlib
import re

from app.cache.redis_client import get_redis_client
from app.config import settings

_KEY_PREFIX = "response_cache"


def _normalize(question: str) -> str:
    """Fold case and whitespace so trivially different phrasings share a cache entry."""
    return re.sub(r"\s+", " ", question.strip().lower())


def _cache_key(question: str, user_id: int) -> str:
    """Build the cache key for a question asked by a specific user.

    Scoped per user because answers are personalized: the long-term memory middleware injects the
    asker's stored preferences into the prompt, so a shared key would serve one user's answer to
    another. The question is hashed so raw request text — which may contain PII and is not covered
    by the model-path redaction — never becomes a Redis key.
    """
    digest = hashlib.sha256(_normalize(question).encode("utf-8")).hexdigest()
    return f"{_KEY_PREFIX}:{user_id}:{digest}"


async def get(question: str, user_id: int) -> str | None:
    """Return the cached answer for this user's question, or None."""
    return await get_redis_client().get(_cache_key(question, user_id))


async def set(question: str, answer: str, user_id: int, ttl: int | None = None) -> None:
    """Cache an answer, defaulting to the configured TTL."""
    await get_redis_client().set(
        _cache_key(question, user_id),
        answer,
        ex=ttl if ttl is not None else settings.response_cache_ttl_seconds,
    )

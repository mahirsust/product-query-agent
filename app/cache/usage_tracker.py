"""Per-user usage accounting in Redis: daily budgets and per-minute burst guards.

Counters expire on their own, so no cleanup job is needed.
"""

import asyncio
from datetime import UTC, datetime
from enum import StrEnum

from app.cache.redis_client import get_redis_client
from app.config import settings

# Slightly over a day, so a read straddling the day boundary still sees the key expire.
_DAY_TTL_SECONDS = 26 * 60 * 60
_MINUTE_TTL_SECONDS = 60


class UsageCheckResult(StrEnum):
    """Why a request may proceed or not. The caller maps these to 429 and 402 respectively."""

    OK = "ok"
    RATE_LIMITED = "rate_limited"
    BUDGET_EXCEEDED = "budget_exceeded"


def _today() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


def _current_minute() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M")


def _daily_key(metric: str, user_id: int) -> str:
    """Redis key for a per-user daily counter; the date in the key is what makes it roll over."""
    return f"usage:{metric}:{user_id}:{_today()}"


def _minute_key(metric: str, user_id: int) -> str:
    """Redis key for a per-user per-minute counter, likewise self-expiring by name."""
    return f"rate:{metric}:{user_id}:{_current_minute()}"


async def check_and_reserve(user_id: int) -> UsageCheckResult:
    """Check a user against the rate and daily caps before a model call.

    Evaluates *prior* recorded usage: a call's own cost is unknown until the model responds, so
    capacity cannot be reserved atomically and concurrent requests may collectively overshoot.
    Acceptable for cost guardrails; not a billing boundary. Per-minute limits are checked first so
    a burst reports as rate-limited rather than out of budget.
    """
    client = get_redis_client()

    call_rate, token_rate = await asyncio.gather(
        client.get(_minute_key("calls", user_id)),
        client.get(_minute_key("tokens", user_id)),
    )
    if int(call_rate or 0) >= settings.max_llm_calls_per_minute_per_user:
        return UsageCheckResult.RATE_LIMITED
    if int(token_rate or 0) >= settings.max_tokens_per_minute_per_user:
        return UsageCheckResult.RATE_LIMITED

    daily_cost, daily_calls, daily_tokens = await asyncio.gather(
        client.get(_daily_key("cost", user_id)),
        client.get(_daily_key("calls", user_id)),
        client.get(_daily_key("tokens", user_id)),
    )
    if float(daily_cost or 0) >= settings.cost_budget_usd_per_user_per_day:
        return UsageCheckResult.BUDGET_EXCEEDED
    if int(daily_calls or 0) >= settings.max_llm_calls_per_user_per_day:
        return UsageCheckResult.BUDGET_EXCEEDED
    if int(daily_tokens or 0) >= settings.max_tokens_per_user_per_day:
        return UsageCheckResult.BUDGET_EXCEEDED

    return UsageCheckResult.OK


async def record_usage(user_id: int, cost: float, tokens: int, calls: int = 1) -> None:
    """Add one call's usage to every daily and per-minute counter."""
    client = get_redis_client()
    pipe = client.pipeline()

    pipe.incrbyfloat(_daily_key("cost", user_id), cost)
    pipe.expire(_daily_key("cost", user_id), _DAY_TTL_SECONDS)
    pipe.incrby(_daily_key("calls", user_id), calls)
    pipe.expire(_daily_key("calls", user_id), _DAY_TTL_SECONDS)
    pipe.incrby(_daily_key("tokens", user_id), tokens)
    pipe.expire(_daily_key("tokens", user_id), _DAY_TTL_SECONDS)

    pipe.incrby(_minute_key("calls", user_id), calls)
    pipe.expire(_minute_key("calls", user_id), _MINUTE_TTL_SECONDS)
    pipe.incrby(_minute_key("tokens", user_id), tokens)
    pipe.expire(_minute_key("tokens", user_id), _MINUTE_TTL_SECONDS)

    await pipe.execute()

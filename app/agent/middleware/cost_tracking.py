"""Records per-user token usage and estimated cost after each model call."""

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from app.cache import usage_tracker
from app.config import settings

_TOKENS_PER_PRICING_UNIT = 1_000_000


def _estimate_cost(usage: dict) -> float:
    """Convert token counts to dollars using the configured per-million rates.

    Rates come from `Settings` because they are a property of `llm_model`: hardcoding them here
    let them silently describe a different model than the one being called. Cost stays a derived,
    best-effort figure — the token counts are the caps' ground truth.
    """
    return (
        usage.get("input_tokens", 0) * settings.llm_input_price_per_million_usd
        + usage.get("output_tokens", 0) * settings.llm_output_price_per_million_usd
    ) / _TOKENS_PER_PRICING_UNIT


class CostTrackingMiddleware(AgentMiddleware):
    """Reports usage to the tracker once per model call.

    Only records; the pre-call budget check lives in the chat route, which must run before the
    agent is invoked at all. Does nothing without a `user_id` in context, as on the CLI path.
    """

    async def aafter_model(self, state, runtime: Runtime) -> None:
        """Record this call's usage against the caller's daily and per-minute counters."""
        if not runtime.context or runtime.context.user_id is None:
            return None

        usage = getattr(state["messages"][-1], "usage_metadata", None)
        if not usage:
            return None

        await usage_tracker.record_usage(
            user_id=runtime.context.user_id,
            cost=_estimate_cost(usage),
            tokens=usage.get("total_tokens", 0),
        )
        return None

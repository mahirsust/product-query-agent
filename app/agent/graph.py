"""Agent construction: model, tools, and the default middleware stack."""

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRetryMiddleware
from langchain_groq import ChatGroq
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from app.agent import (
    memory_tools,  # noqa: F401 -- imported for its @register_tool side effect
)
from app.agent.context import AgentContext
from app.agent.middleware.cost_tracking import CostTrackingMiddleware
from app.agent.middleware.groundedness import GroundednessMiddleware
from app.agent.middleware.guardrails import GuardrailsMiddleware
from app.agent.middleware.hitl import get_hitl_middleware
from app.agent.middleware.long_term_memory import long_term_memory_middleware
from app.agent.middleware.pii import PIIRedactionMiddleware
from app.agent.middleware.tool_dedupe import ToolCallDedupeMiddleware
from app.agent.prompts import MODEL_FAILURE_MESSAGE, SYSTEM_PROMPT
from app.agent.registry import get_all_tools
from app.config import settings

logger = logging.getLogger("app.agent.graph")


def _generic_model_failure(exc: Exception) -> str:
    """Format the user-facing message when a model call fails after all retries.

    The built-in formatter interpolates the raw exception, which for a provider rate-limit error
    exposes the account's organization id, model, tier and quota usage. The real error is logged
    for operators instead.
    """
    logger.error("model call failed after retries", exc_info=exc)
    return MODEL_FAILURE_MESSAGE


def get_default_middleware() -> list[AgentMiddleware]:
    """Build the standard middleware stack, in execution order.

    Every entry adds graph steps to each model call, so anything inert is omitted rather than
    included as a no-op.
    """
    middleware = [
        GuardrailsMiddleware(),
        PIIRedactionMiddleware(),
        # Retries recover non-deterministic model failures such as transient rate limits.
        ModelRetryMiddleware(max_retries=2, on_failure=_generic_model_failure),
        CostTrackingMiddleware(),
        GroundednessMiddleware(),
        # Guards against a model emitting the same call many times in one batch, which inflates
        # the next request past the provider's per-minute token ceiling.
        ToolCallDedupeMiddleware(),
        long_term_memory_middleware,
    ]
    hitl = get_hitl_middleware()
    if hitl is not None:
        middleware.append(hitl)
    return middleware


def build_agent(
    checkpointer: BaseCheckpointSaver | None,
    store: BaseStore | None = None,
    middleware: list[AgentMiddleware] | None = None,
):
    """Create the product-query agent.

    Args:
        checkpointer: Short-term, thread-scoped conversation persistence. None only when the
            host supplies its own, as the LangGraph Studio dev server does.
        store: Long-term, cross-session memory. Optional so the agent can run without it.
        middleware: Overrides the default stack; intended for tests.
    """
    llm = ChatGroq(model=settings.llm_model, temperature=settings.llm_temperature)
    return create_agent(
        llm,
        tools=get_all_tools(),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        store=store,
        context_schema=AgentContext,
        middleware=get_default_middleware() if middleware is None else middleware,
    )

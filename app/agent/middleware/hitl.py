"""Human-in-the-loop approval for tools marked sensitive in the registry."""

from langchain.agents.middleware import HumanInTheLoopMiddleware

from app.agent.registry import get_sensitive_tools


def get_hitl_middleware() -> HumanInTheLoopMiddleware | None:
    """Build interrupt configuration from the registry's `sensitive` flags.

    Returns None when no tool is marked sensitive, so the middleware is left out of the stack
    entirely rather than added as a no-op that still costs a graph step per tool call. Gating a
    future mutating tool needs only `sensitive=True` at registration.
    """
    sensitive = get_sensitive_tools()
    if not sensitive:
        return None
    return HumanInTheLoopMiddleware(interrupt_on={tool.name: True for tool in sensitive})

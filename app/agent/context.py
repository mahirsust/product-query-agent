"""Per-request context threaded through the agent."""

from pydantic import BaseModel


class AgentContext(BaseModel):
    """Identity available to tools and middleware.

    A pydantic model rather than a dataclass: LangGraph coerces either, but a dataclass emits a
    serialization warning on every model call.
    """

    user_id: int | None = None

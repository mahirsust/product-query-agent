"""Injects a user's remembered preferences into the system prompt."""

from langchain.agents.middleware import ModelRequest, dynamic_prompt

from app.agent.prompts import SYSTEM_PROMPT

_MEMORY_KEY = "profile"


@dynamic_prompt
async def long_term_memory_middleware(request: ModelRequest) -> str:
    """Return the system prompt, extended with anything known about the current user.

    Uses `dynamic_prompt` rather than a `before_model` state edit: the system prompt is set at
    agent construction and is not part of the mutable message list, so it can only be replaced on
    the outgoing request.
    """
    context = request.runtime.context
    store = request.runtime.store
    user_id = context.user_id if context else None
    if user_id is None or store is None:
        return SYSTEM_PROMPT

    item = await store.aget(("users", str(user_id), "memory"), _MEMORY_KEY)
    preferences = item.value.get("preferences", []) if item else []
    if not preferences:
        return SYSTEM_PROMPT

    summary = "; ".join(preferences)
    return (
        f"{SYSTEM_PROMPT}\n\nKnown preferences for this user (stated in a past session): {summary}."
    )

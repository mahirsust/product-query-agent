"""Tools that write to long-term memory."""

from langchain.tools import ToolRuntime, tool

from app.agent.registry import register_tool

_MEMORY_KEY = "profile"

# Stored preferences are injected into every prompt for that user, so the list is a per-turn token
# cost that nothing else ever shrinks. Oldest entries are dropped past this many.
MAX_PREFERENCES = 20


def _normalized(preference: str) -> str:
    """Comparison form for duplicate detection — case- and whitespace-insensitive."""
    return " ".join(preference.split()).casefold()


@register_tool
@tool
async def remember_preference(preference: str, runtime: ToolRuntime) -> str:
    """Save a preference the user has just explicitly stated about themselves, such as a price
    range or a product category they favour, so it survives into future sessions.

    Do not call this to look up, confirm, or answer a question about what is already remembered —
    everything already stored is given to you in the system prompt, so answer from there. Call
    this only when the user states something new about their own preferences.
    """
    user_id = runtime.context.user_id if runtime.context else None
    if user_id is None or runtime.store is None:
        return "I can't save preferences outside of a logged-in session."

    namespace = ("users", str(user_id), "memory")
    existing = await runtime.store.aget(namespace, _MEMORY_KEY)
    preferences = list(existing.value.get("preferences", [])) if existing else []

    # A recall-shaped question makes the model call this tool with text it just read back out of
    # the injected memory, which appends a duplicate on every such turn unless caught here. The
    # tool description discourages it; this makes the growth structurally impossible.
    if _normalized(preference) in {_normalized(p) for p in preferences}:
        return "Already noted, nothing to change."

    preferences.append(preference)
    del preferences[:-MAX_PREFERENCES]
    await runtime.store.aput(namespace, _MEMORY_KEY, {"preferences": preferences})
    return "Got it, I'll remember that."

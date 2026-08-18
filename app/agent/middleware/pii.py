"""PII redaction for user input before it reaches the model.

Redacting in `before_model` mutates the message content itself, so checkpointed state, logs and
traces all carry the redacted version. Only the model-call path is covered; anything written
outside it (such as a cache key) needs its own handling.
"""

from langchain.agents.middleware import AgentMiddleware, PIIMiddleware
from langgraph.runtime import Runtime

# Best-effort US-style phone pattern. Not exhaustive across international formats, which is
# acceptable here: the goal is keeping obvious phone numbers out of model-bound traffic.
_PHONE_PATTERN = r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"


def _build_redactors() -> list[PIIMiddleware]:
    """One built-in redactor per PII type; `PIIMiddleware` handles a single type per instance."""
    return [
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="redact", apply_to_input=True),
        PIIMiddleware("phone", strategy="redact", detector=_PHONE_PATTERN, apply_to_input=True),
    ]


class PIIRedactionMiddleware(AgentMiddleware):
    """Applies every PII redactor within a single hook.

    Mounting the built-in redactors directly would add one graph node per hook per type, and each
    node is a persisted checkpoint — six nodes and six extra checkpoint writes per turn for work
    that is pure in-process regex. Composing them keeps the built-in detectors while costing one
    node.
    """

    def __init__(self) -> None:
        """Build the redactors once; they are stateless and reused per call."""
        super().__init__()
        self._redactors = _build_redactors()

    def before_model(self, state, runtime: Runtime) -> dict | None:
        """Redact every PII type from the pending messages, reporting only what changed."""
        original = {message.id: message.content for message in state["messages"]}
        messages = list(state["messages"])

        for redactor in self._redactors:
            update = redactor.before_model({"messages": messages}, runtime)
            if not update:
                continue
            # Merge by id before handing the list to the next redactor, otherwise it would see
            # the unredacted content and undo the previous one's work.
            replacements = {message.id: message for message in update["messages"]}
            messages = [replacements.get(message.id, message) for message in messages]

        # Report only genuinely altered messages: a redactor may echo back the full list, and
        # returning untouched messages would write pointless state updates.
        changed = [m for m in messages if m.content != original.get(m.id)]
        return {"messages": changed} if changed else None

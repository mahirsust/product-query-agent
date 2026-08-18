"""Input guardrails: reject abusive or off-topic questions before any model call.

Screening runs cheapest-first — length cap, then local regex, then the Prompt Guard classifier —
so obvious cases short-circuit without a network round trip.
"""

import logging
import re

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from app.security import prompt_guard

logger = logging.getLogger("app.agent.guardrails")

MAX_INPUT_LENGTH = 2000

# Known injection phrasings. Kept alongside the classifier because they are free, instant, and
# catch the common cases without a network call.
_INJECTION_PATTERNS = [
    re.compile(r"\b(ignore|disregard)\b.{0,50}\b(instructions|prompts|rules)\b", re.IGNORECASE),
    re.compile(r"\byou are now\b", re.IGNORECASE),
    re.compile(r"\bsystem prompt\b", re.IGNORECASE),
    re.compile(r"\breveal\b.{0,20}\b(instructions|prompt)\b", re.IGNORECASE),
    re.compile(r"\bact as (a|an)\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bDAN\b"),
]

# Narrow and high-confidence only: a broad off-topic heuristic would reject legitimately-phrased
# shopping questions, which is worse than occasionally answering an off-topic one.
_OFF_TOPIC_PATTERNS = [
    re.compile(r"\bwrite (me |us )?(a |an )?(poem|essay|story|song|joke)\b", re.IGNORECASE),
    re.compile(
        r"\bwrite (a |an )?(python|javascript|sql|java|c\+\+) (script|function|program|code)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\btranslate (this|the following|that)\b", re.IGNORECASE),
    re.compile(r"\bsolve (this|the following) (math|equation|homework)\b", re.IGNORECASE),
    re.compile(r"\bwhat is the capital of\b", re.IGNORECASE),
    re.compile(r"\btell me a joke\b", re.IGNORECASE),
]

INJECTION_REFUSAL = (
    "I can't follow instructions embedded in a question like that. "
    "Ask me about product prices, reviews, or browsing the catalog instead."
)
OFF_TOPIC_REFUSAL = (
    "I'm a product assistant — I can only help with product prices, reviews, and browsing the "
    "catalog. For anything else, please use a general-purpose assistant."
)
LENGTH_REFUSAL = (
    f"That question is too long (max {MAX_INPUT_LENGTH} characters). Please ask something shorter."
)


def _last_human_text(state) -> str | None:
    """The most recent user message, or None if the turn has none."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return None


def _refuse(reason: str) -> dict:
    """End the turn with a refusal, skipping the model entirely."""
    return {"jump_to": "end", "messages": [AIMessage(content=reason)]}


class GuardrailsMiddleware(AgentMiddleware):
    """Short-circuits the agent before the model is called when input fails screening."""

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime: Runtime) -> dict | None:
        """Screen the user's message, returning a refusal if it fails any check."""
        text = _last_human_text(state)
        if text is None:
            return None

        if len(text) > MAX_INPUT_LENGTH:
            return _refuse(LENGTH_REFUSAL)

        if any(pattern.search(text) for pattern in _INJECTION_PATTERNS):
            return _refuse(INJECTION_REFUSAL)

        if any(pattern.search(text) for pattern in _OFF_TOPIC_PATTERNS):
            return _refuse(OFF_TOPIC_REFUSAL)

        # Catches novel or obfuscated phrasings the patterns above miss.
        verdict = await prompt_guard.classify(text)
        if verdict.is_injection:
            logger.warning(
                "prompt guard blocked an injection attempt",
                extra={"prompt_guard_score": verdict.score},
            )
            return _refuse(INJECTION_REFUSAL)

        return None

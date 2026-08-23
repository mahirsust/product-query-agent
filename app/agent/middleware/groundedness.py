"""Groundedness checking: flag numeric claims the tools did not support.

Ungrounded claims are logged rather than blocked — this is a visibility mechanism, not an
enforcement one.
"""

import logging
import re
from typing import NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from app.agent.prompts import MODEL_FAILURE_MESSAGE

logger = logging.getLogger("app.agent.groundedness")

_NUMBER_PATTERN = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
# Markdown ordered-list markers are formatting, not claims.
_LIST_MARKER_PATTERN = re.compile(r"(?m)^\s*\d+\.\s+")
# Rating-scale denominators ("4.37/5", "2.74 out of 5 stars") are a readability device the model
# adds, not a value sourced from tool output. Phrasing varies, so the whole phrase is stripped.
_RATING_SCALE_PATTERN = re.compile(r"(?i)/\s*\d+(?:\.\d+)?|out of\s+\d+(?:\.\d+)?(?:\s*stars?)?")

_MEMORY_KEY = "profile"


def _canonical(number: str) -> str:
    """Reduce a matched number to one form per value, so notation cannot fake a mismatch.

    Models pad decimals to a fixed width — a rating stored as `3.8` is written `3.80` — and these
    are compared as strings, so the padded form read as an unsupported claim. Only trailing zeros
    after a decimal point are stripped; integers keep every digit, since `100` and `10` differ.
    """
    number = number.lstrip("$").replace(",", "")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def _extract_numbers(text: str) -> set[str]:
    """Return the numeric values in `text`, ignoring formatting-only digits."""
    text = _LIST_MARKER_PATTERN.sub("", text)
    text = _RATING_SCALE_PATTERN.sub("", text)
    return {_canonical(match.group()) for match in _NUMBER_PATTERN.finditer(text)}


def _with_derived_discounts(numbers: set[str]) -> set[str]:
    """Extend a grounded set with the prices implied by applying a grounded percentage to it.

    Products carry `price` and `discountPercentage` separately, and a model asked about price
    routinely states the discounted figure too — `$79.99` less `14.39%` is `$68.48`. That value is
    arithmetic over two grounded numbers, not an invention, but it appears nowhere in the tool
    output, so without this every discounted-price answer reads as a fabricated claim.

    Only percentage-shaped operands are applied, which keeps this from admitting the product of
    any two numbers that happen to appear together.
    """
    values = []
    for number in numbers:
        try:
            values.append(float(number))
        except ValueError:
            continue

    derived = set()
    for amount in values:
        for percent in values:
            if 0 < percent < 100:
                derived.add(_canonical(f"{round(amount * (1 - percent / 100), 2)}"))
    return numbers | derived


def _is_infrastructure_error(content: str) -> bool:
    """Whether the answer is a model-failure notice rather than a real reply.

    Such messages carry incidental numbers (status codes, token counts) that would otherwise read
    as fabricated claims. The second prefix is langchain's built-in formatter, kept as a fallback
    in case the custom `on_failure` hook is bypassed.
    """
    return content == MODEL_FAILURE_MESSAGE or content.startswith("Model call failed after")


class GroundednessState(AgentState):
    """Agent state plus a marker for where the current turn began."""

    turn_start_index: NotRequired[int]


class GroundednessMiddleware(AgentMiddleware):
    """Logs numeric claims in the final answer that this turn's tool results do not support."""

    state_schema = GroundednessState

    async def abefore_agent(self, state, runtime: Runtime) -> dict:
        """Record where this turn begins.

        Runs once per invocation, unlike `aafter_model` which runs per ReAct iteration, so a tool
        result from an earlier turn cannot be mistaken for evidence supporting this one.
        """
        return {"turn_start_index": len(state["messages"])}

    async def aafter_model(self, state, runtime: Runtime) -> None:
        """Compare the final answer's numbers against what this turn actually retrieved."""
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None  # Mid-loop, not the final answer.

        content = str(last.content)
        if _is_infrastructure_error(content):
            return None

        claimed = _extract_numbers(content)
        if not claimed:
            return None

        grounded = await self._grounded_numbers(state, runtime)
        ungrounded = claimed - grounded
        if ungrounded:
            logger.warning(
                "answer contains numeric claims not found in this turn's tool results",
                extra={"ungrounded_numbers": sorted(ungrounded)},
            )
        return None

    async def _grounded_numbers(self, state, runtime: Runtime) -> set[str]:
        """Collect every number the answer is entitled to cite.

        Beyond tool output this includes the user's own question (echoing back "under $100" is not
        a fabrication) and any remembered preference, which the long-term memory middleware may
        have injected into the prompt.
        """
        turn_start = state.get("turn_start_index", 0)
        tool_text = " ".join(
            str(m.content) for m in state["messages"][turn_start:] if isinstance(m, ToolMessage)
        )
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
        )
        question_text = str(last_human.content) if last_human else ""

        # Derived values are computed from tool output only — the question and stored preferences
        # are the user's own words, not a source of catalogue arithmetic.
        grounded = _with_derived_discounts(_extract_numbers(tool_text))
        grounded |= _extract_numbers(question_text)

        user_id = runtime.context.user_id if runtime.context else None
        if user_id is not None and runtime.store is not None:
            namespace = ("users", str(user_id), "memory")
            item = await runtime.store.aget(namespace, _MEMORY_KEY)
            preferences = item.value.get("preferences", []) if item else []
            if preferences:
                grounded |= _extract_numbers("; ".join(preferences))

        return grounded

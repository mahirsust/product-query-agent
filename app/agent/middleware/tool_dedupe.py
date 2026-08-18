"""Drops duplicate tool calls from a single model response.

Smaller models sometimes emit the same call many times in one parallel batch. Every copy runs and
appends its result to the conversation, so the next request can exceed the model's tokens-per-
minute ceiling and fail outright — observed with `llama-3.1-8b-instant`, which emitted ten
identical searches and pushed the following request past its 6000 TPM limit.

Deduplicating is safe because the tools are read-only: identical arguments yield identical
results, so the copies add cost and no information.
"""

import json
import logging

from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger("app.agent.tool_dedupe")


def _signature(tool_call: dict) -> str:
    """A stable identity for a tool call, insensitive to argument ordering."""
    return f"{tool_call.get('name')}:{json.dumps(tool_call.get('args', {}), sort_keys=True)}"


def _deduplicate(tool_calls: list[dict]) -> list[dict]:
    """Keep the first of each distinct call, preserving order."""
    seen: set[str] = set()
    unique = []
    for tool_call in tool_calls:
        signature = _signature(tool_call)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(tool_call)
    return unique


class ToolCallDedupeMiddleware(AgentMiddleware):
    """Collapses repeated identical tool calls within one model response."""

    async def awrap_model_call(self, request, handler):
        """Strip repeated calls from the model's response before the tools run."""
        response = await handler(request)

        for message in getattr(response, "result", []) or []:
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls or len(tool_calls) < 2:
                continue

            unique = _deduplicate(tool_calls)
            if len(unique) < len(tool_calls):
                logger.warning(
                    "dropped duplicate tool calls from a single model response",
                    extra={"dropped": len(tool_calls) - len(unique), "kept": len(unique)},
                )
                message.tool_calls = unique

        return response

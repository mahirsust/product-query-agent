"""Tests for collapsing repeated identical tool calls in one model response."""

import asyncio

from langchain_core.messages import AIMessage

from app.agent.middleware.tool_dedupe import ToolCallDedupeMiddleware


def _run(coro):
    return asyncio.run(coro)


class _Response:
    def __init__(self, messages):
        self.result = messages


def _call(tool_name: str, **args) -> dict:
    return {
        "name": tool_name,
        "args": args,
        "id": f"call_{tool_name}_{len(args)}",
        "type": "tool_call",
    }


def _apply(message: AIMessage) -> AIMessage:
    async def handler(_request):
        return _Response([message])

    _run(ToolCallDedupeMiddleware().awrap_model_call(request=None, handler=handler))
    return message


def test_identical_calls_collapse_to_one():
    """The observed failure: a batch of identical searches whose results together pushed the next
    request past the model's per-minute token ceiling."""
    message = AIMessage(
        content="",
        tool_calls=[_call("search_products", query="fragrances", max_price=100) for _ in range(10)],
    )
    _apply(message)
    assert len(message.tool_calls) == 1


def test_argument_order_does_not_defeat_matching():
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_products", "args": {"query": "a", "max_price": 10}, "id": "1"},
            {"name": "search_products", "args": {"max_price": 10, "query": "a"}, "id": "2"},
        ],
    )
    _apply(message)
    assert len(message.tool_calls) == 1


def test_distinct_calls_are_preserved():
    """Parallel calls are a feature when they differ — only exact repeats are waste."""
    message = AIMessage(
        content="",
        tool_calls=[
            _call("get_product", name="macbook"),
            _call("search_products", query="laptops"),
        ],
    )
    _apply(message)
    assert len(message.tool_calls) == 2


def test_same_tool_with_different_arguments_is_kept():
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_products", "args": {"query": "laptops"}, "id": "1"},
            {"name": "search_products", "args": {"query": "fragrances"}, "id": "2"},
        ],
    )
    _apply(message)
    assert len(message.tool_calls) == 2


def test_single_call_is_untouched():
    message = AIMessage(content="", tool_calls=[_call("get_product", name="macbook")])
    _apply(message)
    assert len(message.tool_calls) == 1


def test_message_without_tool_calls_is_untouched():
    message = AIMessage(content="just an answer")
    _apply(message)
    assert not message.tool_calls

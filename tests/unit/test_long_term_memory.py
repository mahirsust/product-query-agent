import asyncio

from langchain.agents.middleware.types import ModelRequest
from langgraph.store.memory import InMemoryStore

from app.agent.context import AgentContext
from app.agent.middleware.long_term_memory import long_term_memory_middleware
from app.agent.prompts import SYSTEM_PROMPT


class _Runtime:
    def __init__(self, context, store):
        self.context = context
        self.store = store


def _run(coro):
    return asyncio.run(coro)


async def _echo_handler(request):
    return request


def _render_prompt(context, store) -> str:
    request = ModelRequest(model=None, messages=[], runtime=_Runtime(context, store))
    result = _run(long_term_memory_middleware.awrap_model_call(request, _echo_handler))
    return result.system_message.content


def test_no_preferences_returns_base_prompt():
    prompt = _render_prompt(AgentContext(user_id=1), InMemoryStore())
    assert prompt == SYSTEM_PROMPT


def test_no_user_id_returns_base_prompt():
    prompt = _render_prompt(AgentContext(user_id=None), InMemoryStore())
    assert prompt == SYSTEM_PROMPT


def test_no_store_returns_base_prompt():
    prompt = _render_prompt(AgentContext(user_id=1), None)
    assert prompt == SYSTEM_PROMPT


def test_preferences_injected_into_prompt():
    store = InMemoryStore()
    _run(
        store.aput(
            ("users", "1", "memory"), "profile", {"preferences": ["budget laptops under $1000"]}
        )
    )
    prompt = _render_prompt(AgentContext(user_id=1), store)
    assert SYSTEM_PROMPT in prompt
    assert "budget laptops under $1000" in prompt


def test_multiple_preferences_joined():
    store = InMemoryStore()
    _run(
        store.aput(
            ("users", "1", "memory"),
            "profile",
            {"preferences": ["prefers Apple products", "budget under $1000"]},
        )
    )
    prompt = _render_prompt(AgentContext(user_id=1), store)
    assert "prefers Apple products" in prompt
    assert "budget under $1000" in prompt


def test_different_user_gets_no_bleed_through():
    store = InMemoryStore()
    _run(
        store.aput(("users", "1", "memory"), "profile", {"preferences": ["prefers Apple products"]})
    )
    prompt = _render_prompt(AgentContext(user_id=2), store)
    assert prompt == SYSTEM_PROMPT

import asyncio

from langgraph.store.memory import InMemoryStore

from app.agent.context import AgentContext
from app.agent.memory_tools import MAX_PREFERENCES, remember_preference


class _Runtime:
    def __init__(self, context, store):
        self.context = context
        self.store = store


def _save(preference: str, runtime) -> str:
    """Call the tool's underlying coroutine directly, bypassing runtime injection."""
    return asyncio.run(remember_preference.coroutine(preference=preference, runtime=runtime))


def _stored(store, user_id: int) -> list[str]:
    item = asyncio.run(store.aget(("users", str(user_id), "memory"), "profile"))
    return list(item.value["preferences"]) if item else []


def test_preference_is_saved():
    store = InMemoryStore()
    _save("laptops under $1500", _Runtime(AgentContext(user_id=1), store))
    assert _stored(store, 1) == ["laptops under $1500"]


def test_distinct_preferences_both_kept():
    store = InMemoryStore()
    runtime = _Runtime(AgentContext(user_id=1), store)
    _save("laptops under $1500", runtime)
    _save("prefers Apple products", runtime)
    assert _stored(store, 1) == ["laptops under $1500", "prefers Apple products"]


def test_exact_duplicate_is_not_appended():
    """A recall-shaped question makes the model re-save what it just read back."""
    store = InMemoryStore()
    runtime = _Runtime(AgentContext(user_id=1), store)
    _save("laptops under $1500", runtime)
    result = _save("laptops under $1500", runtime)
    assert _stored(store, 1) == ["laptops under $1500"]
    assert "Already noted" in result


def test_duplicate_detection_ignores_case_and_whitespace():
    store = InMemoryStore()
    runtime = _Runtime(AgentContext(user_id=1), store)
    _save("Laptops under $1500", runtime)
    _save("  laptops   UNDER $1500 ", runtime)
    assert _stored(store, 1) == ["Laptops under $1500"]


def test_list_is_capped_dropping_oldest():
    store = InMemoryStore()
    runtime = _Runtime(AgentContext(user_id=1), store)
    for i in range(MAX_PREFERENCES + 5):
        _save(f"preference {i}", runtime)
    stored = _stored(store, 1)
    assert len(stored) == MAX_PREFERENCES
    assert stored[0] == "preference 5"
    assert stored[-1] == f"preference {MAX_PREFERENCES + 4}"


def test_writes_are_scoped_per_user():
    store = InMemoryStore()
    _save("laptops under $1500", _Runtime(AgentContext(user_id=1), store))
    _save("gaming keyboards", _Runtime(AgentContext(user_id=2), store))
    assert _stored(store, 1) == ["laptops under $1500"]
    assert _stored(store, 2) == ["gaming keyboards"]


def test_refuses_without_user_id():
    store = InMemoryStore()
    result = _save("laptops under $1500", _Runtime(AgentContext(user_id=None), store))
    assert "logged-in session" in result
    assert _stored(store, 1) == []


def test_refuses_without_store():
    result = _save("laptops under $1500", _Runtime(AgentContext(user_id=1), None))
    assert "logged-in session" in result

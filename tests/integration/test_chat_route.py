import asyncio

from langchain_core.messages import AIMessage, ToolMessage

from app.cache import response_cache
from app.config import settings
from app.db.repository import create_user
from app.security.auth import create_access_token, hash_password
from tests.integration.conftest import FakeAgent


def _user_and_token(db_session, username="chatuser"):
    user = create_user(db_session, username, hash_password("pw"))
    db_session.commit()
    return user, create_access_token(user.id)


def test_chat_requires_auth(client):
    resp = client.post("/chat", json={"question": "hi", "thread_id": "t1"})
    assert resp.status_code == 401


def test_chat_happy_path_returns_answer_and_tool_calls(
    client, app, db_session, fake_redis, monkeypatch
):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    user, token = _user_and_token(db_session)
    fake_agent = FakeAgent(
        response_messages=[
            ToolMessage(content=str({"price": 79.99}), tool_call_id="1", name="get_product"),
            AIMessage(content="It's $79.99."),
        ]
    )
    app.state.agent = fake_agent

    resp = client.post(
        "/chat",
        json={"question": "price?", "thread_id": "t1"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "It's $79.99."
    assert body["tool_calls"] == ["get_product"]
    assert body["thread_id"] == "t1"


def test_chat_namespaces_thread_id_by_user(client, app, db_session, fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    user, token = _user_and_token(db_session)
    fake_agent = FakeAgent(response_messages=[AIMessage(content="hi")])
    app.state.agent = fake_agent

    client.post(
        "/chat",
        json={"question": "hello", "thread_id": "my-thread"},
        headers={"Authorization": f"Bearer {token}"},
    )

    namespaced = fake_agent.invoked_with["config"]["configurable"]["thread_id"]
    assert namespaced == f"user-{user.id}:my-thread"


def test_chat_two_users_same_client_thread_id_get_different_namespaces(
    client, app, db_session, fake_redis, monkeypatch
):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    user_a, token_a = _user_and_token(db_session, "alice")
    user_b, token_b = _user_and_token(db_session, "bob")
    fake_agent = FakeAgent(response_messages=[AIMessage(content="hi")])
    app.state.agent = fake_agent

    client.post(
        "/chat",
        json={"question": "question a", "thread_id": "shared"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    namespaced_a = fake_agent.invoked_with["config"]["configurable"]["thread_id"]

    client.post(
        "/chat",
        json={"question": "question b", "thread_id": "shared"},
        headers={"Authorization": f"Bearer {token_b}"},
    )
    namespaced_b = fake_agent.invoked_with["config"]["configurable"]["thread_id"]

    assert namespaced_a != namespaced_b


def test_chat_budget_exceeded_returns_402(client, app, db_session, fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 0.0)
    user, token = _user_and_token(db_session)
    fake_agent = FakeAgent(response_messages=[AIMessage(content="should not be reached")])
    app.state.agent = fake_agent

    resp = client.post(
        "/chat",
        json={"question": "a question never asked before", "thread_id": "t1"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 402
    assert fake_agent.invoked_with is None


def test_chat_rate_limited_returns_429(client, app, db_session, fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    monkeypatch.setattr(settings, "max_llm_calls_per_minute_per_user", 0)
    user, token = _user_and_token(db_session)
    fake_agent = FakeAgent(response_messages=[AIMessage(content="should not be reached")])
    app.state.agent = fake_agent

    resp = client.post(
        "/chat",
        json={"question": "a rate-limited question", "thread_id": "t1"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 429
    assert fake_agent.invoked_with is None


def test_chat_serves_cached_answer_on_first_turn_without_calling_agent(
    client, app, db_session, fake_redis, monkeypatch
):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    user, token = _user_and_token(db_session)
    asyncio.run(response_cache.set("a cached question", "cached answer", user.id))
    fake_agent = FakeAgent(response_messages=[AIMessage(content="should not be used")])
    app.state.agent = fake_agent

    resp = client.post(
        "/chat",
        json={"question": "a cached question", "thread_id": "t1"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200
    assert resp.json()["answer"] == "cached answer"
    assert resp.json()["tool_calls"] == []
    assert fake_agent.invoked_with is None


def test_chat_follow_up_turn_is_never_served_from_cache(
    client, app, db_session, fake_redis, monkeypatch
):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    user, token = _user_and_token(db_session)
    asyncio.run(response_cache.set("what are the reviews?", "cached answer", user.id))
    fake_agent = FakeAgent(
        response_messages=[AIMessage(content="fresh answer")],
        prior_messages=[AIMessage(content="earlier turn in this thread")],
    )
    app.state.agent = fake_agent

    resp = client.post(
        "/chat",
        json={"question": "what are the reviews?", "thread_id": "t1"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200
    assert resp.json()["answer"] == "fresh answer"
    assert fake_agent.invoked_with is not None


def test_one_users_cached_answer_is_never_served_to_another(
    client, app, db_session, fake_redis, monkeypatch
):
    """End-to-end guard for the cross-user cache leak: user A's cached (personalized) answer must
    not reach user B asking the identical question. Before the per-user cache key, user B got
    A's answer verbatim — and did so *before* the usage check, bypassing budget limits too."""
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    user_a, _ = _user_and_token(db_session, "cache_leak_a")
    _, token_b = _user_and_token(db_session, "cache_leak_b")
    asyncio.run(
        response_cache.set("what laptops do you have", "Given your $1000 budget...", user_a.id)
    )
    fake_agent = FakeAgent(response_messages=[AIMessage(content="fresh answer for B")])
    app.state.agent = fake_agent

    resp = client.post(
        "/chat",
        json={"question": "what laptops do you have", "thread_id": "t1"},
        headers={"Authorization": f"Bearer {token_b}"},
    )

    assert resp.status_code == 200
    assert "1000" not in resp.json()["answer"]
    assert resp.json()["answer"] == "fresh answer for B"

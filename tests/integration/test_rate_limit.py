from langchain_core.messages import AIMessage

from app.config import settings
from app.db.repository import create_user
from app.security.auth import create_access_token, hash_password
from tests.integration.conftest import FakeAgent


def test_signup_rate_limit_returns_429_after_five_per_minute(client):
    responses = [
        client.post("/auth/signup", json={"username": f"ratelimituser{i}", "password": "pw"})
        for i in range(6)
    ]
    statuses = [r.status_code for r in responses]

    assert statuses[:5] == [200, 200, 200, 200, 200]
    assert statuses[5] == 429


def test_login_rate_limit_isolated_from_signup(client, db_session):
    """auth routes are rate-limited independently — exhausting /auth/signup's 5/minute cap must
    not itself block /auth/login."""
    create_user(db_session, "loginratelimituser", hash_password("pw"))
    db_session.commit()

    for i in range(5):
        client.post("/auth/signup", json={"username": f"unrelated{i}", "password": "pw"})
    exhausted = client.post("/auth/signup", json={"username": "unrelated5", "password": "pw"})
    assert exhausted.status_code == 429

    login_resp = client.post(
        "/auth/login", json={"username": "loginratelimituser", "password": "pw"}
    )
    assert login_resp.status_code == 200


def test_chat_rate_limit_keyed_by_user_not_shared_across_users(
    client, app, db_session, fake_redis, monkeypatch
):
    """/chat's 30/minute cap is keyed by authenticated user_id (see rate_limit_key) — exhausting
    one user's bucket must not affect a different user's requests on the same route."""
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    user_a = create_user(db_session, "bucketuser_a", hash_password("pw"))
    user_b = create_user(db_session, "bucketuser_b", hash_password("pw"))
    db_session.commit()
    token_a = create_access_token(user_a.id)
    token_b = create_access_token(user_b.id)
    app.state.agent = FakeAgent(response_messages=[AIMessage(content="hi")])

    statuses_a = [
        client.post(
            "/chat",
            json={"question": f"question number {i}", "thread_id": "t1"},
            headers={"Authorization": f"Bearer {token_a}"},
        ).status_code
        for i in range(31)
    ]
    assert 429 in statuses_a

    resp_b = client.post(
        "/chat",
        json={"question": "a fresh question for user b", "thread_id": "t1"},
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp_b.status_code != 429

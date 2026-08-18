def test_signup_creates_user_and_returns_token(client):
    resp = client.post("/auth/signup", json={"username": "newuser", "password": "pw123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_signup_duplicate_username_conflicts(client):
    client.post("/auth/signup", json={"username": "dupeuser", "password": "pw123"})
    resp = client.post("/auth/signup", json={"username": "dupeuser", "password": "different"})
    assert resp.status_code == 409


def test_login_success_returns_token(client):
    client.post("/auth/signup", json={"username": "loginuser", "password": "correctpw"})
    resp = client.post("/auth/login", json={"username": "loginuser", "password": "correctpw"})
    assert resp.status_code == 200
    assert resp.json()["access_token"]


def test_login_wrong_password_returns_401(client):
    client.post("/auth/signup", json={"username": "wrongpwuser", "password": "correctpw"})
    resp = client.post("/auth/login", json={"username": "wrongpwuser", "password": "nope"})
    assert resp.status_code == 401


def test_login_unknown_user_returns_401(client):
    resp = client.post("/auth/login", json={"username": "doesnotexist", "password": "pw"})
    assert resp.status_code == 401


def test_signup_then_login_tokens_both_authenticate(client):
    signup_resp = client.post("/auth/signup", json={"username": "twotoken", "password": "pw123"})
    login_resp = client.post("/auth/login", json={"username": "twotoken", "password": "pw123"})

    for token in (signup_resp.json()["access_token"], login_resp.json()["access_token"]):
        chat_resp = client.post(
            "/chat",
            json={"question": "hi", "thread_id": "t1"},
            headers={"Authorization": f"Bearer {token}"},
        )
        # No agent is wired up in this test, so the request may fail downstream, but it must not
        # be rejected as unauthenticated — proves the token itself is valid.
        assert chat_resp.status_code != 401


def test_healthz_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_ok_with_working_db(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200

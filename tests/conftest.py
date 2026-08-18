import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base

load_dotenv()


@pytest.fixture()
def db_session():
    """A fresh in-memory SQLite DB per test, isolated from the real product_query_agent.db."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def fake_redis():
    """Patches app.cache.redis_client's module-level client with an isolated fakeredis instance
    so cache/usage-tracker tests never touch the real dev Redis instance."""
    import fakeredis

    from app.cache import redis_client

    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    original = redis_client._client
    redis_client._client = fake
    try:
        yield fake
    finally:
        redis_client._client = original


@pytest.fixture(autouse=True)
def _clean_tool_registry():
    """The tool registry is process-global state (app/agent/registry.py's module-level dict) —
    reset it around every test so one test's register_tool calls can't leak into another's."""
    from app.agent import registry

    saved = dict(registry._registry)
    registry._registry.clear()
    yield
    registry._registry.clear()
    registry._registry.update(saved)


@pytest.fixture(autouse=True)
def _disable_prompt_guard(monkeypatch):
    """Keep the Prompt Guard classifier off by default so no unit test makes a network call.

    Tests that exercise the classifier re-enable it and stub `prompt_guard.classify` explicitly.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "prompt_guard_enabled", False)

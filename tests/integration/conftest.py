import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import auth, chat, health
from app.db.base import Base, get_session
from app.security.rate_limit import limiter


@pytest.fixture()
def test_engine():
    # StaticPool: FastAPI's TestClient runs each request on a worker thread (via anyio), which
    # would otherwise get its own fresh, table-less `:memory:` DB — SQLite's default per-thread
    # connection pooling for `:memory:` only shares a connection within a single thread.
    # StaticPool forces one shared connection across all threads/checkouts, so the tables created
    # here (on the fixture's thread) are visible to the request-handling thread too.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(test_engine):
    session_factory = sessionmaker(bind=test_engine, autoflush=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def app(test_engine):
    """A minimal FastAPI app wired with the real routers but without the real lifespan (which
    would build a real agent and spawn the MCP subprocess) — routes needing an agent get one via
    `app.state.agent`, set per-test."""
    session_factory = sessionmaker(bind=test_engine, autoflush=False)

    def override_get_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    test_app = FastAPI()
    test_app.state.limiter = limiter
    test_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    test_app.include_router(auth.router)
    test_app.include_router(chat.router)
    test_app.include_router(health.router)
    test_app.dependency_overrides[get_session] = override_get_session
    test_app.state.agent = None
    # Pre-set: the real app builds the agent in a background task and `get_agent` waits on this
    # event. Tests supply `app.state.agent` directly, so warm-up is already "done" as far as they
    # are concerned — leaving it clear would make every agent-backed test block for 60s.
    test_app.state.agent_ready = asyncio.Event()
    test_app.state.agent_ready.set()
    return test_app


@pytest.fixture()
def client(app):
    # raise_server_exceptions=False: some tests deliberately leave app.state.agent unset (a
    # fake-double concern, not something under test in auth-only tests) — without this, an
    # unhandled route exception raises in the test process instead of yielding a normal 500
    # response to assert against.
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """slowapi's limiter is backed by the real dev Redis (storage_uri=settings.redis_url) — reset
    its counters before every test so the 5/minute auth caps don't leak state across tests."""
    limiter.reset()
    yield


class FakeAgentState:
    def __init__(self, messages):
        self.values = {"messages": messages} if messages else {}


class FakeAgent:
    """Stands in for the real LangGraph agent so route tests never need Groq/MCP. Records what it
    was invoked with so tests can assert on thread namespacing/context without a real model."""

    def __init__(self, response_messages, prior_messages=None):
        self._response_messages = response_messages
        self._prior_messages = prior_messages or []
        self.invoked_with: dict | None = None

    async def aget_state(self, _config):
        return FakeAgentState(self._prior_messages)

    async def ainvoke(self, input_, config, context):
        self.invoked_with = {"input": input_, "config": config, "context": context}
        return {"messages": [*self._prior_messages, *self._response_messages]}

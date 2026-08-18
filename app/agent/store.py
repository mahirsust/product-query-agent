"""Long-term, cross-session memory store."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from app.config import settings


@asynccontextmanager
async def get_store() -> AsyncGenerator[BaseStore]:
    """Yield a store appropriate to the configured database.

    Async for the same reason as the checkpointer: callers use `aget`/`aput`, which the
    synchronous Postgres store does not implement. Non-Postgres databases fall back to in-memory
    storage, which does not survive a restart.
    """
    if settings.database_url.startswith("postgresql"):
        from langgraph.store.postgres.aio import AsyncPostgresStore

        async with AsyncPostgresStore.from_conn_string(settings.database_url_psycopg) as store:
            await store.setup()
            yield store
    else:
        yield InMemoryStore()

"""Short-term, thread-scoped conversation persistence."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings


@asynccontextmanager
async def get_checkpointer() -> AsyncGenerator[BaseCheckpointSaver]:
    """Yield a checkpointer appropriate to the configured database.

    Postgres deployments must use the async saver: the application invokes the agent through
    `ainvoke`/`aget_state`, and the synchronous saver leaves those methods unimplemented. Other
    databases fall back to in-memory persistence, which does not survive a restart.
    """
    if settings.database_url.startswith("postgresql"):
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(
            settings.database_url_psycopg
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    else:
        yield InMemorySaver()

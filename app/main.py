"""FastAPI application factory and startup wiring.

Nothing here imports the agent stack at module level. `langchain_groq` and `app.agent.graph` cost
~47s to import on a 0.1-CPU instance, and uvicorn binds its socket only *after* the lifespan
yields — so anything imported or awaited before that point is time the platform sees as "no open
port". Those imports live inside `_warm_up` instead.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import auth, chat, health
from app.config import settings
from app.logging_config import configure_logging, configure_tracing
from app.security.rate_limit import limiter

logger = logging.getLogger("app.main")


async def _warm_up(app: FastAPI, stack: AsyncExitStack) -> None:
    """Migrate the database and build the agent, off the request-serving critical path.

    Runs as a background task so the socket is already accepting connections while this proceeds.
    Requests that need the agent wait on `app.state.agent_ready`; see `deps.get_agent`.

    Migrations run here rather than in the container's CMD for the same reason — they cost ~59s on
    a small instance, all of it before uvicorn would otherwise start. With a single instance
    Alembic's own lock makes this safe; running several replicas would race here, and the fix then
    is a pre-deploy migration step rather than doing it in-process.
    """
    try:
        from alembic.config import Config

        from alembic import command
        from app.agent.graph import build_agent
        from app.agent.mcp_client import get_mcp_tools
        from app.agent.registry import register_tools
        from app.agent.store import get_store
        from app.db.checkpointer import get_checkpointer

        await asyncio.to_thread(command.upgrade, Config("alembic.ini"), "head")

        register_tools(await get_mcp_tools())
        checkpointer = await stack.enter_async_context(get_checkpointer())
        store = await stack.enter_async_context(get_store())
        app.state.agent = build_agent(checkpointer=checkpointer, store=store)
    except Exception:
        # Left unset deliberately: `agent_ready` stays clear, so /readyz keeps reporting 503 and
        # /chat keeps returning "still starting" rather than serving with a half-built agent.
        logger.exception("agent warm-up failed")
        raise
    else:
        app.state.agent_ready.set()
        logger.info("agent warm-up complete")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Yield immediately, then build the agent in the background.

    Doing the work here instead would delay the socket bind past the platform's port-scan window
    (measured at ~240s on a 0.1-CPU instance) and fail the deploy outright.
    """
    app.state.agent = None
    app.state.agent_ready = asyncio.Event()

    async with AsyncExitStack() as stack:
        task = asyncio.create_task(_warm_up(app, stack))
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def create_app() -> FastAPI:
    """Build the application: logging, tracing, CORS, rate limiting, and routes."""
    load_dotenv()
    configure_logging()
    configure_tracing()

    app = FastAPI(title="Product Query Agent", lifespan=lifespan)

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(health.router)
    return app


app = create_app()

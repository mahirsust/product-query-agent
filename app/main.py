"""FastAPI application factory and startup wiring."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.agent.graph import build_agent
from app.agent.mcp_client import get_mcp_tools
from app.agent.registry import register_tools
from app.agent.store import get_store
from app.api.routes import auth, chat, health
from app.config import settings
from app.db.checkpointer import get_checkpointer
from app.logging_config import configure_logging, configure_tracing
from app.security.rate_limit import limiter


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Resolve MCP tools and build the agent once, before the first request.

    Doing this per request would spawn the MCP subprocess and re-open database pools every time.
    """
    register_tools(await get_mcp_tools())
    async with get_checkpointer() as checkpointer, get_store() as store:
        app.state.agent = build_agent(checkpointer=checkpointer, store=store)
        yield


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

"""Shared FastAPI dependencies."""

import asyncio

from fastapi import HTTPException, Request, status

from app.db.base import get_session as get_db_session
from app.security.auth import get_current_user

__all__ = ["get_agent", "get_current_user", "get_db_session"]

# The agent is built by a background task (see app/main.py). Waiting rather than failing instantly
# means the first visitor after a cold start gets a slow answer instead of an error; the ceiling
# keeps a request from hanging indefinitely if warm-up has failed outright.
AGENT_READY_TIMEOUT_SECONDS = 60.0


async def get_agent(request: Request):
    """Return the agent, waiting for warm-up to finish if it is still in progress."""
    ready: asyncio.Event = request.app.state.agent_ready
    if not ready.is_set():
        try:
            await asyncio.wait_for(ready.wait(), timeout=AGENT_READY_TIMEOUT_SECONDS)
        except TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The service is still starting up. Please try again in a moment.",
            ) from None
    return request.app.state.agent

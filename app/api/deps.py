"""Shared FastAPI dependencies."""

from fastapi import Request

from app.db.base import get_session as get_db_session
from app.security.auth import get_current_user

__all__ = ["get_agent", "get_current_user", "get_db_session"]


def get_agent(request: Request):
    """Return the agent built once during application startup."""
    return request.app.state.agent

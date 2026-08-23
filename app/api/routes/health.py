"""Liveness and readiness probes, both unauthenticated."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db_session

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process is up. Deliberately checks no dependencies.

    Point a platform health check here, not at /readyz: this answers as soon as the socket is
    open, whereas /readyz stays 503 until the agent has finished warming up in the background.
    """
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request, session: Session = Depends(get_db_session)) -> dict[str, str]:
    """Readiness: the process can actually serve a chat turn.

    That means a reachable database *and* a built agent — while warm-up is still running the
    process is up but cannot answer anything, and reporting ready would be a lie.
    """
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not reachable"
        ) from exc

    if not request.app.state.agent_ready.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent still starting up"
        )
    return {"status": "ok"}

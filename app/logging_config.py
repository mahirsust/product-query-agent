"""Structured JSON logging with a per-request correlation id."""

import contextvars
import json
import logging
import os
from datetime import UTC, datetime

from app.config import settings

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

_STANDARD_LOG_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


def set_correlation_id(correlation_id: str | None) -> None:
    """Tag every subsequent log line in this context with an id, so one request's lines can be
    followed across modules."""
    _correlation_id.set(correlation_id)


def get_correlation_id() -> str | None:
    """The id for the current context, or None outside a request."""
    return _correlation_id.get()


class _JSONFormatter(logging.Formatter):
    """Renders records as one JSON object per line, merging any `extra` fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize one record, including the active correlation id."""
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        extra = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_LOG_RECORD_ATTRS}
        if extra:
            payload.update(extra)

        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Replace root handlers with a single JSON-formatted stream handler."""
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def configure_tracing() -> None:
    """Exports settings.langsmith_* into os.environ, which is where LangChain's tracing
    machinery actually reads it from — not from our Settings object directly. Must run before
    ChatGroq/create_agent construction."""
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
    if settings.langsmith_api_key.get_secret_value():
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

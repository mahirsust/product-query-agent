# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

# Install deps first, separate from app code, so this layer only rebuilds when the lockfile
# changes rather than on every source edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-default-groups --group backend

COPY app/ ./app/
COPY mcp_servers/ ./mcp_servers/
RUN uv sync --frozen --no-default-groups --group backend

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY app/ ./app/
COPY mcp_servers/ ./mcp_servers/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Drop root: a container process compromised through the app (or through the MCP stdio subprocess
# it spawns) would otherwise be running as uid 0. Nothing here needs write access to the image —
# state lives in Postgres and Redis — so the app files stay owned by root and are merely readable.
RUN useradd --system --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# Migrations are NOT run here. They cost ~59s on a small instance, entirely before uvicorn starts,
# and the platform sees that as a container with no open port. They run inside the app's background
# warm-up task instead (app/main.py), so they still happen automatically — see CLAUDE.md's
# migration gotcha for why they can never simply be skipped.
#
# $PORT rather than a literal: platforms such as Render assign the port at runtime and route to it,
# so a hardcoded 8000 leaves the container listening where nothing connects. The default keeps
# docker-compose and local runs unchanged.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

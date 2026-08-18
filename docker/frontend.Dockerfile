# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

# Installs only the `frontend` dependency group, so none of the agent stack (langchain, langgraph,
# psycopg, alembic) lands in this image. Still one lockfile: the group is resolved from the same
# uv.lock the backend uses.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-default-groups --group frontend

COPY frontend/ ./frontend/
RUN uv sync --frozen --no-default-groups --group frontend

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY frontend/ ./frontend/

# Drop root — see the equivalent note in backend.Dockerfile. Streamlit writes nothing outside
# /tmp and the home directory created here.
RUN useradd --system --create-home --uid 1000 appuser
USER appuser

EXPOSE 8501

# Shell form, not exec form: `${PORT}` only expands through a shell, and platforms such as Render
# assign the port at runtime. The default keeps docker-compose and local runs unchanged.
CMD ["sh", "-c", "streamlit run frontend/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]

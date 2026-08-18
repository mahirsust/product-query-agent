# Product Query Agent

A production-shaped LangChain/LangGraph tool-calling agent that answers questions about product
prices, specs, stock and reviews. It runs as a Streamlit chat UI over a FastAPI backend, with JWT
auth, Postgres, Redis, and product data fetched through an MCP server from the public
[DummyJSON](https://dummyjson.com/docs/products) catalogue.

The LLM backend is Groq (`openai/gpt-oss-120b`).

## Features

- **Tool-calling agent** over an MCP server — two deliberately flat tools, kept small because tool
  schemas are re-sent on every model call.
- **Conversation memory** — short-term per thread (LangGraph checkpointer) and long-term per user
  (preferences recalled across sessions).
- **Middleware pipeline** — prompt-injection guardrails, PII redaction, cost tracking,
  groundedness checking, tool-call de-duplication.
- **Auth and multi-tenancy** — JWT signup/login; threads, caches and memories are scoped per user.
- **Cost controls** — five per-user caps (daily dollars, daily/minute calls, daily/minute tokens),
  a per-user response cache, and a product cache with its own TTL.
- **Evaluated** — a 12-case golden dataset gates changes with a pass-rate threshold and a
  groundedness check.

## Architecture

```
Streamlit (frontend)  ──HTTP + JWT──▶  FastAPI (backend)
                                         ├── LangGraph agent ──▶ Groq (LLM)
                                         │     └── middleware: guardrails · PII redaction ·
                                         │        cost tracking · groundedness · long-term memory
                                         ├── MCP server (stdio subprocess) ──▶ DummyJSON API
                                         ├── Postgres — users, product cache, conversation state
                                         └── Redis — response cache, per-user usage caps
```

**Request path for `POST /chat`:** validate schema → verify JWT → serve from the response cache if
the question is a first turn → check per-user rate and budget caps → run the agent (guardrails,
tool loop, groundedness) → record usage → cache the answer.

Two tools back the agent:

| Tool | Purpose |
|---|---|
| `get_product(name)` | Everything known about one product — price, discount, rating, stock, brand, warranty, shipping, returns, dimensions, tags, reviews |
| `search_products(query, max_price)` | Browse by category or keyword; called with no arguments it lists the catalogue's categories |

## Quickstart

**Prerequisites:** [uv](https://docs.astral.sh/uv/), a Groq API key, and Docker (for the
containerised path).

```bash
# 1. Install dependencies
uv sync

# 2. Configure
cp .env.example .env
#    then set GROQ_API_KEY, JWT_SECRET_KEY and POSTGRES_PASSWORD in .env

# 3a. Everything containerised: backend, frontend, Postgres, Redis
docker compose up --build        # UI on http://localhost:8501

# 3b. ...or run locally
uv run alembic upgrade head      # required — nothing creates tables implicitly
uv run uvicorn app.main:app --reload
uv run streamlit run frontend/streamlit_app.py

# 3c. ...or just the dev REPL, no auth or HTTP
uv run main.py                   # type 'exit' to quit
```

Generate a JWT secret with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

> **The agent ships with a `$0.00` daily budget.** Until you raise
> `COST_BUDGET_USD_PER_USER_PER_DAY`, every `/chat` request returns **402** before reaching the
> model. This is deliberate — it prevents an unattended deployment from spending — but it is also
> the single most common reason a fresh setup appears broken.

> **Migrations are not automatic outside Docker.** Run `uv run alembic upgrade head` first, or
> every product lookup fails with `no such table`. The backend container runs it on startup.

## Configuration

**`.env` is the single source of truth for every secret and tunable**, and `.env.example` is the
complete annotated reference — every field on `Settings` appears in both. Nothing sensitive is
hardcoded in the application, `docker-compose.yml`, or CI: compose loads `.env` via `env_file`, and
CI generates a throwaway `.env` so it exercises the same configuration path as a local run.

### Required

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | LLM backend. Get one at [console.groq.com/keys](https://console.groq.com/keys) |
| `JWT_SECRET_KEY` | Signs auth tokens. **Empty is not safe** — an empty HMAC key still produces valid, forgeable tokens, so the app refuses to boot outside development without one ≥32 chars, and generates a random ephemeral key in development |
| `POSTGRES_PASSWORD` | Postgres password (docker-compose). No default, so a deployment cannot inherit a well-known one |

### Optional — safe defaults in `app/config.py`

| Variable | Default | Purpose |
|---|---|---|
| `ENV` | `development` | `development` / `staging` / `production`. Gates the fail-fast checks on `JWT_SECRET_KEY` and LangSmith |
| `LANGSMITH_API_KEY` / `LANGSMITH_TRACING` / `LANGSMITH_PROJECT` / `LANGSMITH_ENDPOINT` | — | Tracing. Optional in dev/staging, **all three of key/tracing/project required when `ENV=production`** — the app refuses to boot untraced. The API key alone is a no-op without `LANGSMITH_TRACING=true` |
| `DATABASE_URL` | `sqlite:///./product_query_agent.db` | Postgres is used via psycopg3. `postgres://` (what Render and Heroku hand out) and bare `postgresql://` are both rewritten to `postgresql+psycopg://` on load, so a provider's URL can be pasted in unchanged |
| `REDIS_URL` | `redis://localhost:6379/0` | Response cache, usage tracker, and rate limiter |
| `BACKEND_URL` | `http://localhost:8000` | Where the Streamlit frontend reaches the backend |
| `POSTGRES_USER` / `POSTGRES_DB` | `app` / `product_query_agent` | Postgres bootstrap identity; read by docker-compose only, not the app |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Conversational model driving the agent |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature for the agent model |
| `LLM_INPUT_PRICE_PER_MILLION_USD` | `0.15` | Input rate used to estimate cost. **Model-specific — update whenever `LLM_MODEL` changes**, or the budget is enforced against the wrong prices |
| `LLM_OUTPUT_PRICE_PER_MILLION_USD` | `0.60` | Output rate, same caveat |
| `PROMPT_GUARD_MODEL` | `meta-llama/llama-prompt-guard-2-86m` | Jailbreak classifier screening user input. Uses a separate Groq quota from the chat model |
| `PROMPT_GUARD_ENABLED` | `true` | Set `false` to fall back to regex-only injection screening |
| `PROMPT_GUARD_THRESHOLD` | `0.5` | Score (0–1) above which input is treated as an injection attempt |
| `PROMPT_GUARD_TIMEOUT_SECONDS` | `5.0` | Classifier timeout; on timeout screening degrades rather than failing the request |
| `JWT_EXPIRE_MINUTES` | `30` | Access token lifetime |
| `MAX_RECURSION_LIMIT` | `30` | Caps agent graph supersteps per request. Measured: 14 for a one-tool-call turn, ~6 per extra call |
| `COST_BUDGET_USD_PER_USER_PER_DAY` | `0.0` | Daily per-user dollar cap. **Safe-by-default: `/chat` returns 402 until raised** |
| `MAX_LLM_CALLS_PER_USER_PER_DAY` | `30` | Daily per-user LLM call cap |
| `MAX_LLM_CALLS_PER_MINUTE_PER_USER` | `15` | Per-minute LLM call burst guard |
| `MAX_TOKENS_PER_USER_PER_DAY` | `30000` | Daily per-user token cap |
| `MAX_TOKENS_PER_MINUTE_PER_USER` | `4000` | Per-minute token burst guard. **Keep it below the provider's tokens-per-minute ceiling** (8,000 for `openai/gpt-oss-120b`) so this cap rejects first, with your error, instead of the provider returning 429. It is per user while the provider's is per account, so the value also decides how many users can be active in the same minute — 4,000 allows two |
| `RESPONSE_CACHE_TTL_SECONDS` | `300` | TTL for cached final answers (scoped per user) |
| `PRODUCT_CACHE_TTL_SECONDS` | `86400` | TTL for cached product/review rows; a stale row is still served if DummyJSON is unreachable |
| `CORS_ORIGINS` | `""` | Comma-separated **browser** origins allowed to call the backend. Empty is correct by default — the Streamlit frontend calls the API server-side, so no browser cross-origin request occurs. Setting `*` automatically disables credentialed CORS |

DummyJSON requires no API key.

**Three conventions worth knowing:**

- A variable you want to leave at its default should be **commented out**, not written as `KEY=`.
  pydantic-settings treats a present-but-empty value as an explicit empty string, which is a
  validation crash for numeric fields rather than a fallback to the default.
- `DATABASE_URL`, `REDIS_URL` and `BACKEND_URL` are the only values `docker-compose.yml`
  overrides, because inside the compose network services are reached by name rather than
  `localhost`. They are addresses, not secrets — credentials inside them come from `.env`.
- Changing `LLM_MODEL` is a one-line change, but re-run `uv run pytest tests/eval` afterwards and
  update the two price variables. Different models format numbers differently, which the
  groundedness and eval layers are sensitive to.

## Commands

| Command | Purpose |
|---|---|
| `uv sync` | Install/sync dependencies |
| `uv run main.py` | Run the dev CLI agent |
| `uv run uvicorn app.main:app --reload` | Run the backend |
| `uv run streamlit run frontend/streamlit_app.py` | Run the frontend |
| `uv run alembic upgrade head` | Apply database migrations |
| `uv run pytest tests/unit tests/integration` | Run tests (no LLM quota; integration needs a local Redis) |
| `uv run pytest tests/eval` | Run the golden-dataset eval (**real Groq calls**) |
| `uv run python -m scripts.benchmark` | Measure tokens/tool calls per query (**real Groq calls**) |
| `uv run langgraph dev --allow-blocking` | Visualise the agent graph in LangGraph Studio |
| `uv run ruff check . && uv run ruff format .` | Lint and format |
| `uv add --group backend <pkg>` | Add a backend dependency (see below) |

## Development

### Tests

Three suites, deliberately separated by what they cost to run:

- `tests/unit` — pure logic, `fakeredis`, in-memory SQLite. No network, no LLM quota.
- `tests/integration` — FastAPI routes through `TestClient` with a fake agent. Needs a local Redis
  for the real rate limiter; no LLM quota.
- `tests/eval` — the 12-case golden dataset against real Groq and real DummyJSON. This is the CI
  gate, and the only suite that spends quota. Run `alembic upgrade head` first.

### Dependency groups

`[project.dependencies]` holds **only** what both sides import. Everything else lives in a
`backend`, `frontend` or `dev` dependency group, and each Dockerfile installs one side
(`uv sync --no-default-groups --group backend`) so neither image carries the other's stack. Local
and CI installs are unaffected — `[tool.uv] default-groups` includes all three.

This means **new dependencies need a group**: `uv add --group backend <pkg>`. A bare `uv add` lands
in the shared list and re-fattens both images; `tests/unit/test_dependency_layout.py` fails if it
happens.

### Graph visualisation

```bash
uv run langgraph dev --allow-blocking
```

Serves the graph at `http://127.0.0.1:2024` and opens LangGraph Studio. `--allow-blocking` is
required — without it the dev server's blocking-call guard makes the graph endpoint return 500,
which the browser reports only as a generic `Failed to fetch`. Add `--tunnel` if your browser
blocks requests from the hosted Studio UI to a plain-HTTP local server.

This is separate from LangSmith tracing: tracing records what *did* happen on a request; Studio
renders the graph's shape and lets you step through it.

## Deployment

CI/CD is GitHub Actions → GHCR → Render.

- **`ci.yml`** runs on every pull request and on pushes to `main`: `lint` (ruff check + format),
  `test` (unit + integration, with a Redis service), and `eval` (the golden dataset, **real Groq
  calls**). The eval job skips gracefully when `GROQ_API_KEY` is not configured, so a first-time
  fork or PR is not blocked by it.
- **`deploy.yml`** runs after CI succeeds on `main`. It builds both images, pushes them to GHCR
  tagged `:latest` and `:<sha>`, then calls Render's API to deploy the **`:<sha>` tag**. Render
  does not rebuild from source — it runs the exact artifact CI validated, and a rollback is
  redeploying an older tag.

### Required GitHub secrets

| Secret | Used by |
|---|---|
| `GROQ_API_KEY` | `ci.yml` eval job |
| `RENDER_API_KEY` | `deploy.yml` — Render account API key |
| `RENDER_SERVICE_ID_BACKEND` | `deploy.yml` — the `srv-…` id from the service's URL |
| `RENDER_SERVICE_ID_FRONTEND` | `deploy.yml` |

`GITHUB_TOKEN` is provided automatically and is what pushes to GHCR.

### Render setup

Create both services as **"Deploy an existing image"** (not from the repository), pointing at
`ghcr.io/<owner>/<repo>-backend` and `ghcr.io/<owner>/<repo>-frontend`. Render's API only allows
the *tag* to differ from a service's configured image, so the host, repository and image name must
match what `deploy.yml` pushes.

**GHCR packages are private by default**, and Render cannot pull them without credentials. Either
make the two packages public, or add a registry credential in Render (your GitHub username plus a
personal access token with `read:packages`). An image-pull failure on the first deploy is almost
always this.

Backend environment variables on Render: `GROQ_API_KEY`, `JWT_SECRET_KEY` (≥32 chars),
`DATABASE_URL`, `REDIS_URL`, `ENV=production`, all three `LANGSMITH_*`, and a non-zero
`COST_BUDGET_USD_PER_USER_PER_DAY`. Frontend needs only `BACKEND_URL`.

Two things that are already handled, so you do not need to work around them:

- **Ports.** Both images read `${PORT}` with a local default, so Render's assigned port is honoured.
- **Database URL.** Paste Render's Postgres URL unchanged — `postgres://` is normalised to
  `postgresql+psycopg://` on load.

Migrations run automatically: the backend image's entrypoint executes `alembic upgrade head`
before starting uvicorn.

## Project structure

```
app/            FastAPI backend — config, agent graph + middleware, API routes, auth, caches, DB
mcp_servers/    MCP server wrapping the DummyJSON catalogue
frontend/       Streamlit UI (pure HTTP client — never imports app.*)
cli/            Async dev REPL
alembic/        Database migrations
docker/         Backend and frontend Dockerfiles
scripts/        Token/cost benchmark
tests/          unit · integration · eval
```

## Notes and limitations

- **DummyJSON is demo data.** Prices, stock and reviews are not real. The UI says so.
- **The response cache does no quality control.** A poor first answer to a question is served to
  everyone asking it for `RESPONSE_CACHE_TTL_SECONDS`. That is the designed trade-off of caching
  whatever the first answer was.
- **Cost figures are estimates.** Token counts are the caps' ground truth; the dollar figure is
  derived from the configured per-million rates and drifts if provider pricing changes.
- Design notes, decision history and the phase-by-phase build plan are kept as internal working
  documents and are not published in this repository.

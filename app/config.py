"""Application settings, loaded from environment variables and `.env`.

`.env` is the single source of truth for secrets and tunables; see `.env.example` for the
annotated reference. Non-obvious values carry their reasoning inline.
"""

import logging
import secrets
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("app.config")

# Shorter HS256 keys are brute-forceable offline from a single captured token.
MIN_JWT_SECRET_LENGTH = 32


class Settings(BaseSettings):
    """Application configuration, read from the environment and `.env`."""

    # `extra="ignore"` rather than "forbid": the process environment legitimately contains
    # unrelated variables, and forbidding them would make the app refuse to boot on any host.
    # Every variable *this project* defines is declared below, so nothing of ours is silently
    # dropped — a test enforces that against `.env.example`.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Literal["development", "staging", "production"] = "development"

    # --- Credentials ---
    # SecretStr so a stray repr, log line, or traceback prints "**********" instead of the value.
    # Read them with .get_secret_value() at the point of use.
    groq_api_key: SecretStr = SecretStr("")
    jwt_secret_key: SecretStr = SecretStr("")
    jwt_expire_minutes: int = 30

    # --- Infrastructure ---
    database_url: str = "sqlite:///./product_query_agent.db"
    redis_url: str = "redis://localhost:6379/0"
    cors_origins: str = ""
    backend_url: str = "http://localhost:8000"

    # Postgres bootstrap identity. Only docker-compose consumes these — the application connects
    # via DATABASE_URL — but they are declared so `.env` has no variable Settings cannot account
    # for.
    postgres_user: str = "app"
    postgres_password: SecretStr = SecretStr("")
    postgres_db: str = "product_query_agent"

    # --- Models ---
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0
    # Per-million-token Groq rates for `llm_model`, used only to derive the cost estimate in
    # CostTrackingMiddleware. **Update these whenever `llm_model` changes** — rates are
    # model-specific, and a stale pair silently misprices `cost_budget_usd_per_user_per_day`.
    # Defaults are openai/gpt-oss-120b's published rates. Living here rather than in a
    # model-keyed table in the middleware keeps model identifiers out of call sites.
    llm_input_price_per_million_usd: float = 0.15
    llm_output_price_per_million_usd: float = 0.60
    # Meta's Prompt Guard 2 (86M) jailbreak classifier, used by the guardrails middleware. Billed
    # against a separate Groq quota from the chat model, so screening every turn does not consume
    # the conversational token budget.
    prompt_guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    prompt_guard_enabled: bool = True
    # The classifier returns a 0..1 probability; scores cluster near the extremes in practice.
    prompt_guard_threshold: float = 0.5
    prompt_guard_timeout_seconds: float = 5.0

    # --- Agent limits ---
    # Graph supersteps per request. Measured: 14 for a one-tool-call turn, ~6 per additional
    # sequential call, so this allows roughly three. Raise only alongside a measurement — a loose
    # limit lets a looping model burn tokens before it is stopped.
    max_recursion_limit: int = 30

    # --- Per-user usage caps: five limits, whichever trips first ---
    cost_budget_usd_per_user_per_day: float = 0.0
    max_llm_calls_per_user_per_day: int = 30
    max_llm_calls_per_minute_per_user: int = 15
    max_tokens_per_user_per_day: int = 30_000
    # Must stay below the provider's tokens-per-minute ceiling for `llm_model`, so this cap trips
    # first and the user gets our 429 rather than the provider's. openai/gpt-oss-120b allows 8k
    # TPM; 4k leaves room for two users to be active in the same minute without the account-wide
    # limit rejecting either. **Re-check whenever `llm_model` changes** — the ceiling is per-model
    # (llama-3.3-70b-versatile allowed 12k). Nothing tracks aggregate usage, so this division
    # among users is the only protection the account-wide limit gets.
    max_tokens_per_minute_per_user: int = 4_000

    # --- Caching ---
    response_cache_ttl_seconds: int = 300
    product_cache_ttl_seconds: int = 86_400

    # --- Observability ---
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_tracing: bool = False
    langsmith_project: str = ""
    langsmith_endpoint: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        """Allowed origins, parsed from the comma-separated setting."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def cors_allow_credentials(self) -> bool:
        """Whether to allow credentialed cross-origin requests.

        Always False when a wildcard origin is configured: `allow_origins=["*"]` together with
        credentials lets any site issue authenticated cross-site requests.
        """
        return "*" not in self.cors_origin_list

    @property
    def database_url_psycopg(self) -> str:
        """`database_url` with SQLAlchemy's driver suffix removed.

        `create_engine` needs `postgresql+psycopg://`, but the LangGraph Postgres checkpointer and
        store pass the string straight to `psycopg.connect()`, which rejects that suffix.
        """
        return self.database_url.replace("postgresql+psycopg://", "postgresql://")

    @model_validator(mode="after")
    def _normalize_database_url(self) -> "Settings":
        """Rewrite the Postgres URL forms hosting providers hand out into the driver form we use.

        Render and Heroku emit `postgres://`, a scheme SQLAlchemy removed support for outright, and
        a bare `postgresql://` resolves to psycopg2, which is not installed here. Both fail only at
        startup, long after the value was pasted into a dashboard, so they are corrected rather
        than left as a manual step for an operator to remember.
        """
        for prefix in ("postgres://", "postgresql://"):
            if self.database_url.startswith(prefix):
                self.database_url = f"postgresql+psycopg://{self.database_url[len(prefix) :]}"
                break
        return self

    @model_validator(mode="after")
    def _require_strong_jwt_secret(self) -> "Settings":
        """Reject the unsafe empty-secret state in every environment.

        An empty HMAC key still signs and verifies tokens, so a blank secret is forgeable rather
        than merely insecure.
        """
        secret = self.jwt_secret_key.get_secret_value()
        if not secret:
            if self.env == "development":
                self.jwt_secret_key = SecretStr(secrets.token_urlsafe(48))
                logger.warning(
                    "JWT_SECRET_KEY unset - generated an ephemeral development secret. Set it "
                    "explicitly for anything beyond local development."
                )
            else:
                raise ValueError(
                    f"JWT_SECRET_KEY must be set when ENV={self.env}. Generate one with: "
                    f'python -c "import secrets; print(secrets.token_urlsafe(48))"'
                )
        elif self.env != "development" and len(secret) < MIN_JWT_SECRET_LENGTH:
            raise ValueError(
                f"JWT_SECRET_KEY must be at least {MIN_JWT_SECRET_LENGTH} characters when "
                f"ENV={self.env} (got {len(secret)})."
            )
        return self

    @model_validator(mode="after")
    def _require_tracing_in_production(self) -> "Settings":
        """Refuse to boot a production deployment without tracing configured.

        Fail fast rather than warn: a warning lost in logs means an incident gets debugged with
        no trace history, which defeats the point of having tracing.
        """
        if self.env == "production" and not (
            self.langsmith_api_key.get_secret_value()
            and self.langsmith_tracing
            and self.langsmith_project
        ):
            raise ValueError(
                "LANGSMITH_API_KEY, LANGSMITH_TRACING, and LANGSMITH_PROJECT must all be set "
                "when ENV=production."
            )
        return self


settings = Settings()

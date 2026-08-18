import os

import pytest

from app.config import Settings

# A strong-enough value for tests that aren't about the JWT validator itself.
_VALID_SECRET = "x" * 48


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Strip every Settings-backed variable from the process environment.

    `_env_file=None` alone is NOT isolation: it stops pydantic-settings reading the *file*, but it
    still reads `os.environ` — which `tests/conftest.py`'s `load_dotenv()` has already populated
    from the developer's real `.env`. Without this fixture these tests silently assert whatever
    happens to be on the machine running them, which is how several of them passed for the wrong
    reason (and started failing the moment `.env` gained a `COST_BUDGET_...` entry)."""
    for key in list(os.environ):
        if key.lower() in Settings.model_fields:
            monkeypatch.delenv(key, raising=False)


def test_production_requires_langsmith_settings():
    with pytest.raises(ValueError):
        Settings(
            env="production",
            jwt_secret_key=_VALID_SECRET,
            langsmith_api_key="",
            langsmith_tracing=False,
            langsmith_project="",
            _env_file=None,
        )


def test_production_boots_with_full_langsmith_config():
    s = Settings(
        env="production",
        jwt_secret_key=_VALID_SECRET,
        langsmith_api_key="key",
        langsmith_tracing=True,
        langsmith_project="proj",
        _env_file=None,
    )
    assert s.env == "production"


def test_development_boots_without_langsmith_config():
    s = Settings(env="development", _env_file=None)
    assert s.env == "development"


def test_staging_boots_without_langsmith_config():
    s = Settings(env="staging", jwt_secret_key=_VALID_SECRET, _env_file=None)
    assert s.env == "staging"


def test_cors_origin_list_parses_and_strips_comma_separated_values():
    s = Settings(cors_origins="http://a.com, http://b.com,,", _env_file=None)
    assert s.cors_origin_list == ["http://a.com", "http://b.com"]


def test_cors_origin_list_empty_by_default():
    s = Settings(_env_file=None)
    assert s.cors_origin_list == []


def test_database_url_psycopg_strips_sqlalchemy_driver_suffix():
    """psycopg.connect() (used directly by PostgresSaver/PostgresStore.from_conn_string) doesn't
    understand SQLAlchemy's `+psycopg` dialect+driver suffix — found when first running against
    real Postgres in Phase 6; see app/db/checkpointer.py and app/agent/store.py."""
    s = Settings(database_url="postgresql+psycopg://user:pw@host:5432/dbname", _env_file=None)
    assert s.database_url_psycopg == "postgresql://user:pw@host:5432/dbname"


def test_database_url_psycopg_leaves_sqlite_url_unchanged():
    s = Settings(database_url="sqlite:///./product_query_agent.db", _env_file=None)
    assert s.database_url_psycopg == "sqlite:///./product_query_agent.db"


def test_render_style_postgres_scheme_is_normalized():
    """Render and Heroku hand out `postgres://`, which SQLAlchemy removed support for outright."""
    s = Settings(database_url="postgres://user:pw@host:5432/dbname", _env_file=None)
    assert s.database_url == "postgresql+psycopg://user:pw@host:5432/dbname"


def test_bare_postgresql_scheme_is_normalized():
    """A bare `postgresql://` resolves to psycopg2, which is not a dependency of this project."""
    s = Settings(database_url="postgresql://user:pw@host:5432/dbname", _env_file=None)
    assert s.database_url == "postgresql+psycopg://user:pw@host:5432/dbname"


def test_already_correct_postgres_url_is_untouched():
    s = Settings(database_url="postgresql+psycopg://user:pw@host:5432/dbname", _env_file=None)
    assert s.database_url == "postgresql+psycopg://user:pw@host:5432/dbname"


def test_sqlite_url_is_not_rewritten():
    s = Settings(database_url="sqlite:///./product_query_agent.db", _env_file=None)
    assert s.database_url == "sqlite:///./product_query_agent.db"


def test_normalized_url_still_yields_a_usable_psycopg_form():
    """The two rewrites compose: Render's URL must reach psycopg.connect() in a form it accepts."""
    s = Settings(database_url="postgres://user:pw@host:5432/dbname", _env_file=None)
    assert s.database_url_psycopg == "postgresql://user:pw@host:5432/dbname"


def test_normalized_url_is_detected_as_postgres_by_the_backends():
    """checkpointer.py and store.py branch on this prefix to pick async Postgres over in-memory."""
    s = Settings(database_url="postgres://user:pw@host:5432/dbname", _env_file=None)
    assert s.database_url.startswith("postgresql")


def _prod(**kwargs) -> dict:
    """Baseline kwargs for a production Settings (tracing vars satisfied) so JWT-specific
    assertions aren't masked by the unrelated tracing validator."""
    return {
        "env": "production",
        "langsmith_api_key": "k",
        "langsmith_tracing": True,
        "langsmith_project": "p",
        "_env_file": None,
        **kwargs,
    }


def test_empty_jwt_secret_gets_random_value_in_development():
    """An empty HMAC key still signs and verifies tokens, so a blank secret means anyone can forge
    a token for any user. Development gets a random per-process key instead of the empty string."""
    s = Settings(jwt_secret_key="", _env_file=None)
    generated = s.jwt_secret_key.get_secret_value()
    assert generated != ""
    assert len(generated) >= 32
    other = Settings(jwt_secret_key="", _env_file=None).jwt_secret_key.get_secret_value()
    assert other != generated


def test_empty_jwt_secret_refuses_to_boot_outside_development():
    for env in ("staging", "production"):
        with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
            Settings(**_prod(env=env, jwt_secret_key=""))


def test_short_jwt_secret_refuses_to_boot_outside_development():
    with pytest.raises(ValueError, match="at least"):
        Settings(**_prod(jwt_secret_key="tooshort"))


def test_strong_jwt_secret_boots_in_production():
    s = Settings(**_prod(jwt_secret_key="x" * 48))
    assert s.jwt_secret_key.get_secret_value() == "x" * 48


def test_cors_credentials_disabled_when_origin_is_wildcard():
    """`allow_origins=["*"]` with `allow_credentials=True` lets any site make authenticated
    cross-origin requests; the combination must be unreachable via config."""
    assert Settings(cors_origins="*", _env_file=None).cors_allow_credentials is False
    assert Settings(cors_origins="http://a.com,*", _env_file=None).cors_allow_credentials is False


def test_cors_credentials_enabled_for_explicit_origins():
    assert Settings(cors_origins="http://a.com", _env_file=None).cors_allow_credentials is True


def test_shipped_default_budget_is_zero_dollars():
    """The safe-by-default cap: a fresh deployment serves only cached answers until an operator
    raises it. `_env_file=None` matters — without it this reads whatever the local .env sets and
    would assert nothing about the shipped default."""
    assert Settings(_env_file=None).cost_budget_usd_per_user_per_day == 0.0


def test_secret_values_never_appear_in_repr_or_str():
    """SecretStr keeps credentials out of tracebacks, log lines and debug dumps — a full Settings
    object is printed in exactly those places."""
    s = Settings(
        groq_api_key="gsk_live_value",
        jwt_secret_key="j" * 48,
        postgres_password="pg_live_value",
        langsmith_api_key="lsv2_live_value",
        _env_file=None,
    )
    for rendered in (repr(s), str(s), f"{s}"):
        for secret in ("gsk_live_value", "pg_live_value", "lsv2_live_value", "j" * 48):
            assert secret not in rendered


def test_every_env_example_variable_is_a_settings_field():
    """`.env` must have no variable Settings cannot account for: `extra="ignore"` would otherwise
    drop it silently, which is how LANGSMITH_ENDPOINT went unread."""
    import re
    from pathlib import Path

    declared = set(Settings.model_fields)
    documented = {
        m.lower()
        for m in re.findall(
            r"^#?\s*([A-Z][A-Z0-9_]+)=", Path(".env.example").read_text(encoding="utf-8"), re.M
        )
    }
    assert not (documented - declared), (
        f"documented but not a Settings field: {documented - declared}"
    )

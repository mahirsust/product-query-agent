"""Frontend configuration.

Reads the same `.env` as the backend, so configuration has one source of truth. Uses its own
settings class rather than importing `app.config`: `frontend/` must stay a pure HTTP client with
no application imports, and `BACKEND_URL` is the only value it needs.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    """Where the Streamlit app reaches the API."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Defaults to a host-run backend; docker-compose overrides it with the in-network service name.
    backend_url: str = "http://localhost:8000"

    @field_validator("backend_url")
    @classmethod
    def _strip_trailing_slash(cls, url: str) -> str:
        """Avoid double slashes when paths are appended."""
        return url.rstrip("/")


settings = FrontendSettings()

# Kept as a module-level name so callers read one obvious value rather than the settings object.
BACKEND_URL = settings.backend_url

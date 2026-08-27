"""The frontend must never surface a platform error page to a user.

A gateway 502 returns a full HTML document rather than the backend's JSON. Falling back to the
raw body put that document on screen during a Render cold start.
"""

import sys
from pathlib import Path

import pytest

# frontend/ has no package prefix — Streamlit puts the script's own directory on sys.path, and
# `import api_client` relies on that. Mirror it here rather than adding an __init__.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

import api_client  # noqa: E402


class _Response:
    def __init__(self, status_code, payload=None, text="", ok=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.ok = (200 <= status_code < 300) if ok is None else ok

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


def test_html_error_page_is_not_surfaced():
    html = "<!DOCTYPE html><html><head><style>@font-face{src:url(data:font/woff2;base64,AAAA)}"
    with pytest.raises(api_client.ApiError) as exc:
        api_client._handle_response(_Response(502, text=html))
    assert "<" not in exc.value.detail
    assert "still starting up" in exc.value.detail
    assert exc.value.status_code == 502


def test_backend_json_detail_is_preserved():
    """The application's own messages are user-facing and must pass through untouched."""
    with pytest.raises(api_client.ApiError) as exc:
        api_client._handle_response(
            _Response(503, payload={"detail": "The service has reached its daily capacity."})
        )
    assert exc.value.detail == "The service has reached its daily capacity."


def test_non_gateway_non_json_gets_a_status_message():
    with pytest.raises(api_client.ApiError) as exc:
        api_client._handle_response(_Response(418, text="<html>teapot</html>"))
    assert exc.value.detail == "Unexpected response from the backend (HTTP 418)."


def test_empty_json_detail_falls_back():
    with pytest.raises(api_client.ApiError) as exc:
        api_client._handle_response(_Response(502, payload={}))
    assert "still starting up" in exc.value.detail


def test_successful_response_is_returned():
    assert api_client._handle_response(_Response(200, payload={"answer": "hi"})) == {"answer": "hi"}


def test_auth_timeout_survives_a_cold_start():
    """52s measured on Render's free tier; a 10s timeout failed every cold start."""
    assert api_client._AUTH_TIMEOUT_SECONDS >= 60


def test_slowapi_rate_limit_message_is_preserved():
    """slowapi answers with {"error": ...}, not FastAPI's {"detail": ...}. Reading only "detail"
    turned a clear rate-limit message into the generic fallback."""
    with pytest.raises(api_client.ApiError) as exc:
        api_client._handle_response(
            _Response(429, payload={"error": "Rate limit exceeded: 5 per 1 minute"})
        )
    assert exc.value.detail == "Rate limit exceeded: 5 per 1 minute"
    assert exc.value.status_code == 429


def test_detail_wins_when_both_keys_are_present():
    with pytest.raises(api_client.ApiError) as exc:
        api_client._handle_response(
            _Response(400, payload={"detail": "from app", "error": "other"})
        )
    assert exc.value.detail == "from app"

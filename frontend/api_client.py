"""HTTP client for the backend API.

Deliberately free of `app.*` imports so the frontend stays a pure API consumer with no shared
in-process state.
"""

import requests
from config import BACKEND_URL

# Generous even for a trivial request: on a platform that spins idle services down, the first call
# after a quiet period pays for the whole wake-up. Measured at 52s on Render's free tier, where a
# 10s timeout failed every cold start before the backend could answer.
_AUTH_TIMEOUT_SECONDS = 90
# A tool-using turn can involve several model calls plus retries, on top of any wake-up.
_CHAT_TIMEOUT_SECONDS = 120

# Emitted by the platform's router, not the application, so the body is an HTML error page.
_GATEWAY_STATUSES = frozenset({502, 503, 504})


class ApiError(Exception):
    """A non-2xx response, carrying the backend's status code and detail message."""

    def __init__(self, status_code: int, detail: str):
        """Capture the status and message so the UI can react per status code."""
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


def _fallback_detail(status_code: int) -> str:
    """A message safe to show a user when the body carries no usable one."""
    if status_code in _GATEWAY_STATUSES:
        return "The backend is unavailable or still starting up. Please try again in a moment."
    return f"Unexpected response from the backend (HTTP {status_code})."


def _handle_response(response: requests.Response) -> dict:
    """Return the decoded body, or raise `ApiError` describing the failure.

    A failure does not necessarily come from the application: a platform gateway answering 502
    returns a full HTML error page. Falling back to `response.text` put that entire document in
    the UI, so a non-JSON body is replaced with a short message instead.
    """
    if response.ok:
        return response.json()
    try:
        detail = response.json().get("detail") or _fallback_detail(response.status_code)
    except ValueError:
        detail = _fallback_detail(response.status_code)
    raise ApiError(response.status_code, detail)


def signup(username: str, password: str) -> dict:
    """Create an account. Returns a token, so no separate login call is needed."""
    response = requests.post(
        f"{BACKEND_URL}/auth/signup",
        json={"username": username, "password": password},
        timeout=_AUTH_TIMEOUT_SECONDS,
    )
    return _handle_response(response)


def login(username: str, password: str) -> dict:
    """Exchange credentials for an access token."""
    response = requests.post(
        f"{BACKEND_URL}/auth/login",
        json={"username": username, "password": password},
        timeout=_AUTH_TIMEOUT_SECONDS,
    )
    return _handle_response(response)


def post_chat(token: str, question: str, thread_id: str) -> dict:
    """Send a question to the agent within a conversation thread."""
    response = requests.post(
        f"{BACKEND_URL}/chat",
        json={"question": question, "thread_id": thread_id},
        headers={"Authorization": f"Bearer {token}"},
        timeout=_CHAT_TIMEOUT_SECONDS,
    )
    return _handle_response(response)

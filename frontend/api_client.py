"""HTTP client for the backend API.

Deliberately free of `app.*` imports so the frontend stays a pure API consumer with no shared
in-process state.
"""

import requests
from config import BACKEND_URL

_AUTH_TIMEOUT_SECONDS = 10
# A tool-using turn can involve several model calls plus retries, so this is far longer than the
# auth endpoints need.
_CHAT_TIMEOUT_SECONDS = 90


class ApiError(Exception):
    """A non-2xx response, carrying the backend's status code and detail message."""

    def __init__(self, status_code: int, detail: str):
        """Capture the status and message so the UI can react per status code."""
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


def _handle_response(response: requests.Response) -> dict:
    """Return the decoded body, or raise `ApiError` describing the failure."""
    if response.ok:
        return response.json()
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
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

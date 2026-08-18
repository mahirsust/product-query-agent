"""Request and response bodies.

FastAPI validates against these before a route handler runs, so malformed input is rejected
with a 422 rather than reaching application logic.
"""

from pydantic import BaseModel, Field, field_validator


class SignupRequest(BaseModel):
    """New account details."""

    username: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1)

    @field_validator("password")
    @classmethod
    def _password_within_bcrypt_limit(cls, password: str) -> str:
        """Enforce bcrypt's length ceiling, which is 72 *bytes* rather than 72 characters.

        Non-ASCII passwords can exceed it well before 72 characters, and bcrypt raises instead of
        truncating, so a character-count check would under-validate.
        """
        if len(password.encode("utf-8")) > 72:
            raise ValueError("password must be at most 72 bytes")
        return password


class LoginRequest(BaseModel):
    """Credentials presented to /auth/login."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """Bearer token issued by signup and login alike."""

    access_token: str
    token_type: str = "bearer"


class ChatRequest(BaseModel):
    """A question and the conversation it belongs to.

    Carries no user identity: that is derived server-side from the bearer token, never trusted
    from the body.
    """

    question: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """The agent's reply, plus the tools used on this turn only."""

    answer: str
    thread_id: str
    tool_calls: list[str]

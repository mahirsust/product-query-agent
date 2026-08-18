"""HTTP-level rate limiting, keyed by authenticated user."""

from fastapi import Request
from jose import JWTError
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.security.auth import decode_access_token


def rate_limit_key(request: Request) -> str:
    """Return the rate-limit bucket for a request.

    Buckets by user id when a valid token is present, so one user cannot exhaust another's
    quota, and by client IP otherwise (unauthenticated routes).
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[len("bearer ") :]
        try:
            user_id = decode_access_token(token)
            return f"user:{user_id}"
        except JWTError:
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key, storage_uri=settings.redis_url)

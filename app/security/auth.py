"""Password hashing and JWT issue/verification.

The only module permitted to hash or verify passwords or decode a token; everything downstream
receives a resolved `User`.
"""

from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import get_session
from app.db.models import User
from app.db.repository import get_user_by_id

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_ALGORITHM = "HS256"
_bearer_scheme = HTTPBearer(auto_error=False)

_credentials_error = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def hash_password(password: str) -> str:
    """Hash a password for storage. Never store or log the plaintext."""
    return _pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Check a password against its stored hash."""
    return _pwd_context.verify(password, hashed_password)


# Hash of a value no account can hold, used to equalize login timing.
_DUMMY_HASH = _pwd_context.hash("not-a-real-password-timing-equalizer")


def verify_password_dummy() -> None:
    """Perform the same bcrypt work a real verification would and discard the result.

    Called on the unknown-username path so login latency does not reveal whether an account
    exists.
    """
    _pwd_context.verify("x", _DUMMY_HASH)


def create_access_token(user_id: int) -> str:
    """Issue a signed token identifying `user_id`."""
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key.get_secret_value(), algorithm=_ALGORITHM)


def decode_access_token(token: str) -> int:
    """Return the user id encoded in `token`.

    Raises:
        JWTError: The token is malformed, expired, or incorrectly signed.
    """
    payload = jwt.decode(token, settings.jwt_secret_key.get_secret_value(), algorithms=[_ALGORITHM])
    return int(payload["sub"])


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    session: Session = Depends(get_session),
) -> User:
    """Resolve the authenticated user from the bearer token, or raise 401."""
    if credentials is None:
        raise _credentials_error
    try:
        user_id = decode_access_token(credentials.credentials)
    except JWTError:
        raise _credentials_error from None

    user = get_user_by_id(session, user_id)
    if user is None:
        raise _credentials_error
    return user

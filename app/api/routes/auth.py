"""Signup and login endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.api.schemas import LoginRequest, SignupRequest, TokenResponse
from app.db.repository import create_user, get_user_by_username
from app.security.auth import (
    create_access_token,
    hash_password,
    verify_password,
    verify_password_dummy,
)
from app.security.rate_limit import limiter

router = APIRouter(prefix="/auth", tags=["auth"])

_username_taken_error = HTTPException(
    status_code=status.HTTP_409_CONFLICT, detail="Username already taken"
)


@router.post("/signup", response_model=TokenResponse)
@limiter.limit("5/minute")
def signup(
    request: Request, payload: SignupRequest, session: Session = Depends(get_db_session)
) -> TokenResponse:
    """Create an account and return a token, so no separate login call is needed.

    Duplicate usernames are rejected twice over: a pre-check, and the unique constraint caught on
    commit. Two concurrent signups can both pass the pre-check, so the constraint is the real
    guard and the catch turns the loser's crash into the same 409.
    """
    if get_user_by_username(session, payload.username) is not None:
        raise _username_taken_error

    user = create_user(session, payload.username, hash_password(payload.password))
    try:
        session.commit()
    except IntegrityError:
        # two concurrent signups for the same username can both pass the check above before
        # either commits — the unique constraint is the real guard, this just turns the loser's
        # crash into the same 409 the pre-check gives everyone else.
        session.rollback()
        raise _username_taken_error from None
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(
    request: Request, payload: LoginRequest, session: Session = Depends(get_db_session)
) -> TokenResponse:
    """Exchange credentials for a token.

    Unknown username and wrong password are indistinguishable to the caller: identical 401s, and
    equal timing via `verify_password_dummy`.
    """
    user = get_user_by_username(session, payload.username)
    if user is None:
        # Do the bcrypt work anyway so an unknown username takes as long as a known one — see
        # verify_password_dummy(). Short-circuiting here would leak account existence via timing.
        verify_password_dummy()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    return TokenResponse(access_token=create_access_token(user.id))

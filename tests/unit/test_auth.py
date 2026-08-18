from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import JWTError, jwt

from app.config import settings
from app.db.repository import create_user
from app.security import auth


def test_password_hash_and_verify_roundtrip():
    hashed = auth.hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert auth.verify_password("correct horse battery staple", hashed)


def test_verify_password_rejects_wrong_password():
    hashed = auth.hash_password("correct horse battery staple")
    assert not auth.verify_password("wrong password", hashed)


def test_create_and_decode_access_token_roundtrip():
    token = auth.create_access_token(user_id=42)
    assert auth.decode_access_token(token) == 42


def test_decode_access_token_invalid_raises():
    with pytest.raises(JWTError):
        auth.decode_access_token("not-a-real-token")


def test_decode_access_token_expired_raises():
    payload = {"sub": "1", "exp": datetime.now(UTC) - timedelta(minutes=1)}
    expired_token = jwt.encode(
        payload, settings.jwt_secret_key.get_secret_value(), algorithm="HS256"
    )
    with pytest.raises(JWTError):
        auth.decode_access_token(expired_token)


def test_get_current_user_missing_credentials_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        auth.get_current_user(credentials=None, session=None)
    assert exc_info.value.status_code == 401


def test_get_current_user_valid_token_returns_user(db_session):
    user = create_user(db_session, "alice", auth.hash_password("pw"))
    db_session.commit()
    token = auth.create_access_token(user.id)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    result = auth.get_current_user(credentials=creds, session=db_session)
    assert result.username == "alice"


def test_get_current_user_unknown_user_id_raises_401(db_session):
    token = auth.create_access_token(user_id=999_999)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as exc_info:
        auth.get_current_user(credentials=creds, session=db_session)
    assert exc_info.value.status_code == 401


def test_get_current_user_invalid_token_raises_401(db_session):
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="garbage")
    with pytest.raises(HTTPException) as exc_info:
        auth.get_current_user(credentials=creds, session=db_session)
    assert exc_info.value.status_code == 401

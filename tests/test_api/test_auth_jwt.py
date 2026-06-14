"""JWT auth tests for admin/user token classification."""

import uuid

import jwt
import pytest
from fastapi import HTTPException

from graph_core.api.auth import ADMIN_SCOPE, USER_SCOPE, resolve_bearer_identity
from graph_core.config import settings


def _encode_token(claims: dict) -> str:
    return jwt.encode(claims, settings.jwt_secret, algorithm="HS256")


def test_resolve_bearer_identity_user_jwt(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", "jwt-test-secret-32-bytes-minimum!!")
    namespace_id = uuid.uuid4()
    token = _encode_token(
        {
            "sub": "user-123",
            "namespace_id": str(namespace_id),
            "scope": USER_SCOPE,
        }
    )
    identity = resolve_bearer_identity(f"Bearer {token}")
    assert identity.kind == "user"
    assert identity.namespace_id == namespace_id


def test_resolve_bearer_identity_admin_jwt(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", "jwt-test-secret-32-bytes-minimum!!")
    token = _encode_token(
        {
            "sub": "admin-123",
            "scope": ADMIN_SCOPE,
            "token_type": "admin",
        }
    )
    identity = resolve_bearer_identity(f"Bearer {token}")
    assert identity.kind == "admin"


def test_resolve_bearer_identity_rejects_jwt_without_scope(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", "jwt-test-secret-32-bytes-minimum!!")
    token = _encode_token({"sub": "missing-scope"})
    with pytest.raises(HTTPException):
        resolve_bearer_identity(f"Bearer {token}")

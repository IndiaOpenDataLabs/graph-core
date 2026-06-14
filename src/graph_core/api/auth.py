"""Auth dependency — resolves namespace context from request headers.

Handles namespace API keys, legacy X-Namespace-ID, and JWT bearer tokens.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import jwt
from fastapi import Header, HTTPException, Request
from jwt import InvalidTokenError

from graph_core.database import AsyncSessionLocal, current_namespace_id
from graph_core.models.namespace import Namespace
from graph_core.config import settings
from graph_core.models.namespace import Namespace
from graph_core.services import auth_service

logger = logging.getLogger(__name__)

ADMIN_SCOPE = "graph-core:admin"
USER_SCOPE = "graph-core:user"


@dataclass(frozen=True)
class BearerIdentity:
    kind: Literal["admin", "user"]
    namespace_id: uuid.UUID | None = None
    claims: dict[str, Any] | None = None


@dataclass
class AuthContext:
    namespace_id: uuid.UUID | None
    is_admin: bool
    token_kind: Literal["admin", "user", "legacy"]
    namespace: Namespace | None = None
    claims: dict[str, Any] | None = None


def _scope_values(claims: dict[str, Any]) -> set[str]:
    raw = claims.get("scope") or claims.get("scp") or claims.get("scopes") or []
    if isinstance(raw, str):
        return {scope for scope in raw.split() if scope}
    if isinstance(raw, list | tuple | set):
        return {str(scope) for scope in raw if str(scope)}
    return set()


def _namespace_id_from_claims(claims: dict[str, Any]) -> uuid.UUID | None:
    for key in ("namespace_id", "namespace", "ns_id"):
        value = claims.get(key)
        if value is None or value == "":
            continue
        try:
            return uuid.UUID(str(value))
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid namespace UUID in JWT claim {key}: {value}",
            ) from exc
    return None


def _decode_jwt(token: str) -> dict[str, Any]:
    if not settings.jwt_secret:
        raise HTTPException(
            status_code=401,
            detail="JWT authentication is not configured on this server",
        )
    options = {"require": []}
    kwargs: dict[str, Any] = {
        "algorithms": ["HS256"],
        "options": options,
    }
    if settings.jwt_issuer:
        kwargs["issuer"] = settings.jwt_issuer
    if settings.jwt_audience:
        kwargs["audience"] = settings.jwt_audience
    try:
        payload = jwt.decode(token, settings.jwt_secret, **kwargs)
    except InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid JWT bearer token") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=401, detail="Invalid JWT payload")
    return payload


def resolve_bearer_identity(authorization: str | None) -> BearerIdentity:
    """Resolve the token kind from an Authorization header.

    This is shared by the REST dependencies and the MCP gateway.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer":
        raise HTTPException(status_code=401, detail="Authorization header required")

    token = parts[1]

    if token.startswith("ns_key_"):
        return BearerIdentity(kind="user")

    claims = _decode_jwt(token)
    scopes = _scope_values(claims)
    token_type = str(claims.get("token_type") or claims.get("kind") or claims.get("role") or "").lower()
    namespace_id = _namespace_id_from_claims(claims)

    if token_type == "admin" or ADMIN_SCOPE in scopes:
        return BearerIdentity(kind="admin", claims=claims)
    if token_type == "user" or USER_SCOPE in scopes or namespace_id is not None:
        return BearerIdentity(kind="user", namespace_id=namespace_id, claims=claims)

    raise HTTPException(
        status_code=401,
        detail="JWT must declare graph-core:admin or graph-core:user scope",
    )


async def get_auth_context(
    request: Request,
    authorization: str | None = Header(default=None),
    x_namespace_id: str = Header(default=""),
) -> AuthContext:
    """Resolve auth context from request headers.

    Priority:
    1. Authorization: Bearer <ns_key_...>          → namespace-scoped
    2. Authorization: Bearer <JWT with scopes>     → admin or namespace-scoped
    3. X-Namespace-ID: <uuid>                      → legacy (deprecated)
    """
    del request

    # 1. Authorization: Bearer <...>
    if authorization:
        identity = resolve_bearer_identity(authorization)

        if identity.kind == "admin":
            return AuthContext(
                namespace_id=None,
                is_admin=True,
                token_kind="admin",
                claims=identity.claims,
            )

        token = authorization.split(" ", 1)[1]
        if token.startswith("ns_key_"):
            async with AsyncSessionLocal() as session:
                ns = await auth_service.verify_namespace_api_key(session, token)
            if ns:
                current_namespace_id.set(ns.id)
                return AuthContext(
                    namespace_id=ns.id,
                    is_admin=False,
                    token_kind="user",
                    namespace=ns,
                )
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        namespace_id = identity.namespace_id
        if namespace_id is None:
            raise HTTPException(
                status_code=400,
                detail="JWT user tokens must include a namespace_id claim",
            )
        current_namespace_id.set(namespace_id)
        async with AsyncSessionLocal() as session:
            ns = await session.get(Namespace, namespace_id)
        if ns is None:
            raise HTTPException(status_code=404, detail=f"Namespace {namespace_id} not found")
        return AuthContext(
            namespace_id=namespace_id,
            is_admin=False,
            token_kind="user",
            namespace=ns,
            claims=identity.claims,
        )

    # 2. Legacy X-Namespace-ID header (deprecated, self-hosted only)
    if x_namespace_id:
        logger.warning("X-Namespace-ID header is deprecated; use Authorization: Bearer <ns_key>")
        try:
            ns_id = uuid.UUID(x_namespace_id)
            current_namespace_id.set(ns_id)
            return AuthContext(namespace_id=ns_id, is_admin=False, token_kind="legacy")
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid UUID: {x_namespace_id}")

    raise HTTPException(status_code=401, detail="Authorization header required")


async def get_namespace_id(
    request: Request,
    authorization: str | None = Header(default=None),
    x_namespace_id: str = Header(default=""),
) -> uuid.UUID:
    """Backward-compatible dependency — returns namespace_id from any auth method.

    This wraps get_auth_context to maintain compatibility with existing endpoints
    that expect a plain uuid.UUID return type.
    """
    ctx = await get_auth_context(
        request,
        authorization=authorization,
        x_namespace_id=x_namespace_id,
    )
    if ctx.is_admin:
        raise HTTPException(status_code=400, detail="Namespace required — use /platform/namespaces endpoint")
    if ctx.namespace_id is None:
        raise HTTPException(status_code=401, detail="Namespace required")
    return ctx.namespace_id

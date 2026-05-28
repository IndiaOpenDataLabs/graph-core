"""Auth dependency — resolves namespace context from request headers.

Handles both self-hosted (admin key, namespace API key) and legacy
(X-Namespace-ID header) authentication. Multi-tenant JWT support is
added in Phase 2.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request

from graph_core.database import AsyncSessionLocal, current_namespace_id
from graph_core.models.namespace import Namespace
from graph_core.services import auth_service

logger = logging.getLogger(__name__)


@dataclass
class AuthContext:
    namespace_id: uuid.UUID
    is_admin: bool
    namespace: Namespace | None = None


async def get_auth_context(
    request: Request,
    authorization: str | None = Header(default=None),
    x_namespace_id: str = Header(default=""),
) -> AuthContext:
    """Resolve auth context from request headers.

    Priority:
    1. Authorization: Bearer <admin_key>  → admin (all namespaces)
    2. Authorization: Bearer <ns_key_...> → namespace-scoped
    3. X-Namespace-ID: <uuid>            → legacy (deprecated)
    """
    # 1. Check Authorization: Bearer header
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0] == "Bearer":
            token = parts[1]

            # Check admin key
            if auth_service.is_admin_key(token):
                # Admin gets broad access — no single namespace forced yet.
                return AuthContext(
                    namespace_id=uuid.uuid4(),  # placeholder, RLS bypassed for admin
                    is_admin=True,
                )

            # Check namespace API key (self-hosted)
            if token.startswith("ns_key_"):
                async with AsyncSessionLocal() as session:
                    ns = await auth_service.verify_namespace_api_key(session, token)
                if ns:
                    current_namespace_id.set(ns.id)
                    return AuthContext(
                        namespace_id=ns.id,
                        is_admin=False,
                        namespace=ns,
                    )

            raise HTTPException(status_code=401, detail="Invalid or expired token")

    # 2. Legacy X-Namespace-ID header (deprecated, self-hosted only)
    if x_namespace_id:
        logger.warning("X-Namespace-ID header is deprecated; use Authorization: Bearer <ns_key>")
        try:
            ns_id = uuid.UUID(x_namespace_id)
            current_namespace_id.set(ns_id)
            return AuthContext(namespace_id=ns_id, is_admin=False)
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
    ctx = await get_auth_context(request, authorization=authorization, x_namespace_id=x_namespace_id)
    if ctx.is_admin:
        raise HTTPException(status_code=400, detail="Namespace required — use /platform/namespaces endpoint")
    return ctx.namespace_id

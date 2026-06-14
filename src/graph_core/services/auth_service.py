"""Authentication service — namespace and JWT minting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from graph_core.config import settings
from graph_core.models.namespace import Namespace


def _jwt_payload(namespace_id: str, *, subject: str | None, expires_in_days: int) -> dict[str, object]:
    now = datetime.now(UTC)
    expires_at = now + timedelta(days=expires_in_days)
    payload: dict[str, object] = {
        "token_type": "user",
        "scope": "graph-core:user",
        "namespace_id": namespace_id,
        "sub": subject or f"namespace:{namespace_id}",
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    if settings.jwt_issuer:
        payload["iss"] = settings.jwt_issuer
    if settings.jwt_audience:
        payload["aud"] = settings.jwt_audience
    return payload


async def issue_namespace_user_token(
    session: AsyncSession,
    namespace_id: str,
    *,
    subject: str | None = None,
    expires_in_days: int = 365,
) -> tuple[Namespace, str, datetime] | None:
    """Issue a long-lived user JWT for a namespace."""
    from uuid import UUID

    ns = await session.get(Namespace, UUID(namespace_id))
    if ns is None:
        return None
    if not settings.jwt_secret:
        raise RuntimeError("JWT signing is not configured")
    now = datetime.now(UTC)
    expires_at = now + timedelta(days=expires_in_days)
    token = jwt.encode(
        _jwt_payload(str(ns.id), subject=subject, expires_in_days=expires_in_days),
        settings.jwt_secret,
        algorithm="HS256",
    )
    return ns, token, expires_at


async def create_namespace(
    session: AsyncSession,
    *,
    name: str,
    owner_app_id: str | None = None,
    owner_user_sub: str | None = None,
) -> Namespace:
    """Create a new namespace."""
    ns = Namespace(
        name=name,
        owner_app_id=owner_app_id,
        owner_user_sub=owner_user_sub,
    )
    session.add(ns)
    await session.flush()
    await session.refresh(ns)
    return ns


async def list_namespaces(session: AsyncSession) -> list[Namespace]:
    result = await session.execute(select(Namespace).order_by(Namespace.created_at.desc()))
    return list(result.scalars().all())


async def get_namespace_by_id(session: AsyncSession, namespace_id: str) -> Namespace | None:
    from uuid import UUID

    return await session.get(Namespace, UUID(namespace_id))

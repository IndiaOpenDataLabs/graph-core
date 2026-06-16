"""Authentication service — namespace and JWT minting."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from graph_core.config import settings
from graph_core.models.credential import Credential
from graph_core.models.namespace import Namespace
from graph_core.services.crypto import CredentialCrypto

_crypto = CredentialCrypto()


def _namespace_falkordb_graph_pattern(namespace_id: str) -> str:
    return f"tenant:{namespace_id}:*"


async def _provision_namespace_falkordb_acl(
    *,
    namespace_id: str,
    username: str,
    secret: str,
    base_url: str | None = None,
) -> None:
    from redis.asyncio import Redis

    redis_url = (base_url or settings.falkordb_url or "").strip()
    if not redis_url:
        raise RuntimeError("FalkorDB URL is not configured")
    if redis_url.startswith("falkordb://"):
        redis_url = "redis://" + redis_url[len("falkordb://") :]

    client = Redis.from_url(redis_url, decode_responses=True)
    try:
        await client.execute_command(
            "ACL",
            "SETUSER",
            username,
            "reset",
            "on",
            f">{secret}",
            "+AUTH",
            "+INFO",
            "+PING",
            "+GRAPH.LIST",
            "+GRAPH.QUERY",
            "+GRAPH.RO_QUERY",
            f"~{_namespace_falkordb_graph_pattern(namespace_id)}",
            f"%R~{_namespace_falkordb_graph_pattern(namespace_id)}",
            f"%W~{_namespace_falkordb_graph_pattern(namespace_id)}",
        )
    finally:
        await client.aclose()


def _namespace_falkordb_metadata(
    ns: Namespace,
    credential: Credential,
    username: str,
    base_url: str | None,
) -> dict[str, str | None]:
    return {
        "credential_id": str(credential.id),
        "username": username,
        "base_url": base_url or credential.base_url,
        "graph_pattern": _namespace_falkordb_graph_pattern(str(ns.id)),
    }


def _jwt_payload(
    namespace_id: str, *, subject: str | None, expires_in_days: int
) -> dict[str, object]:
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


async def provision_namespace_falkordb_credential(
    session: AsyncSession,
    namespace_id: str,
    *,
    username: str | None = None,
    secret: str | None = None,
    base_url: str | None = None,
) -> tuple[Namespace, Credential, str]:
    """Create or replace a namespace-scoped FalkorDB credential."""
    from uuid import UUID

    ns = await session.get(Namespace, UUID(namespace_id))
    if ns is None:
        raise ValueError(f"Namespace {namespace_id} not found")

    falkor_username = (username or f"ns_{ns.id.hex}").strip()
    if not falkor_username:
        raise ValueError("username must not be empty")
    falkor_secret = secret or secrets.token_urlsafe(32)

    credential = Credential(
        namespace_id=ns.id,
        provider="falkordb",
        label=falkor_username,
        encrypted_secret=_crypto.encrypt(falkor_secret),
        base_url=base_url,
    )
    session.add(credential)
    await session.flush()

    try:
        await _provision_namespace_falkordb_acl(
            namespace_id=str(ns.id),
            username=falkor_username,
            secret=falkor_secret,
            base_url=base_url,
        )
        metadata = dict(ns.metadata_json or {})
        metadata["falkordb"] = _namespace_falkordb_metadata(
            ns,
            credential,
            falkor_username,
            base_url,
        )
        ns.metadata_json = metadata
        await session.commit()
        await session.refresh(ns)
        await session.refresh(credential)
        return ns, credential, falkor_secret
    except Exception:
        await session.rollback()
        raise


async def ensure_namespace_falkordb_credential(
    session: AsyncSession,
    namespace_id: str,
    *,
    username: str | None = None,
    secret: str | None = None,
    base_url: str | None = None,
) -> tuple[Namespace, Credential, str | None]:
    """Ensure a namespace has a FalkorDB credential and metadata.

    Returns the existing credential when one is already provisioned. In that
    case the secret is ``None`` because it is not re-generated.
    """
    from uuid import UUID

    ns = await session.get(Namespace, UUID(namespace_id))
    if ns is None:
        raise ValueError(f"Namespace {namespace_id} not found")

    metadata = dict(ns.metadata_json or {})
    falkordb_meta = metadata.get("falkordb")
    existing_credential: Credential | None = None
    if isinstance(falkordb_meta, dict):
        credential_id = falkordb_meta.get("credential_id")
        if credential_id:
            existing_credential = await session.get(
                Credential, UUID(str(credential_id))
            )

    if existing_credential is None:
        result = await session.execute(
            select(Credential)
            .where(
                Credential.namespace_id == ns.id,
                Credential.provider == "falkordb",
            )
            .order_by(Credential.created_at.desc())
        )
        existing_credential = result.scalars().first()

    if existing_credential is not None:
        falkor_username = (
            username or existing_credential.label or f"ns_{ns.id.hex}"
        ).strip()
        if not falkor_username:
            raise ValueError("username must not be empty")
        secret = _crypto.decrypt(existing_credential.encrypted_secret)
        try:
            await _provision_namespace_falkordb_acl(
                namespace_id=str(ns.id),
                username=falkor_username,
                secret=secret,
                base_url=base_url or existing_credential.base_url,
            )
            metadata["falkordb"] = _namespace_falkordb_metadata(
                ns,
                existing_credential,
                falkor_username,
                base_url or existing_credential.base_url,
            )
            ns.metadata_json = metadata
            await session.commit()
            await session.refresh(ns)
            await session.refresh(existing_credential)
            return ns, existing_credential, None
        except Exception:
            await session.rollback()
            raise

    return await provision_namespace_falkordb_credential(
        session,
        namespace_id,
        username=username,
        secret=secret,
        base_url=base_url,
    )


async def resolve_namespace_falkordb_connection(
    session: AsyncSession,
    namespace_id: str,
) -> dict[str, str | int | bool] | None:
    """Resolve a namespace-scoped FalkorDB connection payload.

    Returns a dict compatible with Redis/FalkorDB connection kwargs or
    ``None`` when the namespace has no provisioned FalkorDB credential.
    """
    from uuid import UUID

    ns = await session.get(Namespace, UUID(namespace_id))
    if ns is None:
        return None
    metadata = ns.metadata_json or {}
    falkordb_meta = metadata.get("falkordb")

    credential: Credential | None = None
    if isinstance(falkordb_meta, dict):
        credential_id = falkordb_meta.get("credential_id")
        if credential_id:
            credential = await session.get(Credential, UUID(str(credential_id)))
    if credential is None:
        result = await session.execute(
            select(Credential)
            .where(
                Credential.namespace_id == ns.id,
                Credential.provider == "falkordb",
            )
            .order_by(Credential.created_at.desc())
        )
        credential = result.scalars().first()
    if credential is None:
        return None

    if credential.namespace_id != ns.id:
        return None

    base_url = (credential.base_url or settings.falkordb_url or "").strip()
    parsed = urlparse(base_url.replace("falkordb://", "redis://", 1))

    connection_kwargs: dict[str, str | int | bool] = {}
    if parsed.hostname:
        connection_kwargs["host"] = parsed.hostname
    if parsed.port:
        connection_kwargs["port"] = int(parsed.port)
    if parsed.scheme == "rediss":
        connection_kwargs["ssl"] = True
    username = ""
    if isinstance(falkordb_meta, dict):
        username = str(falkordb_meta.get("username") or "").strip()
    if not username:
        username = str(credential.label or "").strip()
    if username:
        connection_kwargs["username"] = username
    secret = _crypto.decrypt(credential.encrypted_secret)
    if secret:
        connection_kwargs["password"] = secret
    return connection_kwargs


async def list_namespaces(session: AsyncSession) -> list[Namespace]:
    result = await session.execute(
        select(Namespace).order_by(Namespace.created_at.desc())
    )
    return list(result.scalars().all())


async def get_namespace_by_id(
    session: AsyncSession, namespace_id: str
) -> Namespace | None:
    from uuid import UUID

    return await session.get(Namespace, UUID(namespace_id))

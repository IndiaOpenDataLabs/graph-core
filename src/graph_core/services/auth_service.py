"""Authentication service — admin key, namespace key, and secret hashing."""

import secrets
from dataclasses import dataclass

import bcrypt
from hmac import compare_digest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from graph_core.config import settings
from graph_core.models.namespace import Namespace


def _hash_secret(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _verify_secret(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def generate_api_key() -> tuple[str, str]:
    """Generate a namespace API key.

    Returns (full_key, bcrypt_hash). The full key has format ns_key_<32 hex>.
    """
    raw = secrets.token_hex(16)
    full_key = f"ns_key_{raw}"
    return full_key, _hash_secret(full_key)


def is_admin_key(key: str) -> bool:
    """Check if a key matches the primary or secondary admin key.

    Admin keys are plain-text env vars — constant-time comparison, no hashing.
    """
    if key and settings.platform_admin_key and compare_digest(key, settings.platform_admin_key):
        return True
    if key and settings.platform_admin_key_secondary and compare_digest(key, settings.platform_admin_key_secondary):
        return True
    return False


async def verify_namespace_api_key(
    session: AsyncSession,
    key: str,
) -> Namespace | None:
    """Look up a namespace by API key.

    Returns the Namespace if the key matches, None otherwise.
    """
    result = await session.execute(
        select(Namespace).where(Namespace.api_key_hash.isnot(None))
    )
    for ns in result.scalars().all():
        if ns.api_key_hash and _verify_secret(key, ns.api_key_hash):
            return ns
    return None


@dataclass
class NamespaceCreateResult:
    namespace: Namespace
    api_key: str


async def create_namespace_with_key(
    session: AsyncSession,
    *,
    name: str,
    owner_app_id: str | None = None,
    owner_user_sub: str | None = None,
) -> NamespaceCreateResult:
    """Create a new namespace with a generated API key."""
    full_key, key_hash = generate_api_key()
    ns = Namespace(
        name=name,
        api_key_hash=key_hash,
        api_key_prefix=full_key[:11],
        owner_app_id=owner_app_id,
        owner_user_sub=owner_user_sub,
    )
    session.add(ns)
    await session.flush()
    await session.refresh(ns)
    return NamespaceCreateResult(namespace=ns, api_key=full_key)


async def rotate_namespace_key(
    session: AsyncSession,
    namespace_id: str,
) -> NamespaceCreateResult | None:
    """Rotate a namespace's API key. Returns new key or None if not found."""
    from uuid import UUID

    ns = await session.get(Namespace, UUID(namespace_id))
    if ns is None:
        return None

    full_key, key_hash = generate_api_key()
    ns.api_key_hash = key_hash
    ns.api_key_prefix = full_key[:11]
    await session.flush()
    await session.refresh(ns)
    return NamespaceCreateResult(namespace=ns, api_key=full_key)


async def list_namespaces(session: AsyncSession) -> list[Namespace]:
    result = await session.execute(select(Namespace).order_by(Namespace.created_at.desc()))
    return list(result.scalars().all())


async def get_namespace_by_id(session: AsyncSession, namespace_id: str) -> Namespace | None:
    from uuid import UUID

    return await session.get(Namespace, UUID(namespace_id))


def hash_client_secret(plain: str) -> str:
    """Hash a registered app's client secret. Used in multi-tenant mode."""
    return _hash_secret(plain)


def verify_client_secret(plain: str, hashed: str) -> bool:
    """Verify a registered app's client secret. Used in multi-tenant mode."""
    return _verify_secret(plain, hashed)

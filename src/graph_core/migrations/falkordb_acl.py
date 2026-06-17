"""Helpers for replaying namespace FalkorDB ACL provisioning."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from graph_core.models.credential import Credential
from graph_core.models.namespace import Namespace
from graph_core.services.auth_service import _provision_namespace_falkordb_acl
from graph_core.services.crypto import CredentialCrypto

_crypto = CredentialCrypto()


def load_namespace_acl_payloads(
    session: Session,
) -> list[tuple[str, str, str, str | None]]:
    """Load namespace ACL replay payloads from the sync Alembic session."""
    payloads: list[tuple[str, str, str, str | None]] = []
    namespace_rows = session.execute(
        select(Namespace.id, Namespace.metadata_json, Namespace.created_at).order_by(
            Namespace.created_at.asc()
        )
    )
    for namespace_id, metadata_json in namespace_rows.all():
        metadata = metadata_json or {}
        falkordb_meta = metadata.get("falkordb")
        credential: Credential | None = None
        if isinstance(falkordb_meta, dict):
            credential_id = falkordb_meta.get("credential_id")
            if credential_id:
                credential = session.get(Credential, UUID(str(credential_id)))
        if credential is None:
            credential = (
                session.execute(
                    select(Credential)
                    .where(
                        Credential.namespace_id == namespace_id,
                        Credential.provider == "falkordb",
                    )
                    .order_by(Credential.created_at.desc())
                )
                .scalars()
                .first()
            )
        if credential is None:
            continue

        if isinstance(falkordb_meta, dict):
            username = str(falkordb_meta.get("username") or "").strip()
        else:
            username = ""
        if not username:
            username = str(credential.label or f"ns_{namespace_id.hex}").strip()
        if not username:
            continue

        payloads.append(
            (
                str(namespace_id),
                username,
                _crypto.decrypt(credential.encrypted_secret),
                credential.base_url,
            )
        )
    return payloads


async def replay_namespace_acl_payloads(
    payloads: list[tuple[str, str, str, str | None]],
) -> None:
    """Replay namespace ACL provisioning in FalkorDB."""
    for namespace_id, username, secret, base_url in payloads:
        await _provision_namespace_falkordb_acl(
            namespace_id=namespace_id,
            username=username,
            secret=secret,
            base_url=base_url,
        )

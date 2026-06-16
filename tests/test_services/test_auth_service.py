"""Auth service tests."""

import pytest

from graph_core.models.credential import Credential
from graph_core.models.namespace import Namespace
from graph_core.services import auth_service


@pytest.mark.asyncio
async def test_provision_namespace_falkordb_credential_persists_metadata(db_session):
    ns = Namespace(name="ns-a")
    db_session.add(ns)
    await db_session.commit()
    await db_session.refresh(ns)

    (
        namespace,
        credential,
        secret,
    ) = await auth_service.provision_namespace_falkordb_credential(
        db_session,
        str(ns.id),
        username="tenant_ns_a",
        secret="falkor-secret",
        base_url="falkordb://localhost:6379",
    )

    assert namespace.id == ns.id
    assert credential.provider == "falkordb"
    assert credential.label == "tenant_ns_a"
    assert secret == "falkor-secret"

    stored_credential = await db_session.get(Credential, credential.id)
    assert stored_credential is not None
    assert stored_credential.base_url == "falkordb://localhost:6379"

    stored_ns = await db_session.get(Namespace, ns.id)
    assert stored_ns is not None
    assert stored_ns.metadata_json["falkordb"]["credential_id"] == str(credential.id)
    assert stored_ns.metadata_json["falkordb"]["username"] == "tenant_ns_a"
    assert stored_ns.metadata_json["falkordb"]["graph_pattern"] == f"tenant:{ns.id}:*"


@pytest.mark.asyncio
async def test_ensure_namespace_falkordb_credential_backfills_missing_state(
    db_session,
):
    ns = Namespace(name="ns-backfill")
    db_session.add(ns)
    await db_session.commit()
    await db_session.refresh(ns)

    (
        namespace,
        credential,
        secret,
    ) = await auth_service.ensure_namespace_falkordb_credential(
        db_session,
        str(ns.id),
    )

    assert namespace.id == ns.id
    assert credential.provider == "falkordb"
    assert secret is not None

    stored_ns = await db_session.get(Namespace, ns.id)
    assert stored_ns is not None
    assert stored_ns.metadata_json["falkordb"]["credential_id"] == str(credential.id)
    assert stored_ns.metadata_json["falkordb"]["graph_pattern"] == f"tenant:{ns.id}:*"

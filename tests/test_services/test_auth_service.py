"""Auth service tests."""

import pytest

from graph_core.models.credential import Credential
from graph_core.models.namespace import Namespace
from graph_core.services import auth_service


class _FakeRedis:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    async def execute_command(self, *args):
        self.calls.append(args)
        return "OK"

    async def aclose(self, close_connection_pool=None) -> None:  # noqa: ARG002
        return None


@pytest.mark.asyncio
async def test_provision_namespace_falkordb_credential_persists_metadata(
    db_session, monkeypatch
):
    fake_redis = _FakeRedis()

    def _fake_from_url(url: str, decode_responses: bool = False, **kwargs):  # noqa: ARG001
        assert url == "redis://localhost:6379"
        return fake_redis

    monkeypatch.setattr(
        "redis.asyncio.client.Redis.from_url",
        _fake_from_url,
    )

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
    assert stored_ns.metadata_json["falkordb"]["db"] == stored_ns.falkordb_db
    assert stored_ns.metadata_json["falkordb"]["graph_pattern"] == f"tenant:{ns.id}:*"


@pytest.mark.asyncio
async def test_provision_namespace_falkordb_credential_sets_acl_user(
    db_session, monkeypatch
):
    fake_redis = _FakeRedis()

    def _fake_from_url(url: str, decode_responses: bool = False, **kwargs):  # noqa: ARG001
        assert url == "redis://localhost:6379"
        return fake_redis

    monkeypatch.setattr(
        "redis.asyncio.client.Redis.from_url",
        _fake_from_url,
    )

    ns = Namespace(name="ns-acl")
    db_session.add(ns)
    await db_session.commit()
    await db_session.refresh(ns)

    await auth_service.provision_namespace_falkordb_credential(
        db_session,
        str(ns.id),
        username="tenant_ns_acl",
        secret="falkor-secret",
        base_url="falkordb://localhost:6379",
    )

    assert fake_redis.calls == [
        (
            "ACL",
            "SETUSER",
            "tenant_ns_acl",
            "reset",
            "on",
            ">falkor-secret",
            "+AUTH",
            "+INFO",
            "+EXISTS",
            "+MODULE|LIST",
            "+PING",
            "+SELECT",
            "+GRAPH.LIST",
            "+GRAPH.QUERY",
            "+GRAPH.EXPLAIN",
            "+GRAPH.MEMORY",
            "+GRAPH.UDF",
            "+GRAPH.DELETE",
            "+GRAPH.RO_QUERY",
            f"~tenant:{ns.id}:*",
            f"%R~tenant:{ns.id}:*",
            f"%W~tenant:{ns.id}:*",
        ),
    ]


@pytest.mark.asyncio
async def test_ensure_namespace_falkordb_credential_backfills_missing_state(
    db_session,
    monkeypatch,
):
    fake_redis = _FakeRedis()

    def _fake_from_url(url: str, decode_responses: bool = False, **kwargs):  # noqa: ARG001
        assert url == "redis://localhost:6379"
        return fake_redis

    monkeypatch.setattr(
        "redis.asyncio.client.Redis.from_url",
        _fake_from_url,
    )

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
    assert stored_ns.metadata_json["falkordb"]["db"] == stored_ns.falkordb_db
    assert stored_ns.metadata_json["falkordb"]["graph_pattern"] == f"tenant:{ns.id}:*"

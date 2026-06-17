"""Tests for the tenant-scoped FalkorDB proxy."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ResponseError

from graph_core.models.credential import Credential
from graph_core.models.namespace import Namespace
from graph_core.proxy.falkordb_proxy import (
    FalkorDBTenantProxy,
    NamespaceAuth,
    SimpleString,
    _ProxySession,
)
from graph_core.services.crypto import CredentialCrypto


class _FakeRedis:
    def __init__(self) -> None:
        self.ping = AsyncMock(return_value=True)
        self.aclose = AsyncMock(return_value=None)
        self.execute_command = AsyncMock()


@pytest.mark.asyncio
async def test_resolve_namespace_auth_uses_namespace_credential(db_session):
    crypto = CredentialCrypto()
    namespace = Namespace(name="tenant-x", falkordb_db=1)
    db_session.add(namespace)
    await db_session.flush()

    credential = Credential(
        namespace_id=namespace.id,
        provider="falkordb",
        label="tenant_user",
        encrypted_secret=crypto.encrypt("tenant-secret"),
        base_url="falkordb://localhost:6379",
    )
    db_session.add(credential)
    await db_session.commit()

    proxy = FalkorDBTenantProxy(upstream_default_url="redis://localhost:6379")
    auth = await proxy._resolve_namespace_auth("tenant_user", "tenant-secret")

    assert auth is not None
    assert auth.namespace_id == str(namespace.id)
    assert auth.db == 1
    assert auth.graph_prefix == f"tenant:{namespace.id}:"


@pytest.mark.asyncio
async def test_proxy_filters_graph_list_and_blocks_other_graphs(monkeypatch):
    fake_upstream = _FakeRedis()
    fake_upstream.execute_command.side_effect = [
        [
            "tenant:abc:collection:one",
            "tenant:def:collection:two",
        ],
        "OK",
    ]

    monkeypatch.setattr(
        "graph_core.proxy.falkordb_proxy.Redis.from_url",
        lambda *args, **kwargs: fake_upstream,
    )

    proxy = FalkorDBTenantProxy(upstream_default_url="redis://localhost:6379")
    session = _ProxySession(proxy=proxy, reader=None, writer=None)  # type: ignore[arg-type]
    session._proxy._resolve_namespace_auth = AsyncMock(
        return_value=NamespaceAuth(
            namespace_id="abc",
            namespace_name="tenant-abc",
            username="tenant_user",
            password="tenant-secret",
            graph_prefix="tenant:abc:",
            upstream_url="redis://localhost:6379",
            db=1,
        )
    )

    auth_reply = await session.handle_command(["AUTH", "tenant_user", "tenant-secret"])
    assert isinstance(auth_reply, SimpleString)
    assert auth_reply.value == "OK"

    graph_list = await session.handle_command(["GRAPH.LIST"])
    assert graph_list == ["tenant:abc:collection:one"]

    allowed = await session.handle_command(
        [
            "GRAPH.QUERY",
            "tenant:abc:collection:one",
            "MATCH (n) RETURN n",
        ]
    )
    assert allowed == "OK"

    denied = await session.handle_command(
        [
            "GRAPH.QUERY",
            "tenant:def:collection:two",
            "MATCH (n) RETURN n",
        ]
    )
    assert "No permissions to access a key" in str(denied)

    select_reply = await session.handle_command(["SELECT", "0"])
    assert isinstance(select_reply, SimpleString)
    assert select_reply.value == "OK"
    assert fake_upstream.execute_command.call_count == 2


@pytest.mark.asyncio
async def test_proxy_returns_clean_error_for_upstream_failures(monkeypatch):
    fake_upstream = _FakeRedis()
    fake_upstream.execute_command.side_effect = ResponseError(
        "No permissions to access a key"
    )

    monkeypatch.setattr(
        "graph_core.proxy.falkordb_proxy.Redis.from_url",
        lambda *args, **kwargs: fake_upstream,
    )

    proxy = FalkorDBTenantProxy(upstream_default_url="redis://localhost:6379")
    session = _ProxySession(proxy=proxy, reader=None, writer=None)  # type: ignore[arg-type]
    session._proxy._resolve_namespace_auth = AsyncMock(
        return_value=NamespaceAuth(
            namespace_id="abc",
            namespace_name="tenant-abc",
            username="tenant_user",
            password="tenant-secret",
            graph_prefix="tenant:abc:",
            upstream_url="redis://localhost:6379",
            db=1,
        )
    )

    auth_reply = await session.handle_command(["AUTH", "tenant_user", "tenant-secret"])
    assert isinstance(auth_reply, SimpleString)

    error_reply = await session.handle_command(
        [
            "GRAPH.QUERY",
            "tenant:abc:collection:one",
            "MATCH (n) RETURN n",
        ]
    )
    assert error_reply == []


@pytest.mark.asyncio
async def test_proxy_uses_safe_fallbacks_for_permission_errors(monkeypatch):
    fake_upstream = _FakeRedis()
    fake_upstream.execute_command.side_effect = [
        ResponseError("No permissions to access a key"),
        ResponseError("No permissions to access a key"),
        ResponseError("No permissions to access a key"),
    ]

    monkeypatch.setattr(
        "graph_core.proxy.falkordb_proxy.Redis.from_url",
        lambda *args, **kwargs: fake_upstream,
    )

    proxy = FalkorDBTenantProxy(upstream_default_url="redis://localhost:6379")
    session = _ProxySession(proxy=proxy, reader=None, writer=None)  # type: ignore[arg-type]
    session._proxy._resolve_namespace_auth = AsyncMock(
        return_value=NamespaceAuth(
            namespace_id="abc",
            namespace_name="tenant-abc",
            username="tenant_user",
            password="tenant-secret",
            graph_prefix="tenant:abc:",
            upstream_url="redis://localhost:6379",
            db=1,
        )
    )

    auth_reply = await session.handle_command(["AUTH", "tenant_user", "tenant-secret"])
    assert isinstance(auth_reply, SimpleString)

    graph_list = await session.handle_command(["GRAPH.LIST"])
    assert graph_list == []

    graph_query = await session.handle_command(
        [
            "GRAPH.QUERY",
            "tenant:abc:collection:one",
            "MATCH (n) RETURN n",
        ]
    )
    assert graph_query == []


@pytest.mark.asyncio
async def test_proxy_formats_info_as_raw_text(monkeypatch):
    fake_upstream = _FakeRedis()
    fake_upstream.execute_command.side_effect = [
        {"redis_mode": "standalone", "role": "master"},
    ]

    monkeypatch.setattr(
        "graph_core.proxy.falkordb_proxy.Redis.from_url",
        lambda *args, **kwargs: fake_upstream,
    )

    proxy = FalkorDBTenantProxy(upstream_default_url="redis://localhost:6379")
    session = _ProxySession(proxy=proxy, reader=None, writer=None)  # type: ignore[arg-type]
    session._proxy._resolve_namespace_auth = AsyncMock(
        return_value=NamespaceAuth(
            namespace_id="abc",
            namespace_name="tenant-abc",
            username="tenant_user",
            password="tenant-secret",
            graph_prefix="tenant:abc:",
            upstream_url="redis://localhost:6379",
            db=1,
        )
    )

    auth_reply = await session.handle_command(["AUTH", "tenant_user", "tenant-secret"])
    assert isinstance(auth_reply, SimpleString)

    info_reply = await session.handle_command(["INFO"])
    assert isinstance(info_reply, str)
    assert "redis_mode:standalone" in info_reply
    assert info_reply.endswith("\n")


@pytest.mark.asyncio
async def test_proxy_allows_udf_list_without_graph_prefix(monkeypatch):
    fake_upstream = _FakeRedis()
    fake_upstream.execute_command.side_effect = [["lib_a", "lib_b"]]

    monkeypatch.setattr(
        "graph_core.proxy.falkordb_proxy.Redis.from_url",
        lambda *args, **kwargs: fake_upstream,
    )

    proxy = FalkorDBTenantProxy(upstream_default_url="redis://localhost:6379")
    session = _ProxySession(proxy=proxy, reader=None, writer=None)  # type: ignore[arg-type]
    session._proxy._resolve_namespace_auth = AsyncMock(
        return_value=NamespaceAuth(
            namespace_id="abc",
            namespace_name="tenant-abc",
            username="tenant_user",
            password="tenant-secret",
            graph_prefix="tenant:abc:",
            upstream_url="redis://localhost:6379",
            db=1,
        )
    )

    auth_reply = await session.handle_command(["AUTH", "tenant_user", "tenant-secret"])
    assert isinstance(auth_reply, SimpleString)

    udf_reply = await session.handle_command(["GRAPH.UDF", "LIST"])
    assert udf_reply == ["lib_a", "lib_b"]

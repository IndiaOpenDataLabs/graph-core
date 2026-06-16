from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ResponseError

from graph_core.storage import graph_storage as graph_storage_module
from graph_core.storage.graph_storage import FalkorDBGraphStorage


class _FakeGraph:
    def __init__(self) -> None:
        self.query = AsyncMock()


class _FakeClient:
    def __init__(self) -> None:
        self.graph = _FakeGraph()
        self.execute_command = AsyncMock()

    def select_graph(self, _graph_name: str):
        return self.graph


class _FakePool:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeFalkorDB:
    def __init__(self, connection_pool=None):
        self.connection_pool = connection_pool
        self.graph = _FakeGraph()

    def select_graph(self, _graph_name: str):
        return self.graph


@pytest.mark.asyncio
async def test_drop_deletes_graph_namespace_via_graph_delete():
    client = _FakeClient()
    storage = FalkorDBGraphStorage("collection_deadbeef", _client=client)

    await storage.drop()

    client.execute_command.assert_awaited_once_with(
        "GRAPH.DELETE",
        "collection_deadbeef",
    )
    client.graph.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_drop_ignores_missing_graph_delete_error():
    client = _FakeClient()
    client.execute_command.side_effect = ResponseError(
        "Invalid graph operation on empty key"
    )
    storage = FalkorDBGraphStorage("collection_deadbeef", _client=client)

    await storage.drop()

    client.execute_command.assert_awaited_once_with(
        "GRAPH.DELETE",
        "collection_deadbeef",
    )


@pytest.mark.asyncio
async def test_exists_checks_graph_key():
    client = _FakeClient()
    client.execute_command.return_value = 1
    storage = FalkorDBGraphStorage("collection_deadbeef", _client=client)

    assert await storage.exists() is True

    client.execute_command.assert_awaited_once_with("EXISTS", "collection_deadbeef")


@pytest.mark.asyncio
async def test_rename_renames_graph_key():
    client = _FakeClient()
    storage = FalkorDBGraphStorage("collection_deadbeef", _client=client)

    renamed = await storage.rename("collection_new_name")

    assert renamed is True
    client.execute_command.assert_awaited_once_with(
        "RENAME",
        "collection_deadbeef",
        "collection_new_name",
    )


@pytest.mark.asyncio
async def test_rename_ignores_missing_graph_key():
    client = _FakeClient()
    client.execute_command.side_effect = ResponseError("no such key")
    storage = FalkorDBGraphStorage("collection_deadbeef", _client=client)

    renamed = await storage.rename("collection_new_name")

    assert renamed is False


@pytest.mark.asyncio
async def test_node_count_reads_entity_count():
    client = _FakeClient()
    client.graph.query.return_value.result_set = [[7]]
    storage = FalkorDBGraphStorage("collection_deadbeef", _client=client)

    assert await storage.node_count() == 7


@pytest.mark.asyncio
async def test_namespace_connection_resolution_uses_namespace_credentials(
    monkeypatch,
):
    async def _resolver(session, namespace_id):
        assert namespace_id == "ns-123"
        return {
            "host": "tenant-db.example.com",
            "port": 6380,
            "db": 7,
            "username": "tenant_ns",
            "password": "secret",
        }

    monkeypatch.setattr(graph_storage_module, "FalkorDB", _FakeFalkorDB)
    monkeypatch.setattr(graph_storage_module, "BlockingConnectionPool", _FakePool)
    monkeypatch.setattr(
        graph_storage_module.auth_service,
        "resolve_namespace_falkordb_connection",
        _resolver,
    )

    storage = FalkorDBGraphStorage(
        "tenant:ns-123:collection:collection_demo_abcd1234",
        namespace_id=None,
        _connection_resolver=lambda: _resolver(None, "ns-123"),
    )

    graph = await storage._get_graph()
    assert graph is not None
    assert storage._client is not None
    assert storage._client.connection_pool.kwargs["host"] == "tenant-db.example.com"
    assert storage._client.connection_pool.kwargs["port"] == 6380
    assert storage._client.connection_pool.kwargs["db"] == 7
    assert storage._client.connection_pool.kwargs["username"] == "tenant_ns"
    assert storage._client.connection_pool.kwargs["password"] == "secret"

from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ResponseError

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

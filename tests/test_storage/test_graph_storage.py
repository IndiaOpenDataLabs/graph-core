from unittest.mock import AsyncMock

import pytest

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

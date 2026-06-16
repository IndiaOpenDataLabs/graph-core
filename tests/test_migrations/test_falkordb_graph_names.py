"""FalkorDB graph-name migration tests."""

from __future__ import annotations

import uuid

import pytest

from graph_core.migrations import falkordb_graph_names


class _FakeGraphStorage:
    graphs: dict[str, int] = {}

    def __init__(self, graph_name: str, *, namespace_id=None, **kwargs):  # noqa: ANN001,ARG002
        self._graph_name = graph_name
        self._namespace_id = namespace_id

    async def exists(self) -> bool:
        return self._graph_name in self.graphs

    async def node_count(self) -> int:
        return self.graphs.get(self._graph_name, 0)

    async def rename(self, new_graph_name: str) -> bool:
        if self._graph_name not in self.graphs:
            return False
        self.graphs[new_graph_name] = self.graphs.pop(self._graph_name)
        self._graph_name = new_graph_name
        return True

    async def drop(self) -> None:
        self.graphs.pop(self._graph_name, None)


@pytest.mark.asyncio
async def test_replay_collection_graph_names_renames_namespace_graph(monkeypatch):
    namespace_id = uuid.UUID("87654321-4321-8765-4321-876543218765")
    old_graph_name = "collection_rlm_4e57bbb0"
    new_graph_name = (
        "tenant:87654321-4321-8765-4321-876543218765:collection:collection_rlm_4e57bbb0"
    )
    _FakeGraphStorage.graphs = {old_graph_name: 3}

    monkeypatch.setattr(
        falkordb_graph_names,
        "FalkorDBGraphStorage",
        _FakeGraphStorage,
    )

    await falkordb_graph_names.replay_collection_graph_names(
        [(new_graph_name, namespace_id, [old_graph_name])]
    )

    assert old_graph_name not in _FakeGraphStorage.graphs
    assert _FakeGraphStorage.graphs[new_graph_name] == 3

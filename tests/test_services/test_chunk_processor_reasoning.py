import uuid
from types import SimpleNamespace

import networkx as nx
import pytest
from sqlalchemy.dialects import postgresql

from graph_core.models.collection import Collection
from graph_core.services.graph import analytics
from graph_core.services.graph.ingestion.chunk_processor import (
    _upsert_reasoning_entity,
)
from graph_core.storage.graph_rag_vectors import GraphRAGVectorStore
from graph_core.storage.vector_tables import (
    create_entity_embeddings_sql,
    create_relationship_embeddings_sql,
)


def _id(value: int) -> uuid.UUID:
    return uuid.UUID(int=value)


class _ScalarResult:
    def __init__(self, value: uuid.UUID) -> None:
        self.value = value

    def scalar_one(self) -> uuid.UUID:
        return self.value

    def scalar_one_or_none(self) -> uuid.UUID:
        return self.value


class _EmptyScalarResult:
    def scalar_one_or_none(self) -> None:
        return None


class _RecordingSession:
    def __init__(self, canonical_id: uuid.UUID) -> None:
        self.canonical_id = canonical_id
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        if len(self.statements) == 1:
            return _EmptyScalarResult()
        return _ScalarResult(self.canonical_id)


class _VectorSession:
    def __init__(self) -> None:
        self.executions = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def execute(self, statement, parameters=None):
        self.executions.append((str(statement), parameters))
        return _ScalarResult(_id(99))

    async def commit(self) -> None:
        self.committed = True

    async def close(self) -> None:
        return None


class _ProjectionSession(_VectorSession):
    async def execute(self, statement, parameters=None):
        self.executions.append((str(statement), parameters))
        if len(self.executions) == 2:
            return SimpleNamespace(
                scalar_one_or_none=lambda: SimpleNamespace(status="completed")
            )
        return SimpleNamespace()


@pytest.mark.asyncio
async def test_reasoning_entity_conflict_returns_and_reuses_canonical_id() -> None:
    requested_id = _id(10)
    canonical_id = _id(11)
    collection = Collection(
        id=_id(1),
        namespace_id=_id(2),
        name="test",
        strategy="custom_graph_rag",
    )
    session = _RecordingSession(canonical_id)

    resolved_id = await _upsert_reasoning_entity(
        session,
        collection=collection,
        entity_id=requested_id,
        canonical_name="condition:when calm",
        primary_type="CONDITION",
        description="when calm",
        chunk_hash="a" * 64,
        document_id=_id(3),
        document_path="book.md",
    )

    entity_sql = str(
        session.statements[0].compile(dialect=postgresql.dialect())
    )
    description_parameters = session.statements[2].compile(
        dialect=postgresql.dialect()
    ).params
    assert resolved_id == canonical_id
    assert "ON CONFLICT DO NOTHING" in entity_sql
    assert "RETURNING graph_entities.id" in entity_sql
    assert "graph_entities.canonical_name" in str(session.statements[1])
    assert canonical_id in description_parameters.values()
    assert requested_id not in description_parameters.values()


def test_dynamic_vector_tables_enforce_logical_identity() -> None:
    entity_sql = create_entity_embeddings_sql(_id(1), 8)
    relationship_sql = create_relationship_embeddings_sql(_id(1), 8)

    assert "UNIQUE (description_id)" in entity_sql
    assert "UNIQUE (relationship_id)" in relationship_sql


@pytest.mark.asyncio
async def test_vector_writes_use_atomic_conflict_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def dimensions(collection_id: uuid.UUID) -> int:
        return 3

    entity_session = _VectorSession()
    relationship_session = _VectorSession()
    monkeypatch.setattr(
        "graph_core.storage.graph_rag_vectors.get_collection_dimensions",
        dimensions,
    )
    monkeypatch.setattr(
        "graph_core.storage.graph_rag_vectors.AsyncSessionLocal",
        lambda: relationship_session,
    )
    store = GraphRAGVectorStore()

    await store.upsert_entity_embedding(
        entity_id=_id(1),
        collection_id=_id(2),
        name="Alpha",
        description="Alpha description",
        description_id=_id(3),
        embedding=[0.1, 0.2, 0.3],
        session=entity_session,
    )
    await store.upsert_relationship_embeddings(
        _id(2),
        [
            {
                "relationship_id": _id(4),
                "source_name": "Alpha",
                "target_name": "Beta",
                "description": "Alpha affects Beta",
                "embedding": [0.3, 0.2, 0.1],
            }
        ],
    )

    entity_statement = entity_session.executions[0][0]
    relationship_statements = [item[0] for item in relationship_session.executions]
    assert "ON CONFLICT (description_id) DO UPDATE" in entity_statement
    assert len(relationship_statements) == 1
    assert "DELETE FROM" not in relationship_statements[0]
    assert "ON CONFLICT (relationship_id) DO UPDATE" in relationship_statements[0]


@pytest.mark.asyncio
async def test_projection_persistence_locks_before_checking_existing_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _ProjectionSession()
    monkeypatch.setattr(analytics, "AsyncSessionLocal", lambda: session)

    result = await analytics._persist_projection(
        SimpleNamespace(id=_id(1)),
        {"name": "semantic", "directed": False},
        nx.Graph(),
        {},
    )

    assert result == "skipped"
    assert "pg_advisory_xact_lock" in session.executions[0][0]
    assert "graph_projection_snapshots" in session.executions[1][0]

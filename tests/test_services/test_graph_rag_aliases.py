import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from graph_core.models.graph_rag import (
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
    RelationshipTypeAlias,
)
from graph_core.models.rel_types import DEFAULT_REL_TYPE
from graph_core.services.graph.query import graph_rag
from graph_core.services.graph.query.graph_rag import GraphQueryState
from graph_core.services.graph_rag.entity_resolver import IncrementalEntityResolver
from graph_core.services.graph_rag.extractor import LLMGraphExtractor


class _FakeEmbeddingProvider:
    dimensions = 256

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _SessionFactory:
    def __init__(self, session):
        self._session = session

    def __call__(self):
        return self

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_resolve_entity_reuses_case_insensitive_canonical_match(
    db_session,
    test_graph_rag_collection,
):
    entity = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="OAuth",
        primary_type="Protocol",
        description_count=0,
    )
    db_session.add(entity)
    await db_session.commit()

    resolver = IncrementalEntityResolver(
        _FakeEmbeddingProvider(),
        test_graph_rag_collection.id,
    )
    resolver._add_description_and_update_centroid = AsyncMock()
    resolver._add_or_increment_type = AsyncMock()

    result = await resolver.resolve_entity(
        db_session,
        name="oauth",
        entity_type="Protocol",
        description="Authorization framework",
        source_chunk_hash="chunk-1",
    )

    assert result.is_new is False
    assert result.entity_id == entity.id
    assert result.canonical_name == "OAuth"
    resolver._add_or_increment_type.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_or_increment_type_awaits_async_session_execute(
    test_graph_rag_collection,
):
    resolver = IncrementalEntityResolver(
        _FakeEmbeddingProvider(),
        test_graph_rag_collection.id,
    )
    session = AsyncMock()

    await resolver._add_or_increment_type(session, uuid.uuid4(), "Protocol")

    session.execute.assert_awaited_once()


def test_extractor_normalize_entity_name_preserves_casing():
    assert LLMGraphExtractor._normalize_entity_name("  OAuth SDK  ") == "OAuth SDK"
    assert LLMGraphExtractor._normalize_entity_name("iOS") == "iOS"
    assert LLMGraphExtractor._normalize_entity_name("JSON-LD") == "JSON-LD"


@pytest.mark.asyncio
async def test_active_dimensions_consolidates_aliases(
    db_session,
    test_graph_rag_collection,
    monkeypatch,
):
    rel_type = GraphRelationshipType(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_type="CALLS",
    )
    db_session.add_all(
        [
            rel_type,
            RelationshipTypeAlias(
                id=uuid.uuid4(),
                collection_id=test_graph_rag_collection.id,
                relationship_type_id=rel_type.id,
                canonical_type="CALLS",
                alias_type="INVOKES",
            ),
            GraphRelationship(
                id=uuid.uuid4(),
                collection_id=test_graph_rag_collection.id,
                source_entity_id=uuid.uuid4(),
                target_entity_id=uuid.uuid4(),
                relationship_type_id=rel_type.id,
                rel_type="CALLS",
                weight=1,
                keywords=[],
            ),
            GraphRelationship(
                id=uuid.uuid4(),
                collection_id=test_graph_rag_collection.id,
                source_entity_id=uuid.uuid4(),
                target_entity_id=uuid.uuid4(),
                relationship_type_id=rel_type.id,
                rel_type="CALLS",
                weight=1,
                keywords=[],
            ),
        ]
    )
    await db_session.commit()
    monkeypatch.setattr(graph_rag, "AsyncSessionLocal", _SessionFactory(db_session))

    dimensions = await graph_rag._active_dimensions(test_graph_rag_collection)

    assert dimensions[0] == DEFAULT_REL_TYPE
    assert dimensions.count("CALLS") == 1
    assert "INVOKES" not in dimensions


@pytest.mark.asyncio
async def test_derive_route_profile_uses_canonical_rel_type_alias(
    db_session,
    test_graph_rag_collection,
    monkeypatch,
):
    rel_type = GraphRelationshipType(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_type="CALLS",
    )
    relationship = GraphRelationship(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        source_entity_id=uuid.uuid4(),
        target_entity_id=uuid.uuid4(),
        relationship_type_id=rel_type.id,
        rel_type="CALLS",
        weight=1,
        keywords=[],
    )
    db_session.add_all(
        [
            rel_type,
            RelationshipTypeAlias(
                id=uuid.uuid4(),
                collection_id=test_graph_rag_collection.id,
                relationship_type_id=rel_type.id,
                canonical_type="CALLS",
                alias_type="INVOKES",
            ),
            relationship,
        ]
    )
    await db_session.commit()
    monkeypatch.setattr(graph_rag, "AsyncSessionLocal", _SessionFactory(db_session))

    profile = await graph_rag._derive_route_profile(
        test_graph_rag_collection,
        GraphQueryState(
            discovered_entity_ids=set(),
            entity_relevance={},
            traversed_rel_ids=[str(relationship.id)],
            rel_score_cache={},
            rel_combined_score_cache={str(relationship.id): 2.0},
        ),
    )

    assert profile.primary_route == "hub"
    assert "CALLS" in profile.rel_type_scores
    assert "INVOKES" not in profile.rel_type_scores


@pytest.mark.asyncio
async def test_resolve_rel_type_creates_canonical_relationship_type(
    db_session,
    test_graph_rag_collection,
):
    resolver = IncrementalEntityResolver(
        _FakeEmbeddingProvider(),
        test_graph_rag_collection.id,
    )
    resolver._vstore.ensure_prefix_embeddings_table = AsyncMock()
    resolver._vstore.load_all_prefix_embeddings = AsyncMock(return_value={})
    resolver._vstore.upsert_prefix_embedding = AsyncMock()

    result = await resolver._resolve_rel_type(db_session, "invokes")
    await db_session.commit()

    assert result.canonical_type == "INVOKES"
    rows = (
        await db_session.execute(
            select(GraphRelationshipType).where(
                GraphRelationshipType.id == result.relationship_type_id
            )
        )
    ).scalars().all()
    assert len(rows) == 1
    alias_rows = (
        await db_session.execute(
            select(RelationshipTypeAlias).where(
                RelationshipTypeAlias.relationship_type_id == result.relationship_type_id
            )
        )
    ).scalars().all()
    assert {row.alias_type for row in alias_rows} == {"INVOKES"}

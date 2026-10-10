"""Relationship document filtering is an existence check, not a uniqueness check."""

import importlib
import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from graph_core.models.graph_rag import (
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
    RelationshipDescription,
)

graph_rag = importlib.import_module("graph_core.services.graph.query.graph_rag")


@pytest.mark.asyncio
@pytest.mark.parametrize("traversed", [False, True])
@pytest.mark.parametrize(
    "scope",
    [
        "none_matching",
        "one_matching",
        "same_document",
        "multiple_documents",
        "unscoped",
    ],
)
async def test_relationship_document_scope_accepts_any_matching_description(
    db_session,
    test_graph_rag_collection,
    monkeypatch,
    traversed,
    scope,
):
    collection = test_graph_rag_collection
    source, target = [
        GraphEntity(
            id=uuid.uuid4(),
            collection_id=collection.id,
            canonical_name=name,
            primary_type="CONCEPT",
        )
        for name in ["Source", "Target"]
    ]
    rel_type = GraphRelationshipType(
        id=uuid.uuid4(),
        collection_id=collection.id,
        canonical_type="RELATES_TO",
    )
    db_session.add_all([source, target, rel_type])
    await db_session.flush()
    rel = GraphRelationship(
        id=uuid.uuid4(),
        collection_id=collection.id,
        source_entity_id=source.id,
        target_entity_id=target.id,
        relationship_type_id=rel_type.id,
        rel_type=rel_type.canonical_type,
    )
    db_session.add(rel)
    await db_session.flush()
    selected, other_selected, outside = [uuid.uuid4() for _ in range(3)]
    description_docs = {
        "none_matching": [outside, outside],
        "one_matching": [selected, outside],
        "same_document": [selected, selected],
        "multiple_documents": [selected, other_selected],
        "unscoped": [None, outside],
    }[scope]
    db_session.add_all(
        [
            RelationshipDescription(
                id=uuid.uuid4(),
                relationship_id=rel.id,
                description=f"Observation {i}",
                document_id=document_id,
                source_chunk_hashes=[f"chunk-{i}"],
            )
            for i, document_id in enumerate(description_docs)
        ]
    )
    await db_session.commit()
    monkeypatch.setattr(
        graph_rag,
        "AsyncSessionLocal",
        lambda: AsyncSession(bind=db_session.bind, expire_on_commit=False),
    )
    candidates = AsyncMock(return_value=[("Source", [], 0.9), ("Target", [], 0.8)])
    monkeypatch.setattr(graph_rag, "_top_entity_candidates", candidates)
    rel_id = str(rel.id)
    entity_ids = {str(source.id), str(target.id)}
    state = graph_rag.GraphQueryState(
        discovered_entity_ids=entity_ids,
        entity_relevance={str(source.id): 0.3, str(target.id): 0.2},
        traversed_rel_ids=[rel_id] if traversed else [],
        rel_score_cache={rel_id: 0.42},
        rel_combined_score_cache={rel_id: 0.21},
    )
    document_ids = None if scope == "unscoped" else [selected]
    if scope == "multiple_documents":
        document_ids.append(other_selected)
    result = await graph_rag._filter_relationship_state_by_entity_score(
        collection,
        state,
        [],
        min_entity_score=0.5,
        document_ids=document_ids,
    )
    assert result.discovered_entity_ids == entity_ids
    candidates.assert_awaited_once()
    assert candidates.await_args.kwargs["document_ids"] == document_ids
    if scope == "none_matching":
        assert result.traversed_rel_ids == []
        assert result.rel_score_cache == {}
        assert result.rel_combined_score_cache == {}
    else:
        assert result.traversed_rel_ids == [rel_id]
        assert result.rel_score_cache == {rel_id: 0.42}
        assert result.rel_combined_score_cache == {rel_id: 0.21}

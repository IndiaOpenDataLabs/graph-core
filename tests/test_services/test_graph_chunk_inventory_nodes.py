"""Custom Graph RAG projects every resolved inventory entity to graph storage."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graph_core.models.graph_rag import GraphEntity
from graph_core.services.graph.ingestion import chunk_processor
from graph_core.services.graph_rag.extractor import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)


@pytest.mark.parametrize("raw_cache_hit", [False, True])
@pytest.mark.parametrize("name_cache_hit", [False, True])
@pytest.mark.parametrize("with_relationship", [False, True])
@pytest.mark.parametrize("missing_endpoint", [False, True])
async def test_inventory_nodes_upserted_independently_of_edges(
    db_session,
    test_graph_rag_collection,
    monkeypatch,
    raw_cache_hit,
    name_cache_hit,
    with_relationship,
    missing_endpoint,
):
    collection = test_graph_rag_collection
    document_id = uuid.uuid4()
    document_path = "teachings/duty.txt"
    chunk_hash = "a" * 64
    duty_id = uuid.uuid4()
    names_by_id = {duty_id: "Duty"}
    # Two extracted names resolve to the same canonical entity. The projection
    # must deduplicate by resolved ID, not by source name or title casing.
    ids_by_name = {"Dharma": duty_id, "Duty": duty_id}
    entities = [ExtractedEntity("Dharma", "CONCEPT", "An obligation to act.")]
    relationships = []
    if with_relationship:
        entities.append(
            ExtractedEntity("Duty", "CONCEPT", "Action one should perform.")
        )
        for name in ["Krishna", "Arjuna"]:
            entity_id = uuid.uuid4()
            names_by_id[entity_id] = name
            ids_by_name[name] = entity_id
            if not (missing_endpoint and name == "Arjuna"):
                entities.append(ExtractedEntity(name, "PERSON", f"The participant {name}."))
        relationships.append(
            ExtractedRelationship(
                source_name="Krishna",
                target_name="Arjuna",
                source_description="The participant Krishna.",
                target_description="The participant Arjuna.",
                description="Krishna teaches Arjuna about duty.",
                keywords=["teaching"],
                rel_type="TEACHES",
            )
        )

    for entity_id, name in names_by_id.items():
        db_session.add(
            GraphEntity(
                id=entity_id,
                collection_id=collection.id,
                canonical_name=name,
                primary_type="CONCEPT" if entity_id == duty_id else "PERSON",
            )
        )
    await db_session.commit()
    extraction = ExtractionResult(entities, relationships)

    async def resolve_entities(session, inventory, *args, **kwargs):
        return [SimpleNamespace(entity_id=ids_by_name[entity.name],
                                canonical_name=names_by_id[ids_by_name[entity.name]], is_new=False)
                for entity in inventory]

    resolver = SimpleNamespace(
        resolve_entities=AsyncMock(side_effect=resolve_entities),
        resolve_relationships=AsyncMock(return_value=[SimpleNamespace(relationship_id=uuid.uuid4())
                                                     for _ in relationships]),
        flush_support=AsyncMock(),
    )
    name_cache = SimpleNamespace(
        get=AsyncMock(
            side_effect=lambda name: ids_by_name.get(name) if name_cache_hit else None
        ),
        set_many=AsyncMock(),
    )
    extractor = SimpleNamespace(
        extract_with_gleaning=AsyncMock(return_value=extraction)
    )
    storage = SimpleNamespace(upsert_nodes=AsyncMock(), upsert_edges=AsyncMock())
    raw_cache = AsyncMock(return_value=extraction if raw_cache_hit else None)
    monkeypatch.setattr(chunk_processor, "_get_raw_extraction", raw_cache)
    monkeypatch.setattr(chunk_processor, "_save_raw_extraction", AsyncMock())
    monkeypatch.setattr(chunk_processor, "LLMGraphExtractor", lambda **_: extractor)
    monkeypatch.setattr(
        chunk_processor, "IncrementalEntityResolver", lambda **_: resolver
    )
    monkeypatch.setattr(chunk_processor, "EntityNameCache", lambda _: name_cache)
    monkeypatch.setattr(chunk_processor, "get_graph_storage", lambda _: storage)
    monkeypatch.setattr(
        chunk_processor,
        "_resolve_embedding_provider",
        AsyncMock(
            return_value=SimpleNamespace(embed_query=AsyncMock(return_value=[0.1, 0.2]))
        ),
    )
    monkeypatch.setattr(chunk_processor, "resolve_llm_provider", AsyncMock())
    monkeypatch.setattr(
        chunk_processor._graph_rag_vectors, "upsert_chunk_embedding", AsyncMock()
    )

    result = await chunk_processor._ingest_graph_chunk(
        text="Duty is action one should perform.",
        collection=collection,
        chunk_hash=chunk_hash,
        report=None,
        document_id=document_id,
        document_path=document_path,
    )

    storage.upsert_nodes.assert_awaited_once()
    nodes = storage.upsert_nodes.await_args.args[0]
    assert {node["id"]: node["name"] for node in nodes} == {
        str(entity_id): name for entity_id, name in names_by_id.items()
    }
    assert len(nodes) == len(names_by_id)
    assert all(node["collection_id"] == str(collection.id) for node in nodes)
    assert all(node["document_id"] == str(document_id) for node in nodes)
    assert all(node["document_path"] == document_path for node in nodes)
    assert result.entity_count == len(entities)
    assert result.relationship_count == len(relationships)
    # Cache hits are hints, never an identity authorization bypass.
    resolver.resolve_entities.assert_awaited_once()
    assert resolver.resolve_entities.await_args.args[2] == chunk_hash
    resolver.flush_support.assert_awaited_once()
    if with_relationship and missing_endpoint:
        planned = resolver.resolve_entities.await_args.args[1]
        assert next(entity for entity in planned if entity.name == "Arjuna").description == "The participant Arjuna."
    assert extractor.extract_with_gleaning.await_count == (0 if raw_cache_hit else 1)
    if with_relationship:
        storage.upsert_edges.assert_awaited_once()
        edges = storage.upsert_edges.await_args.args[0]
        assert len(edges) == 1
        assert edges[0]["source_id"] == str(ids_by_name["Krishna"])
        assert edges[0]["target_id"] == str(ids_by_name["Arjuna"])
        assert edges[0]["rel_type"] == "TEACHES"
    else:
        assert resolver.resolve_relationships.await_args.args[1] == []
        storage.upsert_edges.assert_not_awaited()

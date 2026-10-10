"""Offline batching regressions: count forward passes, not just decision accuracy."""

import json
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import select

from graph_core.config import settings
from graph_core.database import AsyncSessionLocal
from graph_core.decisions import GraphDecisions, SystemOneDecisionProvider
from graph_core.decisions.batching import (
    estimated_tokens,
    identity_record,
    support_record,
)
from graph_core.decisions.systemone import DecisionError
from graph_core.models.graph_rag import (
    EntityAlias,
    EntityDescription,
    EntityResolutionDecision,
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
)
from graph_core.services.graph_rag.entity_resolver import IncrementalEntityResolver
from graph_core.services.graph_rag.extractor import ExtractedEntity


@pytest.fixture(autouse=True)
def offline_slots(monkeypatch):
    @asynccontextmanager
    async def slot():
        yield

    monkeypatch.setattr("graph_core.decisions.systemone.decision_model_call_slot", slot)


def scorer(calls, choose=None, before_request=None):
    def handle(request):
        if before_request:
            before_request()
        body = json.loads(request.content)
        calls.append(body)
        answers = {}
        for key, question in body["questions"].items():
            options = list(question["criteria"])
            choice = (
                choose(body, key)
                if choose
                else ("different" if "different" in options else "supported")
            )
            answers[key] = {
                "type": "choice",
                "choice": choice,
                "probabilities": {
                    option: 0.99 if option == choice else 0.005 for option in options
                },
            }
        return httpx.Response(200, json={"answers": answers})

    return GraphDecisions(SystemOneDecisionProvider(httpx.MockTransport(handle)))


def claims(count, passage="Shared original evidence."):
    return [
        {
            "id": f"claim_{i}",
            "claim": {"description": f"Fact {i}"},
            "evidence": [
                {
                    "source_passage": passage,
                    "chunk_hash": "chunk",
                    "support_confidence": 1,
                    "support_assessment": {"choice": "supported"},
                }
            ],
        }
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_multiple_scores_share_a_forward_pass_and_source():
    calls = []
    result = await scorer(calls).support_many(claims(12))
    assert len(result) == 12
    assert len(calls) == 1
    assert len(calls[0]["questions"]) == 12
    assert list(calls[0]["state"]["source_passages"].values()) == [
        "Shared original evidence."
    ]
    assert json.dumps(calls[0]).count("Shared original evidence.") == 1
    assert "support_confidence" not in json.dumps(calls[0])
    assert "support_assessment" not in json.dumps(calls[0])


@pytest.mark.asyncio
async def test_context_budget_splits_without_dropping_items(monkeypatch):
    items = claims(9, "source " * 80)
    budget = estimated_tokens(support_record(items[:3]))
    monkeypatch.setattr(settings, "decision_model_batch_token_budget", budget)
    calls = []
    result = await scorer(calls).support_many(items)
    assert set(result) == {item["id"] for item in items}
    assert len(calls) == 3
    assert all(estimated_tokens(call) <= budget for call in calls)
    assert all(len(call["questions"]) == 3 for call in calls)


@pytest.mark.asyncio
async def test_question_cap_splits_and_empty_batch_never_calls_model(monkeypatch):
    monkeypatch.setattr(settings, "decision_model_batch_max_questions", 3)
    calls = []
    provider = scorer(calls)
    assert await provider.support_many([]) == {}
    assert calls == []
    assert len(await provider.support_many(claims(7))) == 7
    assert [len(call["questions"]) for call in calls] == [3, 3, 1]


@pytest.mark.asyncio
async def test_oversized_single_item_fails_without_truncation(monkeypatch):
    monkeypatch.setattr(settings, "decision_model_batch_token_budget", 256)
    calls = []
    with pytest.raises(DecisionError, match="Evidence was not truncated"):
        await scorer(calls).support_many(claims(1, "source " * 2000))
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("packing", ["together", "question_cap", "token_budget"])
async def test_identity_uses_real_instructions_and_explicit_pair_bindings(
    monkeypatch, packing
):
    source = "Idā and Ida name the lunar nadi. Pingala is a distinct solar nadi."
    items = [
        {
            "id": f"identity_{i}",
            "incoming_id": "0",
            "candidate_id": str(i),
            "incoming": {
                "name": "Idā",
                "type": "NADI",
                "description": "The lunar nadi.",
            },
            "candidate": {"name": name, "type": "NADI", "descriptions": [description]},
        }
        for i, (name, description) in enumerate(
            [
                ("Ida", "The lunar nadi."),
                ("Pingala", "The solar nadi."),
                ("Lunar Nadi", "The channel associated with the moon."),
            ]
        )
    ]
    monkeypatch.setattr(settings, "decision_model_batch_token_budget", 6000)
    monkeypatch.setattr(settings, "decision_model_batch_max_questions", 64)
    if packing == "question_cap":
        monkeypatch.setattr(settings, "decision_model_batch_max_questions", 2)
    elif packing == "token_budget":
        monkeypatch.setattr(
            settings,
            "decision_model_batch_token_budget",
            estimated_tokens(identity_record(items[:2], source)),
        )
    calls = []
    decisions = await scorer(calls).identity_many(items, source)
    assert set(decisions) == {item["id"] for item in items}
    assert len(calls) == (1 if packing == "together" else 2)
    for call in calls:
        assert "referential identity, NOT semantic similarity" in call["instructions"]
        assert "task_instructions" not in call["state"]
        assert call["state"]["source_passage"] == source
        assert json.dumps(call, ensure_ascii=False).count(source) == 1
        # Packing estimates the builder's record; the provider may serialize
        # top-level keys in a different order (notably with Unicode names).
        packed = identity_record(
            [item for item in items if item["id"] in call["questions"]], source
        )
        assert call == packed
        assert estimated_tokens(packed) <= settings.decision_model_batch_token_budget
        for key, question in call["questions"].items():
            pair = next(p for p in call["state"]["pairs"] if p["id"] == key)
            instruction = question["instructions"]
            assert f"incoming_entities[{pair['incoming_id']!r}]" in instruction
            assert f"candidate_entities[{pair['candidate_id']!r}]" in instruction
            assert (
                repr(call["state"]["incoming_entities"][pair["incoming_id"]]["name"])
                in instruction
            )
            assert (
                repr(call["state"]["candidate_entities"][pair["candidate_id"]]["name"])
                in instruction
            )
            assert question["type"] == "choice"
            assert set(question["criteria"]) == {"same", "different", "uncertain"}
            assert (
                decisions[key].trace("entity_identity")["instructions"]
                == call["instructions"]
            )


def resolver_for(collection, calls, choose=None, before_request=None):
    embedding = SimpleNamespace(
        dimensions=256,
        embed_query=AsyncMock(return_value=[0.1, 0.2]),
        embed_documents=AsyncMock(
            side_effect=lambda texts: [[0.1, 0.2] for _ in texts]
        ),
    )
    resolver = IncrementalEntityResolver(
        embedding,
        collection.id,
        source_text="A shared source passage.",
        decisions=scorer(calls, choose, before_request),
    )
    for name in [
        "upsert_entity_embedding",
        "upsert_entity_centroid",
        "upsert_relationship_embedding",
    ]:
        setattr(resolver._vstore, name, AsyncMock())
    resolver._vstore.get_entity_centroid = AsyncMock(return_value=None)
    resolver._vstore.search_entity_centroids = AsyncMock(return_value=[])
    return resolver


@pytest.mark.asyncio
async def test_identity_candidates_are_batched_and_deduplicated(
    db_session, test_graph_rag_collection
):
    candidate = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Varuṇa",
        primary_type="deity",
    )
    db_session.add(candidate)
    db_session.add(
        EntityAlias(
            entity_id=candidate.id,
            collection_id=test_graph_rag_collection.id,
            alias_name="Varuna",
            source_chunk_hash="old",
        )
    )
    await db_session.commit()
    calls = []
    resolver = resolver_for(test_graph_rag_collection, calls)
    resolver._vstore.search_entity_centroids.return_value = [
        SimpleNamespace(distance=0.001, metadata={"entity_id": str(candidate.id)})
    ]
    result = await resolver.resolve_entities(
        db_session, [ExtractedEntity("Varuna", "deity", "")], "chunk"
    )
    assert result[0].entity_id != candidate.id
    assert len(calls) == 1
    assert (
        len(calls[0]["questions"]) == 1
    )  # Alias, centroid, and fuzzy paths propose the SAME pair.
    audits = (
        (await db_session.execute(select(EntityResolutionDecision))).scalars().all()
    )
    assert len(audits) == 1
    assert audits[0].decision["accepted"] is False


@pytest.mark.asyncio
async def test_identity_batch_applies_high_probability_merge(
    db_session, test_graph_rag_collection
):
    candidate = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Varuṇa",
        primary_type="deity",
    )
    db_session.add(candidate)
    await db_session.commit()
    calls = []
    resolver = resolver_for(
        test_graph_rag_collection, calls, choose=lambda body, key: "same"
    )
    resolver._vstore.search_entity_centroids.return_value = [
        SimpleNamespace(distance=0.001, metadata={"entity_id": str(candidate.id)})
    ]
    result = await resolver.resolve_entities(
        db_session, [ExtractedEntity("Varuna", "deity", "")], "chunk"
    )
    assert result[0].entity_id == candidate.id
    alias = (await db_session.execute(select(EntityAlias))).scalars().one()
    assert alias.identity_decision["probabilities"]["same"] == 0.99


@pytest.mark.asyncio
async def test_within_chunk_candidates_remain_bounded(
    db_session, test_graph_rag_collection
):
    calls = []
    resolver = resolver_for(test_graph_rag_collection, calls)
    # Fuzzy names are deliberately dissimilar; all embeddings are identical.
    entities = [
        ExtractedEntity(name, "concept", "")
        for name in [
            "Apple",
            "Bronze",
            "Cloud",
            "Dharma",
            "Elephant",
            "Fire",
            "Gravity",
            "Horizon",
            "Island",
            "Jungle",
        ]
    ]
    await resolver.resolve_entities(db_session, entities, "chunk")
    pairs = [pair for call in calls for pair in call["state"]["pairs"]]
    for index in range(10):
        assert sum(pair["incoming_id"] == str(index) for pair in pairs) <= 5
    assert (
        len(pairs) > 0
    )  # New in-chunk entities are candidates even before centroids are stored.


@pytest.mark.asyncio
async def test_chunk_persists_entities_and_relationships_without_source_validation(
    db_session, test_graph_rag_collection
):
    calls, lock_sessions = [], []

    def unlocked():
        assert all(not session.in_transaction() for session in lock_sessions)

    resolver = resolver_for(test_graph_rag_collection, calls, before_request=unlocked)
    resolver._decisions.support_many = AsyncMock(
        side_effect=AssertionError("Ingestion must not validate source support")
    )

    async def lock(session, *_):
        lock_sessions.append(session)

    resolver._acquire_entity_lock = lock
    kind = GraphRelationshipType(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_type="SYMBOLIZES",
    )
    db_session.add(kind)
    await db_session.commit()
    resolver._resolve_rel_type = AsyncMock(
        return_value=SimpleNamespace(
            relationship_type_id=kind.id, canonical_type="SYMBOLIZES"
        )
    )
    results = await resolver.resolve_entities(
        db_session,
        [
            ExtractedEntity("Agni", "deity", "Agni is sacred fire."),
            ExtractedEntity("Fire", "concept", "Fire is an element."),
        ],
        "chunk",
        defer_descriptions=True,
    )
    identity_calls = len(calls)
    relations = [
        {
            "source_entity_id": results[0].entity_id,
            "target_entity_id": results[1].entity_id,
            "description": "Agni symbolizes fire.",
            "keywords": ["fire"],
            "rel_type": "SYMBOLIZES",
        }
    ]
    rels = await resolver.resolve_relationships(
        db_session, relations, "chunk", defer_descriptions=True
    )
    assert len(calls) == identity_calls
    await resolver.flush_descriptions(db_session)
    assert len(calls) == identity_calls
    resolver._decisions.support_many.assert_not_awaited()
    rel = await db_session.get(GraphRelationship, rels[0].relationship_id)
    assert rel.confidence is None
    assert rel.weight == 1
    assert rel.support_count is None
    assert rel.score_metadata["source_count"] == 1
    descs = (await db_session.execute(select(EntityDescription))).scalars().all()
    assert len(descs) == 2
    assert all(desc.confidence is None for desc in descs)
    assert all(desc.score_metadata is None for desc in descs)
    assert all(
        desc.source_evidence[0]["source_passage"] == resolver._source_text
        for desc in descs
    )
    assert resolver._vstore.upsert_entity_embedding.await_count == 2
    assert resolver._vstore.upsert_relationship_embedding.await_count == 1
    await resolver.resolve_relationships(db_session, relations, "chunk")
    assert len(calls) == identity_calls
    assert resolver._vstore.upsert_relationship_embedding.await_count == 1


@pytest.mark.asyncio
async def test_concurrent_evidence_change_is_merged_without_model_validation(
    db_session, test_graph_rag_collection
):
    entity = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Agni",
        primary_type="deity",
    )
    db_session.add(entity)
    await db_session.commit()
    calls = []
    resolver = resolver_for(test_graph_rag_collection, calls)
    await resolver.resolve_entities(
        db_session,
        [ExtractedEntity("Agni", "deity", "Agni is fire.")],
        "chunk",
        defer_descriptions=True,
    )
    batch = resolver._ingestion_batch
    real_embed = batch._embed_changes
    injected = False

    async def concurrent_embed(plans):
        nonlocal injected
        await real_embed(plans)
        if not injected:
            injected = True
            async with AsyncSessionLocal(bind=db_session.bind) as writer:
                writer.add(
                    EntityDescription(
                        entity_id=entity.id,
                        description="Agni is fire.",
                        source_evidence=[
                            {
                                "chunk_hash": "concurrent",
                                "document_id": None,
                                "source_passage": "Another passage.",
                            }
                        ],
                    )
                )
                await writer.commit()

    batch._embed_changes = AsyncMock(side_effect=concurrent_embed)
    await resolver.flush_descriptions(db_session)
    assert not calls
    assert batch._embed_changes.await_count == 2
    desc = (await db_session.execute(select(EntityDescription))).scalars().one()
    assert {item["chunk_hash"] for item in desc.source_evidence} == {
        "chunk",
        "concurrent",
    }
    assert desc.weight == 2


@pytest.mark.asyncio
async def test_extracted_entity_is_indexed_without_consulting_source_validator(
    db_session, test_graph_rag_collection
):
    calls = []
    resolver = resolver_for(
        test_graph_rag_collection, calls, choose=lambda body, key: "contradicted"
    )
    await resolver.resolve_entities(
        db_session, [ExtractedEntity("Agni", "deity", "Incorrect claim.")], "chunk"
    )
    assert not calls
    resolver._vstore.upsert_entity_centroid.assert_awaited_once()
    resolver._vstore.upsert_entity_embedding.assert_awaited_once()
    desc = (await db_session.execute(select(EntityDescription))).scalars().one()
    assert desc.confidence is None
    assert desc.score_metadata is None
    assert desc.description == "Incorrect claim."
    assert desc.source_evidence[0]["source_passage"] == resolver._source_text


@pytest.mark.asyncio
async def test_multiple_descriptions_for_one_entity_share_transactional_centroid(
    db_session, test_graph_rag_collection
):
    calls = []
    resolver = resolver_for(
        test_graph_rag_collection,
        calls,
        choose=lambda body, key: "same" if "pairs" in body["state"] else "supported",
    )

    async def embed(texts):
        return [
            [0.0, 1.0] if text == "Agni: Second claim." else [1.0, 0.0]
            for text in texts
        ]

    resolver._embedding.embed_documents.side_effect = embed
    current = None
    sessions = []

    async def read_centroid(*args, **kwargs):
        sessions.append(kwargs["session"])
        return current

    async def write_centroid(**kwargs):
        nonlocal current
        current = kwargs["embedding"]
        assert kwargs["session"] is sessions[-1]

    resolver._vstore.get_entity_centroid.side_effect = read_centroid
    resolver._vstore.upsert_entity_centroid.side_effect = write_centroid
    resolved = await resolver.resolve_entities(
        db_session,
        [
            ExtractedEntity("Agni", "deity", "First claim."),
            ExtractedEntity("Agnee", "deity", "Second claim."),
        ],
        "chunk",
    )
    assert resolved[0].entity_id == resolved[1].entity_id
    assert current == [0.5, 0.5]
    assert sessions[0] is sessions[1]
    entity = await db_session.get(GraphEntity, resolved[0].entity_id)
    assert entity.description_count == 2
    assert len(calls) == 1  # Identity only; neither description is revalidated.


@pytest.mark.asyncio
async def test_deadlock_retry_does_not_repeat_identity_inference(
    db_session, test_graph_rag_collection, monkeypatch
):
    from sqlalchemy.exc import DBAPIError

    from graph_core.services.graph_rag import ingestion_batch

    class DeadlockError(Exception):
        sqlstate = "40P01"

    candidate = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Varuṇa",
        primary_type="deity",
    )
    db_session.add(candidate)
    await db_session.commit()
    calls = []
    resolver = resolver_for(
        test_graph_rag_collection, calls, choose=lambda body, key: "same"
    )
    resolver._vstore.search_entity_centroids.return_value = [
        SimpleNamespace(distance=0.001, metadata={"entity_id": str(candidate.id)})
    ]
    real_add_type = resolver._add_or_increment_type
    attempts = 0

    async def add_type(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DBAPIError("write", {}, DeadlockError())
        await real_add_type(*args)

    resolver._add_or_increment_type = add_type
    monkeypatch.setattr(ingestion_batch.asyncio, "sleep", AsyncMock())
    result = await resolver.resolve_entities(
        db_session, [ExtractedEntity("Varuna", "deity", "")], "chunk"
    )
    assert result[0].entity_id == candidate.id
    assert attempts == 2
    assert len(calls) == 1
    audits = (
        (await db_session.execute(select(EntityResolutionDecision))).scalars().all()
    )
    assert len(audits) == 1

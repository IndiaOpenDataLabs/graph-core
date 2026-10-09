"""Offline native-protocol and graph integration tests; no decision model server needed."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import select

from graph_core.decisions import (
    Decision,
    DecisionError,
    GraphDecisions,
    SystemOneDecisionProvider,
)
from graph_core.models.graph_rag import (
    EntityAlias,
    EntityDescription,
    EntityResolutionDecision,
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
    RelationshipDescription,
)
from graph_core.services.graph.query.decision_context import build_context
from graph_core.services.graph.query.graph_rag import GraphQueryState
from graph_core.services.graph_rag.entity_resolver import IncrementalEntityResolver
from graph_core.services.graph_rag.extractor import ExtractedEntity, LLMGraphExtractor


def native_provider(choose, calls, probability=0.9):
    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        answers = {}
        for key, question in body["questions"].items():
            options = list(question["criteria"])
            option = choose(body, key, options)
            probabilities = {o: (1 - probability) / (len(options) - 1) for o in options}
            probabilities[option] = probability
            answers[key] = {
                "type": "choice",
                "choice": option,
                "probabilities": probabilities,
            }
        return httpx.Response(200, json={"answers": answers})

    return SystemOneDecisionProvider(httpx.MockTransport(handle))


@pytest.mark.asyncio
async def test_native_endpoint_and_no_chat_requests():
    def handle(request):
        assert str(request.url) == "http://localhost:8081/v1/systemone"
        assert "messages" not in json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "decision-model",
                "answers": {
                    "support": {
                        "type": "choice",
                        "choice": "supported",
                        "probabilities": {
                            "supported": 0.9,
                            "contradicted": 0.05,
                            "uncertain": 0.05,
                        },
                    }
                },
            },
        )

    scorer = GraphDecisions(SystemOneDecisionProvider(httpx.MockTransport(handle)))
    decision = await scorer.support(
        {"source": "Agni", "target": "Fire"}, [{"source_passage": "Agni is fire."}]
    )
    assert decision.probabilities["supported"] == 0.9
    assert decision.trace("support")["provider"] == "systemone"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        {
            "choice": "same",
            "probabilities": {"same": 9, "different": -8, "uncertain": 0},
        },
        {
            "choice": "same",
            "probabilities": {"same": 0.1, "different": 0.8, "uncertain": 0.1},
        },
        {"choice": "same", "probabilities": {"same": 0.2}},
        {
            "choice": "same",
            "probabilities": {"same": True, "different": 0, "uncertain": 0},
        },
        {
            "choice": "same",
            "probabilities": {"same": 0.2, "different": 0.2, "uncertain": 0.2},
        },
    ],
)
async def test_invalid_scores_fail_closed(answer):
    provider = SystemOneDecisionProvider(
        httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"answers": {"identity": {"type": "choice", **answer}}}
            )
        )
    )
    with pytest.raises(DecisionError):
        await GraphDecisions(provider).identity(
            {"name": "Varuna"}, {"name": "Indra"}, "source"
        )


@pytest.mark.asyncio
async def test_absent_server_does_not_fall_back_to_text_generation():
    def handle(request):
        raise httpx.ConnectError("offline", request=request)

    provider = SystemOneDecisionProvider(httpx.MockTransport(handle))
    with pytest.raises(DecisionError, match="localhost:8081/v1/systemone"):
        await GraphDecisions(provider).support({"description": "claim"}, [])


@pytest.mark.asyncio
async def test_high_embedding_similarity_cannot_merge_varuna_into_indra(
    db_session, test_graph_rag_collection
):
    candidate = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Indra",
        primary_type="deity",
    )
    db_session.add(candidate)
    await db_session.commit()
    calls = []
    scorer = GraphDecisions(native_provider(lambda *_: "different", calls))
    resolver = IncrementalEntityResolver(
        SimpleNamespace(dimensions=256),
        test_graph_rag_collection.id,
        decisions=scorer,
        source_text="Varuna and Indra are distinct deities.",
    )
    resolver._vstore.search_entity_centroids = AsyncMock(
        return_value=[
            SimpleNamespace(
                distance=0.001,
                metadata={"entity_id": str(candidate.id), "canonical_name": "Indra"},
            )
        ]
    )
    match = await resolver._find_similar_entity(
        db_session,
        [0.1],
        "Varuna",
        "deity",
        description="Varuna governs order.",
        source_chunk_hash="chunk",
    )
    assert match is None
    await db_session.commit()
    audit = (await db_session.execute(select(EntityResolutionDecision))).scalars().one()
    assert audit.incoming_name == "Varuna"
    assert audit.decision["accepted"] is False
    assert not (await db_session.execute(select(EntityAlias))).scalars().all()


@pytest.mark.asyncio
async def test_only_high_probability_identity_decision_allows_alias(
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
    scorer = SimpleNamespace(
        identity=AsyncMock(
            return_value=Decision(
                "same", {"same": 0.99, "different": 0.005, "uncertain": 0.005}
            )
        )
    )
    resolver = IncrementalEntityResolver(
        SimpleNamespace(), test_graph_rag_collection.id, decisions=scorer
    )
    assert await resolver._same_entity(
        db_session, "Varuna", "deity", "order", candidate, "chunk"
    )
    await resolver._add_alias(db_session, candidate.id, "Varuna", "chunk")
    await db_session.commit()
    alias = (await db_session.execute(select(EntityAlias))).scalars().one()
    assert alias.entity_id == candidate.id
    assert alias.identity_decision["probabilities"]["same"] == 0.99


@pytest.mark.asyncio
async def test_confidence_and_support_count_never_share_units(
    db_session, test_graph_rag_collection
):
    source = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Agni",
    )
    target = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Fire",
    )
    kind = GraphRelationshipType(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_type="SYMBOLIZES",
    )
    db_session.add_all([source, target, kind])
    await db_session.commit()
    calls = []
    scorer = GraphDecisions(native_provider(lambda *_: "supported", calls))
    embedding = SimpleNamespace(embed_query=AsyncMock(return_value=[0.1]))
    resolver = IncrementalEntityResolver(
        embedding,
        test_graph_rag_collection.id,
        decisions=scorer,
        source_text="Agni symbolizes fire.",
    )
    resolver._resolve_rel_type = AsyncMock(
        return_value=SimpleNamespace(
            relationship_type_id=kind.id, canonical_type=kind.canonical_type
        )
    )
    resolver._vstore.upsert_relationship_embedding = AsyncMock()
    args = dict(
        source_entity_id=source.id,
        target_entity_id=target.id,
        description="Agni symbolizes fire.",
        keywords=["fire"],
        weight=0.01,
        rel_type="SYMBOLIZES",
        original_source_name="Agni",
        original_target_name="Fire",
    )
    first = await resolver.resolve_relationship(
        db_session, source_chunk_hash="one", **args
    )
    for chunk in ["two", "two"]:
        await resolver.resolve_relationship(db_session, source_chunk_hash=chunk, **args)
    rel = await db_session.get(GraphRelationship, first.relationship_id)
    assert rel.confidence == 0.9  # Not the text model's 0.01 and not the count 2.
    assert rel.weight == 90
    assert rel.support_count == 2
    desc = (await db_session.execute(select(RelationshipDescription))).scalars().one()
    assert desc.weight == 2
    assert len(desc.source_evidence) == 2
    assert desc.source_evidence[0]["original_target_name"] == "Fire"
    assert all("original_question" not in c["state"] for c in calls)


@pytest.mark.asyncio
async def test_query_filters_noise_even_with_high_legacy_weight(
    db_session, test_graph_rag_collection, monkeypatch
):
    agni = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Agni",
        primary_type="deity",
    )
    sun = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Sun",
        primary_type="deity",
    )
    kind = GraphRelationshipType(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_type="MANIFESTS_AS",
    )
    db_session.add_all([agni, sun, kind])
    await db_session.flush()
    noise = GraphRelationship(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        source_entity_id=sun.id,
        target_entity_id=sun.id,
        relationship_type_id=kind.id,
        rel_type=kind.canonical_type,
        weight=100,
    )
    db_session.add(noise)
    await db_session.flush()
    db_session.add_all(
        [
            EntityDescription(entity_id=agni.id, description="Agni is sacred fire."),
            EntityDescription(entity_id=sun.id, description="Sun manifests as itself."),
            EntityDescription(entity_id=agni.id, description="Sarasvati is a river."),
            RelationshipDescription(
                relationship_id=noise.id, description="Sun manifests as itself."
            ),
        ]
    )
    await db_session.commit()
    calls = []
    scorer = GraphDecisions(
        native_provider(
            lambda body, key, _: (
                "direct"
                if next(c for c in body["state"]["candidates"] if c["id"] == key)[
                    "name"
                ]
                == "Agni"
                and next(c for c in body["state"]["candidates"] if c["id"] == key)["descriptions"] == ["Agni is sacred fire."]
                else "irrelevant"
            ),
            calls,
        )
    )
    monkeypatch.setattr(
        "graph_core.services.graph.query.decision_context.GraphDecisions", lambda: scorer
    )
    state = GraphQueryState(
        {str(agni.id), str(sun.id)},
        {str(agni.id): 0.1, str(sun.id): 1.0},
        [str(noise.id)],
        {str(noise.id): 0.95},
        {str(noise.id): 1.0},
    )
    context, entities, relationships, _ = await build_context(
        state, test_graph_rag_collection, "What is Agni?"
    )
    assert entities == ["Agni"]
    assert relationships == []
    assert "Sun manifests" not in context
    assert "Sarasvati" not in context
    assert all(c["state"]["original_question"] == "What is Agni?" for c in calls)
    assert any(
        t["name"] == "Sun" and not t["included"] for t in state.relevance_decisions
    )


def test_gleaning_keeps_shorter_facts_instead_of_overwriting_them():
    merged, added = LLMGraphExtractor._merge_entities(
        [ExtractedEntity("Agni", "deity", "Agni is fire.")],
        [ExtractedEntity("Agni", "deity", "Agni mediates offerings to the devas.")],
    )
    assert added == 0
    assert "Agni is fire." in merged[0].description
    assert "Agni mediates" in merged[0].description


@pytest.mark.asyncio
async def test_corrupt_exact_alias_is_not_identity_authority(db_session, test_graph_rag_collection):
    indra = GraphEntity(id=uuid.uuid4(), collection_id=test_graph_rag_collection.id,
                        canonical_name="Indra", primary_type="deity")
    varuna = GraphEntity(id=uuid.uuid4(), collection_id=test_graph_rag_collection.id,
                         canonical_name="Varuṇa", primary_type="deity")
    db_session.add_all([indra, varuna]); await db_session.flush()
    db_session.add(EntityAlias(entity_id=indra.id, collection_id=test_graph_rag_collection.id,
                               alias_name="Varuna", source_chunk_hash="old"))
    await db_session.commit()
    calls = []
    scorer = GraphDecisions(native_provider(
        lambda body, *_: "same" if body["state"]["candidate"]["name"] == "Varuṇa" else "different",
        calls, probability=0.99,
    ))
    resolver = IncrementalEntityResolver(SimpleNamespace(dimensions=256, embed_query=AsyncMock(return_value=[0.1])),
                                        test_graph_rag_collection.id, decisions=scorer,
                                        source_text="Varuna and Indra are different deities.")
    resolver._vstore.search_entity_centroids = AsyncMock(return_value=[SimpleNamespace(
        distance=0.001, metadata={"entity_id": str(varuna.id)}
    )])
    resolver._add_description_and_update_centroid = AsyncMock()
    resolver._add_or_increment_type = AsyncMock()
    result = await resolver.resolve_entity(db_session, "Varuna", "deity", "Varuna governs order.", "new")
    assert result.entity_id == varuna.id
    audits = (await db_session.execute(select(EntityResolutionDecision))).scalars().all()
    assert {a.candidate_id: a.decision["accepted"] for a in audits} == {indra.id: False, varuna.id: True}
    # Legacy corruption is not silently rewritten. New evidence routes safely.
    alias = (await db_session.execute(select(EntityAlias).where(EntityAlias.alias_name == "Varuna"))).scalars().one()
    assert alias.entity_id == indra.id


@pytest.mark.asyncio
async def test_collective_confidence_does_not_count_contradictory_passage_as_support(db_session, test_graph_rag_collection):
    source = GraphEntity(id=uuid.uuid4(), collection_id=test_graph_rag_collection.id, canonical_name="Agni")
    target = GraphEntity(id=uuid.uuid4(), collection_id=test_graph_rag_collection.id, canonical_name="Fire")
    kind = GraphRelationshipType(id=uuid.uuid4(), collection_id=test_graph_rag_collection.id, canonical_type="SYMBOLIZES")
    db_session.add_all([source, target, kind]); await db_session.commit()
    calls = []
    def choose(body, *_):
        passages = body["state"]["source_evidence"]
        assert all("support_confidence" not in p and "support_assessment" not in p for p in passages)
        return "contradicted" if all(p["source_passage"] == "Not a fire symbol." for p in passages) else "supported"
    resolver = IncrementalEntityResolver(SimpleNamespace(embed_query=AsyncMock(return_value=[0.1])),
                                        test_graph_rag_collection.id,
                                        decisions=GraphDecisions(native_provider(choose, calls)),
                                        source_text="Not a fire symbol.")
    resolver._resolve_rel_type = AsyncMock(return_value=SimpleNamespace(relationship_type_id=kind.id, canonical_type=kind.canonical_type))
    resolver._vstore.upsert_relationship_embedding = AsyncMock()
    args = dict(source_entity_id=source.id, target_entity_id=target.id, description="Agni symbolizes fire.", keywords=[], weight=10, rel_type="SYMBOLIZES")
    result = await resolver.resolve_relationship(db_session, source_chunk_hash="contradiction", **args)
    resolver._source_text = "Agni symbolizes fire."
    await resolver.resolve_relationship(db_session, source_chunk_hash="support", **args)
    rel = await db_session.get(GraphRelationship, result.relationship_id)
    assert rel.confidence == 0.9
    assert rel.support_count == 1
    desc = (await db_session.execute(select(RelationshipDescription))).scalars().one()
    assert desc.weight == 2  # Two observed passages, only one supporting passage.


@pytest.mark.asyncio
async def test_text_rewriter_cannot_select_entities(monkeypatch):
    import graph_core.services.graph.query.graph_rag as query
    calls = []
    scorer = GraphDecisions(native_provider(lambda body, key, _: "direct" if next(
        c for c in body["state"]["candidates"] if c["id"] == key)["name"] == "Agni" else "irrelevant", calls))
    monkeypatch.setattr(query, "GraphDecisions", lambda: scorer)
    text_provider = SimpleNamespace(structured_extract=AsyncMock(return_value={
        "selected_entities": ["Sun"], "retrieval_subqueries": ["What is Agni?"]
    }))
    result = await query._interpret_mix_queries("What is Agni?", [("Agni", "Fire", 0.1), ("Sun", "Sun", 1)], text_provider)
    assert result.selected_entities == ["Agni"]
    schema = text_provider.structured_extract.await_args.kwargs["schema"]
    assert set(schema["properties"]) == {"retrieval_subqueries"}

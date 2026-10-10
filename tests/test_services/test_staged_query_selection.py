"""Offline tests for native node gates, edge ranking, and edge-only gates."""

import json
import uuid
from contextlib import asynccontextmanager

import httpx
import pytest

from graph_core.config import settings
from graph_core.decisions import GraphDecisions, SystemOneDecisionProvider
from graph_core.decisions.batching import estimated_tokens
from graph_core.decisions.graph import passes_query_gate
from graph_core.decisions.systemone import DecisionError
from graph_core.models.graph_rag import (
    EntityDescription,
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
    RelationshipDescription,
)
from graph_core.services.graph.query import decision_context, graph_rag


@pytest.fixture(autouse=True)
def offline_slots(monkeypatch):
    @asynccontextmanager
    async def slot():
        yield

    monkeypatch.setattr("graph_core.decisions.systemone.decision_model_call_slot", slot)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probability,accepted", [(0, False), (0.5, False), (0.5001, True), (1, True)]
)
async def test_native_noul_and_strict_gate(probability, accepted):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200, json={"answers": {"node": {"type": "noul", "noul": probability}}}
        )

    provider = SystemOneDecisionProvider(httpx.MockTransport(handle))
    result = await provider.decide(
        {},
        {"node": {"type": "noul", "instructions": "Relevant?"}},
        instructions="Select graph nodes.",
    )
    assert passes_query_gate(result["node"]) is accepted
    assert result["node"].probabilities == {
        "true": probability,
        "false": 1 - probability,
    }
    assert calls[0]["instructions"] == "Select graph nodes."
    assert result["node"].trace("query_nodes")["instructions"] == "Select graph nodes."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        {"type": "noul", "noul": True},
        {"type": "noul", "noul": -0.1},
        {"type": "noul", "noul": 1.1},
        {"type": "noul"},
        {"type": "choice", "noul": 0.9},
    ],
)
async def test_invalid_noul_fails_closed(answer):
    provider = SystemOneDecisionProvider(
        httpx.MockTransport(
            lambda _: httpx.Response(200, json={"answers": {"node": answer}})
        )
    )
    with pytest.raises(DecisionError):
        await provider.decide({}, {"node": {"type": "noul"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("score,valid", [(1.5, True), (2, False), (True, False)])
async def test_native_score_validates_expected_value(score, valid):
    answer = {
        "type": "score",
        "score": score,
        "probabilities": {str(i): 0.25 for i in range(4)},
    }
    provider = SystemOneDecisionProvider(
        httpx.MockTransport(
            lambda _: httpx.Response(200, json={"answers": {"edge": answer}})
        )
    )
    questions = {
        "edge": {"type": "score", "criteria": ["None", "Weak", "Useful", "Direct"]}
    }
    if not valid:
        with pytest.raises(DecisionError):
            await provider.decide({}, questions)
    else:
        result = (await provider.decide({}, questions))["edge"]
        assert result.score == 1.5
        assert result.trace("query_edge_scores")["score"] == 1.5


@pytest.mark.asyncio
async def test_new_stages_preserve_batch_budget_and_instructions(monkeypatch):
    monkeypatch.setattr(settings, "decision_model_batch_token_budget", 6000)
    monkeypatch.setattr(settings, "decision_model_batch_max_questions", 2)
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        answers = {}
        for key, q in body["questions"].items():
            answers[key] = (
                {"type": "noul", "noul": 0.8}
                if q["type"] == "noul"
                else {
                    "type": "score",
                    "score": 3,
                    "probabilities": {"0": 0, "1": 0, "2": 0, "3": 1},
                }
            )
        return httpx.Response(200, json={"answers": answers})

    scorer = GraphDecisions(SystemOneDecisionProvider(httpx.MockTransport(handle)))
    items = [
        {
            "id": str(i),
            "name": "Actual node",
            "descriptions": ["Extracted description."],
        }
        for i in range(5)
    ]
    for method in [scorer.entity_relevance, scorer.edge_scores, scorer.edge_relevance]:
        calls.clear()
        assert len(await method("Original question", items)) == 5
        assert [len(c["questions"]) for c in calls] == [2, 2, 1]
        assert all(c["instructions"] and estimated_tokens(c) <= 6000 for c in calls)
        assert all(c["state"]["user_question"] == "Original question" for c in calls)
    monkeypatch.setattr(settings, "decision_model_batch_token_budget", 256)
    with pytest.raises(DecisionError, match="not truncated"):
        await scorer.entity_relevance(
            "Question",
            [
                {
                    "id": "huge",
                    "name": "Node",
                    "descriptions": ["Complete evidence. " * 2000],
                }
            ],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gate_mode,expected_count", [("all", 40), ("even", 20), ("none", 0)]
)
async def test_rank_full_pool_then_gate_only_top40_without_backfill(
    db_session,
    test_graph_rag_collection,
    monkeypatch,
    gate_mode,
    expected_count,
):
    collection = test_graph_rag_collection
    anchor = GraphEntity(
        id=uuid.uuid4(),
        collection_id=collection.id,
        canonical_name="Anchor",
        primary_type="CONCEPT",
    )
    kind = GraphRelationshipType(
        id=uuid.uuid4(), collection_id=collection.id, canonical_type="CONNECTS_TO"
    )
    db_session.add_all([anchor, kind])
    await db_session.flush()
    db_session.add(
        EntityDescription(entity_id=anchor.id, description="Actual anchor description.")
    )
    relationships = []
    for i in range(60):
        target = GraphEntity(
            id=uuid.uuid4(),
            collection_id=collection.id,
            canonical_name=f"Target {i:02}",
            primary_type="CONCEPT",
        )
        db_session.add(target)
        await db_session.flush()
        rel = GraphRelationship(
            id=uuid.uuid4(),
            collection_id=collection.id,
            source_entity_id=anchor.id,
            target_entity_id=target.id,
            relationship_type_id=kind.id,
            rel_type=kind.canonical_type,
            weight=100 if i == 0 else 1,
        )
        db_session.add(rel)
        await db_session.flush()
        db_session.add(
            RelationshipDescription(
                relationship_id=rel.id, description=f"Extracted relationship {i:02}."
            )
        )
        relationships.append(rel)
    # A case-variant name match must not admit a node that failed gate 1.
    rejected = GraphEntity(
        id=uuid.uuid4(),
        collection_id=collection.id,
        canonical_name="anchor",
        primary_type="CONCEPT",
    )
    db_session.add(rejected)
    await db_session.commit()
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        answers = {}
        for key, q in body["questions"].items():
            if "entities" in body["state"]:
                answers[key] = {"type": "noul", "noul": 0.9}
            else:
                candidate = body["state"]["relationships"][key]
                i = int(candidate["target"].split()[-1])
                assert candidate["description"] == f"Extracted relationship {i:02}."
                assert candidate["connected_anchors"] == ["Anchor"]
                if q["type"] == "score":
                    p = i / 59
                    answers[key] = {
                        "type": "score",
                        "score": 3 * p,
                        "probabilities": {"0": 1 - p, "1": 0, "2": 0, "3": p},
                    }
                else:
                    keep = gate_mode == "all" or (gate_mode == "even" and i % 2 == 0)
                    answers[key] = {"type": "noul", "noul": 0.8 if keep else 0.2}
        return httpx.Response(200, json={"answers": answers})

    scorer = GraphDecisions(SystemOneDecisionProvider(httpx.MockTransport(handle)))
    monkeypatch.setattr(decision_context, "GraphDecisions", lambda: scorer)
    monkeypatch.setattr(graph_rag, "AsyncSessionLocal", lambda: db_session)
    # Anchor expansion must NOT take 40 arbitrary weight-1 edges before scoring.
    expanded = await graph_rag._entity_anchor_state(
        collection, ["Anchor"], query_tokens=set(), entity_ids={str(anchor.id)}
    )
    assert len(expanded.traversed_rel_ids) == 60
    assert expanded.anchor_entity_ids == {str(anchor.id)}
    assert str(rejected.id) not in expanded.discovered_entity_ids
    # Simulate gate 1 having already run, as in mix mode: no duplicate node call.
    expanded.node_gate_complete = True
    expanded.entity_relevance[str(anchor.id)] = 0.9
    context, entities, labels, _ = await decision_context.build_context(
        expanded, collection, "Original question"
    )
    assert entities == ["Anchor"]
    assert len(labels) == expected_count
    assert len(expanded.traversed_rel_ids) == expected_count
    assert len(expanded.relevance_decisions) == 100  # 60 ranking + 40 binary traces
    score_calls = [
        c for c in calls if next(iter(c["questions"].values()))["type"] == "score"
    ]
    gate_calls = [
        c for c in calls if next(iter(c["questions"].values()))["type"] == "noul"
    ]
    assert sum(len(c["questions"]) for c in score_calls) == 60
    assert sum(len(c["questions"]) for c in gate_calls) == 40
    gated_targets = {
        e["target"] for c in gate_calls for e in c["state"]["relationships"].values()
    }
    assert gated_targets == {f"Target {i:02}" for i in range(20, 60)}
    assert "Extracted relationship 00." not in context
    assert relationships[0].weight == 100  # query ranking never changes stored weights
    assert not any("entities" in c["state"] for c in calls)
    assert all(c["state"]["user_question"] == "Original question" for c in calls)


@pytest.mark.asyncio
async def test_mix_hydrates_description_instead_of_exact_name_placeholder(
    db_session,
    test_graph_rag_collection,
    monkeypatch,
):
    entity = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Ida",
        primary_type="NADI",
    )
    db_session.add(entity)
    await db_session.flush()
    db_session.add(
        EntityDescription(
            entity_id=entity.id,
            description="The lunar nadi energized by left nostril breathing.",
        )
    )
    await db_session.commit()
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200,
            json={
                "answers": {
                    key: {"type": "noul", "noul": 0.85} for key in body["questions"]
                }
            },
        )

    scorer = GraphDecisions(SystemOneDecisionProvider(httpx.MockTransport(handle)))
    monkeypatch.setattr(graph_rag, "GraphDecisions", lambda: scorer)
    monkeypatch.setattr(graph_rag, "AsyncSessionLocal", lambda: db_session)
    result = await graph_rag._interpret_mix_queries(
        "Balance Ida?",
        [("Ida", "Ida", 1.0)],
        graph_rag.LocalEchoLLMProvider(),
        collection=test_graph_rag_collection,
    )
    assert result.selected_entities == ["Ida"]
    record = next(iter(calls[0]["state"]["entities"].values()))
    assert record["type"] == "NADI"
    assert record["descriptions"] == [
        "The lunar nadi energized by left nostril breathing."
    ]
    assert result.node_decisions[0]["probabilities"]["true"] == 0.85


@pytest.mark.asyncio
async def test_mix_keeps_all_passing_nodes_not_just_eight(monkeypatch):
    def handle(request):
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "answers": {
                    key: {"type": "noul", "noul": 0.5 if key == "candidate_19" else 0.8}
                    for key in body["questions"]
                }
            },
        )

    scorer = GraphDecisions(SystemOneDecisionProvider(httpx.MockTransport(handle)))
    monkeypatch.setattr(graph_rag, "GraphDecisions", lambda: scorer)
    result = await graph_rag._interpret_mix_queries(
        "Original question",
        [(f"Node {i}", "Real description", 1.0) for i in range(20)],
        graph_rag.LocalEchoLLMProvider(),
    )
    assert len(result.selected_entities) == 19
    assert "Node 19" not in result.selected_entities
    assert len(result.node_decisions) == 20


@pytest.mark.asyncio
async def test_description_hydration_respects_document_scope(
    db_session,
    test_graph_rag_collection,
):
    selected_document, other_document = uuid.uuid4(), uuid.uuid4()
    entity = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Ida",
        primary_type="NADI",
    )
    db_session.add(entity)
    await db_session.flush()
    db_session.add_all(
        [
            EntityDescription(
                entity_id=entity.id,
                document_id=other_document,
                description="Out of scope",
                weight=100,
            ),
            EntityDescription(
                entity_id=entity.id,
                document_id=selected_document,
                description="In scope",
                weight=1,
            ),
        ]
    )
    await db_session.commit()
    record = await decision_context.load_entity_candidate(
        db_session, entity, [selected_document]
    )
    assert record["descriptions"] == ["In scope"]
    assert record["provenance"][0]["document_id"] == str(selected_document)

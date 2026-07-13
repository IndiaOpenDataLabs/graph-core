import uuid

import pytest

from graph_core.services.graph import reasoning
from graph_core.services.graph.analytics import (
    NodeRecord,
    RelationshipRecord,
    _build_projection,
    _compute_projection_analytics,
)
from graph_core.services.graph.incremental_ingestion import (
    PredicateMappingInput,
    coalesce_predicate_mappings,
    merge_predicate_property_observation,
)
from graph_core.services.graph.reasoning import ReasoningSeed, activate_reasoning
from graph_core.services.graph_rag.extractor import (
    LLMGraphExtractor,
    normalize_predicate_properties,
)


def _id(value: int) -> uuid.UUID:
    return uuid.UUID(int=value)


@pytest.mark.asyncio
async def test_activation_blocks_a_rule_when_query_activates_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = {
        _id(1): reasoning._Node(_id(1), "CONDITION", "practice", "practice daily"),
        _id(2): reasoning._Node(_id(2), "EXCEPTION", "cold", "when suffering a cold"),
        _id(3): reasoning._Node(_id(3), "RULE", "rule", ""),
        _id(4): reasoning._Node(_id(4), "PROPOSITION", "drink milk", ""),
    }
    edges = [
        reasoning._Edge(_id(1), _id(3), "ANTECEDENT_OF"),
        reasoning._Edge(_id(2), _id(3), "BLOCKS"),
        reasoning._Edge(_id(3), _id(4), "CONCLUDES"),
    ]

    async def bounded_graph(*args, **kwargs):
        return nodes, edges

    monkeypatch.setattr(reasoning, "_bounded_graph", bounded_graph)
    result = await activate_reasoning(
        _id(100),
        "Can I drink milk when I have a cold?",
        [
            ReasoningSeed(
                proposition_id=_id(4),
                frame_text="Milk may be used as food during practice.",
                argument_ids=(_id(1),),
                exceptions=("when suffering a cold",),
            )
        ],
    )

    assert result["rules"][0]["status"] == "blocked"
    assert result["propositions"][0]["status"] == "blocked"
    assert result["operator"] == "prove_or_disprove"


def test_rule_projection_excludes_semantic_edges() -> None:
    nodes = [
        NodeRecord(_id(1), "condition", "CONDITION"),
        NodeRecord(_id(2), "rule", "RULE"),
        NodeRecord(_id(3), "claim", "PROPOSITION"),
        NodeRecord(_id(4), "entity", "CONCEPT"),
    ]
    relationships = [
        RelationshipRecord(_id(10), _id(1), "condition", _id(2), "rule", "ANTECEDENT_OF", 1),
        RelationshipRecord(_id(11), _id(2), "rule", _id(3), "claim", "CONCLUDES", 1),
        RelationshipRecord(_id(12), _id(3), "claim", _id(4), "entity", "RELATES_TO", 1),
    ]
    spec = {
        "name": "rule_dependency_directed",
        "directed": True,
        "node_policy": "reasoning_and_arguments",
        "edge_policy": "reasoning_only",
    }

    graph = _build_projection(nodes, relationships, spec)
    analytics = _compute_projection_analytics(graph)

    assert set(graph.edges) == {(_id(1), _id(2)), (_id(2), _id(3))}
    assert analytics["diagnostics"]["component_count"] == 2
    assert "pagerank" in analytics["node_metrics"]


def test_predicate_property_consensus_is_conservative() -> None:
    first = normalize_predicate_properties(
        {"causal": "causal", "transitivity": "transitive", "confidence": 0.9}
    )
    votes, consensus = merge_predicate_property_observation(None, first)

    assert consensus["causal"] == "causal"
    assert consensus["transitivity"] == "transitive"

    _, conflicted = merge_predicate_property_observation(
        votes,
        normalize_predicate_properties(
            {
                "causal": "non_causal",
                "transitivity": "non_transitive",
                "confidence": 0.9,
            }
        ),
    )
    assert conflicted["causal"] == "unknown"
    assert conflicted["transitivity"] == "unknown"

    predicate_id = _id(90)
    coalesced = coalesce_predicate_mappings(
        [
            PredicateMappingInput(
                "CAUSES",
                predicate_id,
                confidence=0.9,
                inferred_properties=first,
            ),
            PredicateMappingInput(
                "CAUSES",
                predicate_id,
                confidence=0.9,
                inferred_properties=normalize_predicate_properties(
                    {"causal": "non_causal", "confidence": 0.9}
                ),
            ),
        ]
    )
    assert coalesced[0].inferred_properties["causal"] == "unknown"


def test_predicate_properties_are_specific_to_each_emitted_type() -> None:
    base = {
        "source": {"name": "A", "description": "A"},
        "target": {"name": "B", "description": "B"},
        "description": "A affects B.",
        "keywords": [],
        "weight": 1.0,
        "rel_type": [
            {
                "name": "CAUSES",
                "predicate_properties": {
                    "causal": "causal",
                    "confidence": 0.95,
                },
            },
            {
                "name": "ASSOCIATED_WITH",
                "predicate_properties": {
                    "causal": "non_causal",
                    "confidence": 0.9,
                },
            },
        ],
    }

    relationships = LLMGraphExtractor._extract_generic_relationships(
        [base], [], None
    )

    assert [item.predicate_properties["causal"] for item in relationships] == [
        "causal",
        "non_causal",
    ]


def test_causal_projection_and_predicate_derivations_use_properties() -> None:
    nodes = [NodeRecord(_id(value), f"node-{value}", "CONCEPT") for value in range(1, 4)]
    properties = {
        "causal": "causal",
        "symmetry": "asymmetric",
        "transitivity": "transitive",
    }
    relationships = [
        RelationshipRecord(_id(10), _id(1), "a", _id(2), "b", "CAUSES", 1, predicate_properties=properties),
        RelationshipRecord(_id(11), _id(2), "b", _id(3), "c", "CAUSES", 1, predicate_properties=properties),
        RelationshipRecord(_id(12), _id(1), "a", _id(3), "c", "MENTIONS", 1),
    ]
    spec = {
        "name": "causal_semantics_directed",
        "directed": True,
        "node_policy": "semantic_entities",
        "edge_policy": "non_reasoning",
        "property_filter": {"causal": "causal"},
    }

    graph = _build_projection(nodes, relationships, spec)
    derived = reasoning._predicate_derivations(
        [
            reasoning._Edge(_id(1), _id(2), "CAUSES", properties),
            reasoning._Edge(_id(2), _id(3), "CAUSES", properties),
        ]
    )

    assert set(graph.edges) == {(_id(1), _id(2)), (_id(2), _id(3))}
    assert derived == [
        {
            "source": str(_id(1)),
            "target": str(_id(3)),
            "predicate": "CAUSES",
            "derivation": "transitivity",
        }
    ]

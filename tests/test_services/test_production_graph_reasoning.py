import uuid

import pytest

from graph_core.services.graph import reasoning
from graph_core.services.graph.analytics import (
    NodeRecord,
    RelationshipRecord,
    _build_projection,
    _compute_projection_analytics,
)
from graph_core.services.graph.reasoning import ReasoningSeed, activate_reasoning


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

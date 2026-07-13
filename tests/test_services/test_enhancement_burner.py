import uuid

from graph_core.scripts.vedas_enhance_burner import (
    PROJECTION_SPECS,
    Edge,
    Node,
    build_projection,
    compile_community_frame_values,
    compute_analytics,
    spec_hash,
)


def _id(value: int) -> uuid.UUID:
    return uuid.UUID(int=value)


def test_projection_separates_semantic_and_reasoning_structure() -> None:
    nodes = [
        Node(_id(1), "PRACTICE"),
        Node(_id(2), "OUTCOME"),
        Node(_id(3), "PROPOSITION"),
        Node(_id(4), "EVIDENCE_CHUNK"),
    ]
    edges = [
        Edge(_id(1), _id(2), "IMPROVES", 1),
        Edge(_id(3), _id(1), "SUBJECT", 1),
        Edge(_id(4), _id(3), "SUPPORTS", 1),
    ]

    semantic = build_projection(nodes, edges, PROJECTION_SPECS[0])
    reasoning = build_projection(nodes, edges, PROJECTION_SPECS[2])

    assert set(semantic.nodes) == {_id(1), _id(2)}
    assert set(semantic.edges) == {(_id(1), _id(2))}
    assert _id(3) in reasoning
    assert (_id(3), _id(1)) in reasoning.edges
    assert _id(4) not in reasoning


def test_analytics_and_spec_hash_are_deterministic() -> None:
    nodes = [Node(_id(value), "CONCEPT") for value in range(1, 5)]
    edges = [
        Edge(_id(1), _id(2), "RELATED_TO", 1),
        Edge(_id(2), _id(3), "RELATED_TO", 1),
        Edge(_id(3), _id(1), "RELATED_TO", 1),
        Edge(_id(3), _id(4), "RELATED_TO", 1),
    ]
    graph = build_projection(nodes, edges, PROJECTION_SPECS[1])

    analytics = compute_analytics(graph)

    assert analytics["diagnostics"]["component_count"] == 1
    assert analytics["node_metrics"]["is_articulation"][_id(3)] == 1.0
    assert spec_hash(PROJECTION_SPECS[0]) == spec_hash(dict(PROJECTION_SPECS[0]))


def test_community_frames_preserve_retrieval_scale_without_becoming_evidence() -> None:
    from graph_core.models.collection import Collection
    from graph_core.models.incremental_graph import GraphVersion

    collection = Collection(id=_id(20), namespace_id=_id(21), name="test")
    version = GraphVersion(id=_id(22), collection_id=collection.id, version=1)
    nodes = [Node(_id(1), "CONCEPT", "Agni"), Node(_id(2), "CONCEPT", "Devas")]
    edges = [Edge(_id(1), _id(2), "SERVES_AS", 1)]
    analytics = {
        "communities": [{_id(1), _id(2)}],
        "node_metrics": {"pagerank": {_id(1): 0.6, _id(2): 0.4}},
    }

    frames, arguments = compile_community_frame_values(
        collection, version, nodes, edges, analytics
    )

    assert frames[0]["executable_status"] == "navigation_only"
    assert "Agni" in frames[0]["frame_text"]
    assert [argument["role"] for argument in arguments] == ["member", "member"]

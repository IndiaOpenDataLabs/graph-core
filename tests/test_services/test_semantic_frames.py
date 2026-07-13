import uuid

from graph_core.services.graph.analytics import (
    NodeRecord,
    RelationshipRecord,
    _build_semantic_communities,
)
from graph_core.services.graph.query.graph_rag import (
    SemanticFrameEvidence,
    _frame_query_tokens,
    _semantic_frame_context,
    _semantic_frame_suffix,
)
from graph_core.services.graph.semantic_frames import build_proposition_frame


def _id(value: int) -> uuid.UUID:
    return uuid.UUID(int=value)


def test_proposition_frame_preserves_retrieval_text_and_executable_links() -> None:
    frame = build_proposition_frame(
        collection_id=_id(1),
        chunk_hash="a" * 64,
        relationship_index=2,
        proposition_entity_id=_id(2),
        source_relationship_id=_id(3),
        source_entity_id=_id(4),
        source_name="Parser",
        target_entity_id=_id(5),
        target_name="Syntax tree",
        predicate="PRODUCES",
        description="The parser produces a syntax tree from valid input.",
        polarity="positive",
        modality="asserted",
        conditions=("input is valid",),
        exceptions=("parse failure",),
        scopes=("compile phase",),
    )

    assert frame.title == "Parser produces Syntax tree"
    assert "Conditions: input is valid" in frame.frame_text
    assert frame.proposition_entity_id == _id(2)
    assert frame.source_relationship_id == _id(3)
    assert [argument.role for argument in frame.arguments] == ["subject", "object"]
    assert _frame_query_tokens("How does Agni take offerings to the devas?") == {
        "agni",
        "devas",
        "offerings",
    }


def test_structural_communities_are_deterministic_navigation_regions() -> None:
    nodes = [NodeRecord(_id(value), f"node-{value}") for value in range(1, 5)]
    nodes.append(NodeRecord(_id(5), "evidence", "EVIDENCE_CHUNK"))
    relationships = [
        RelationshipRecord(_id(10), _id(1), "node-1", _id(2), "node-2", "CALLS", 4),
        RelationshipRecord(_id(11), _id(3), "node-3", _id(4), "node-4", "USES", 4),
        RelationshipRecord(_id(12), _id(5), "evidence", _id(1), "node-1", "SUPPORTS", 9),
    ]

    first = _build_semantic_communities(nodes, relationships)
    second = _build_semantic_communities(nodes, relationships)

    assert first == second
    assert {tuple(item["member_names"]) for item in first} == {
        ("node-1", "node-2"),
        ("node-3", "node-4"),
    }


def test_query_context_separates_proof_frames_from_navigation_frames() -> None:
    proposition = SemanticFrameEvidence(
        frame_id=_id(1),
        frame_kind="proposition",
        title="Parser produces Syntax tree",
        frame_text="Parser produces a syntax tree when input is valid.",
        predicate="PRODUCES",
        score=0.8,
        proposition_id=_id(2),
        source_relationship_id=_id(3),
        executable_status="grounded_binary",
        polarity="positive",
        modality="conditional",
        conditions=("input is valid",),
        exceptions=(),
        arguments=(("subject", _id(4), "Parser"), ("object", _id(5), "Syntax tree")),
    )
    community = SemanticFrameEvidence(
        frame_id=_id(6),
        frame_kind="community_summary",
        title="Community: Parser, Syntax tree",
        frame_text="Use this community only to locate propositions.",
        predicate="",
        score=0.7,
        proposition_id=None,
        source_relationship_id=None,
        executable_status="navigation_only",
        polarity="unknown",
        modality="navigation",
        conditions=(),
        exceptions=(),
        arguments=(("member", _id(4), "Parser"),),
    )

    context, names, relationship_ids, discovered, _ = _semantic_frame_context(
        [proposition, community]
    )

    assert "Semantic Proposition Evidence" in context
    assert "Internal Community Navigation" in context
    assert "not evidence" in context
    assert names == ["Parser", "Syntax tree"]
    assert relationship_ids == [str(_id(3))]
    assert str(_id(2)) in discovered
    assert _semantic_frame_suffix(f"Context:\nbase\n\n{context}") == context

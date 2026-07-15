"""Run bounded graph-native reasoning over the incrementally compiled graph.

The embedder is used only to resolve initial seeds. Reasoning then operates on a
bounded SQL-loaded working graph and explicit proposition/rule structure. The
LLM is optional and, when requested, only verbalizes the completed trace.

Usage:
    uv run python -m graph_core.scripts.vedas_reasoning_burner \
        --question "When should I practice nadi shodhana?"
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import math
import re
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from sqlalchemy import case, func, or_, select

from graph_core.database import AsyncSessionLocal
from graph_core.models.collection import Collection
from graph_core.models.graph_rag import (
    EntityAlias,
    EntityDescription,
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
    RelationshipDescription,
)
from graph_core.models.incremental_graph import (
    GraphCommunity,
    GraphCommunityMembership,
    GraphFrameArgument,
    GraphNodeMetric,
    GraphProjectionSnapshot,
    GraphSemanticFrame,
    GraphVersion,
)
from graph_core.services.graph.query import graph_rag as query_logic
from graph_core.storage.graph_rag_vectors import GraphRAGVectorStore

DEFAULT_COLLECTION_ID = uuid.UUID("855f1950-ff65-4f47-9549-9a20f9a1332d")
STRUCTURAL_REL_TYPES = {
    "ANTECEDENT_OF",
    "APPLIES_TO",
    "BLOCKS",
    "CONCLUDES",
    "CONDITION",
    "EXCEPTION",
    "OBJECT",
    "SUBJECT",
    "SUPPORTS",
}
PROVENANCE_TYPES = {
    "ASSERTION",
    "CONTEXT",
    "EVIDENCE_CHUNK",
    "SOURCE_DOCUMENT",
    "SOURCE_FOLDER",
    "SOURCE_SECTION",
}
REASONING_TYPES = {"CONDITION", "EXCEPTION", "PROPOSITION", "RULE", "SCOPE"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "being",
    "can",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "should",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "why",
    "with",
}


@dataclass(frozen=True)
class QueryPlan:
    operator: str
    question: str
    state_tokens: tuple[str, ...]
    desired_outputs: tuple[str, ...]


@dataclass(frozen=True)
class FrameArgumentPattern:
    role: str
    entity_id: uuid.UUID | None = None
    variable_name: str | None = None
    literal_value: Any | None = None


@dataclass(frozen=True)
class FrameSeed:
    id: uuid.UUID
    kind: str
    title: str
    text: str
    predicate: str
    score: float
    proposition_id: uuid.UUID | None
    argument_ids: tuple[uuid.UUID, ...]
    executable_status: str
    conditions: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    arguments: tuple[FrameArgumentPattern, ...] = ()


@dataclass(frozen=True)
class GoalPattern:
    predicate: str
    arguments: tuple[FrameArgumentPattern, ...]
    source: str = "question"


@dataclass
class WorkingNode:
    id: uuid.UUID
    name: str
    node_type: str
    description: str = ""
    seed_score: float = 0.0
    distance: int = 999
    pagerank: float = 0.0


@dataclass(frozen=True)
class WorkingEdge:
    id: uuid.UUID
    source_id: uuid.UUID
    target_id: uuid.UUID
    rel_type: str
    weight: int
    description: str = ""


@dataclass
class WorkingGraph:
    nodes: dict[uuid.UUID, WorkingNode] = field(default_factory=dict)
    edges: dict[uuid.UUID, WorkingEdge] = field(default_factory=dict)


@dataclass(frozen=True)
class Limits:
    max_seeds: int = 12
    max_hops: int = 3
    max_nodes: int = 500
    max_edges: int = 900
    max_landmarks: int = 4
    max_conclusions: int = 30
    max_goal_depth: int = 4
    max_goal_states: int = 128
    max_goal_matches: int = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded reasoning over vedas2")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--collection-id", type=uuid.UUID, default=DEFAULT_COLLECTION_ID
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=None,
        help="Override the production query plan's bounded reasoning depth.",
    )
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help="Print the graph reasoning trace without calling the LLM.",
    )
    parser.add_argument("--answer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--ungrounded-plan",
        action="store_true",
        help="Use production's current one-shot planner for comparison.",
    )
    return parser.parse_args()


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{2,}", text.casefold())
        if token not in STOPWORDS
    }


def _goal_token(value: str) -> str:
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        return value[:-1]
    if value.endswith("ing") and len(value) > 5:
        return value[:-3]
    return value


def goal_tokens(text: str) -> set[str]:
    return {_goal_token(token) for token in tokenize(text)}


def compile_query(question: str) -> QueryPlan:
    lowered = question.casefold()
    if any(value in lowered for value in ("redesign", "refactor", "restructure")):
        operator = "redesign"
        outputs = ("constraints", "failure_modes", "rewrite_candidates")
    elif any(
        value in lowered
        for value in (" vs ", "versus", "which", "when should", "how should")
    ):
        operator = "choose"
        outputs = ("conditions", "exceptions", "effects", "comparison")
    elif any(
        value in lowered for value in ("good", "bad", "ugly", "evaluate", "assess")
    ):
        operator = "evaluate"
        outputs = ("support", "harm", "constraints", "severity")
    elif any(
        value in lowered
        for value in (
            "how and when",
            "why",
            "how does",
            "how to",
            "cause",
            "mechanism",
            "first principles",
        )
    ):
        if any(value in lowered for value in ("how and when", "how to")):
            operator = "procedure"
            outputs = ("steps", "timing", "conditions", "exceptions")
        else:
            operator = "explain"
            outputs = ("causes", "mechanisms", "conditions", "exceptions")
    else:
        operator = "prove_or_disprove"
        outputs = ("support", "attack", "conditions", "exceptions")
    return QueryPlan(
        operator=operator,
        question=question,
        state_tokens=tuple(sorted(tokenize(question))),
        desired_outputs=outputs,
    )


def seed_token_coverage(
    plan: QueryPlan,
    seeds: list[WorkingNode],
    frame_seeds: list[FrameSeed] | None = None,
) -> tuple[str, ...]:
    entity_tokens: set[str] = set()
    for seed in seeds:
        entity_tokens.update(goal_tokens(seed.name))
    for frame in frame_seeds or []:
        entity_tokens.update(
            goal_tokens(f"{frame.title} {frame.predicate} {frame.text}")
        )
    return tuple(
        sorted(
            token for token in plan.state_tokens if _goal_token(token) in entity_tokens
        )
    )


async def load_collection(
    collection_id: uuid.UUID,
) -> tuple[Collection, GraphVersion]:
    async with AsyncSessionLocal() as session:
        collection = await session.get(Collection, collection_id)
        if collection is None:
            raise ValueError(f"Collection {collection_id} not found")
        version = (
            await session.execute(
                select(GraphVersion)
                .where(GraphVersion.collection_id == collection_id)
                .order_by(GraphVersion.version.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if version is None:
            raise ValueError("Collection has no graph version")
    return collection, version


async def resolve_seeds(
    collection: Collection, question: str, limit: int, min_score: float
) -> list[WorkingNode]:
    provider = await query_logic._resolve_embedding_provider(collection)
    embedding = await query_logic._embed_entity_query(provider, question)
    hits = await GraphRAGVectorStore().search_entity_embeddings(
        collection.id, embedding, top_k=max(limit * 4, 20)
    )
    best: dict[uuid.UUID, tuple[float, str, str]] = {}
    for hit in hits:
        entity_id = uuid.UUID(hit.metadata["entity_id"])
        score = 1.0 - hit.distance
        current = best.get(entity_id)
        if current is None or score > current[0]:
            best[entity_id] = (
                score,
                str(hit.metadata.get("name") or ""),
                hit.content,
            )

    candidate_ids = list(best)
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(
                    GraphEntity.id, GraphEntity.canonical_name, GraphEntity.primary_type
                ).where(GraphEntity.id.in_(candidate_ids))
            )
        ).all()
    entities = {row.id: row for row in rows}
    seeds: list[WorkingNode] = []
    for entity_id, (score, fallback_name, description) in sorted(
        best.items(), key=lambda item: (-item[1][0], str(item[0]))
    ):
        row = entities.get(entity_id)
        if row is None:
            continue
        node_type = str(row.primary_type or "").upper()
        if (
            score < min_score
            or node_type in PROVENANCE_TYPES
            or node_type in REASONING_TYPES
            or node_type.startswith("MENTION_")
        ):
            continue
        seeds.append(
            WorkingNode(
                id=entity_id,
                name=str(row.canonical_name or fallback_name),
                node_type=node_type,
                description=description,
                seed_score=score,
                distance=0,
            )
        )
        if len(seeds) >= limit:
            break
    return seeds


async def resolve_frame_seeds(
    collection: Collection, question: str, limit: int, _min_score: float
) -> list[FrameSeed]:
    provider = await query_logic._resolve_embedding_provider(collection)
    embedding = await query_logic._embed_entity_query(provider, question)
    evidence = await query_logic._semantic_frame_evidence(
        question,
        collection,
        embedding,
        top_k=limit,
    )
    scored_ids = {item.frame_id: item.score for item in evidence}
    if not scored_ids:
        return []
    async with AsyncSessionLocal() as session:
        frames = (
            (
                await session.execute(
                    select(GraphSemanticFrame).where(
                        GraphSemanticFrame.id.in_(scored_ids)
                    )
                )
            )
            .scalars()
            .all()
        )
        arguments = (
            await session.execute(
                select(
                    GraphFrameArgument.frame_id,
                    GraphFrameArgument.role,
                    GraphFrameArgument.entity_id,
                    GraphFrameArgument.variable_name,
                    GraphFrameArgument.literal_value,
                )
                .where(
                    GraphFrameArgument.frame_id.in_(scored_ids),
                )
                .order_by(GraphFrameArgument.position)
            )
        ).all()
    argument_ids: dict[uuid.UUID, list[uuid.UUID]] = defaultdict(list)
    argument_patterns: dict[uuid.UUID, list[FrameArgumentPattern]] = defaultdict(list)
    for frame_id, role, entity_id, variable_name, literal_value in arguments:
        if entity_id is not None:
            argument_ids[frame_id].append(entity_id)
        argument_patterns[frame_id].append(
            FrameArgumentPattern(
                role=str(role),
                entity_id=entity_id,
                variable_name=str(variable_name) if variable_name else None,
                literal_value=literal_value,
            )
        )
    seeds = [
        FrameSeed(
            id=frame.id,
            kind=frame.frame_kind,
            title=frame.title,
            text=frame.frame_text,
            predicate=str(frame.predicate or ""),
            score=scored_ids[frame.id],
            proposition_id=frame.proposition_entity_id,
            argument_ids=tuple(argument_ids[frame.id]),
            executable_status=frame.executable_status,
            conditions=tuple(frame.conditions_json or ()),
            exceptions=tuple(frame.exceptions_json or ()),
            arguments=tuple(argument_patterns[frame.id]),
        )
        for frame in frames
    ]
    query_tokens = tokenize(question)

    def retrieval_rank(item: FrameSeed) -> tuple[float, str]:
        overlap = len(
            query_tokens & tokenize(f"{item.title} {item.predicate} {item.text}")
        )
        return (-(item.score + min(overlap, 2) * 0.08), str(item.id))

    propositions = sorted(
        (item for item in seeds if item.kind == "proposition"), key=retrieval_rank
    )[:limit]
    communities = sorted(
        (item for item in seeds if item.kind == "community_summary"),
        key=retrieval_rank,
    )[:2]
    return sorted([*propositions, *communities], key=retrieval_rank)


async def expand_frame_seeds(frame_seeds: list[FrameSeed]) -> list[WorkingNode]:
    """Map retrieval frames onto atoms without treating summaries as proof."""
    scores: dict[uuid.UUID, float] = {}
    distances: dict[uuid.UUID, int] = {}
    community_count = 0
    for frame in frame_seeds:
        if frame.kind == "community_summary":
            community_count += 1
            if community_count > 2:
                continue
            entity_ids = frame.argument_ids[:4]
            score = frame.score * 0.65
            distance = 0
        else:
            entity_ids = frame.argument_ids
            score = frame.score
            distance = 1
            if frame.proposition_id:
                entity_ids = (*entity_ids, frame.proposition_id)
        for entity_id in entity_ids:
            scores[entity_id] = max(scores.get(entity_id, 0.0), score)
            candidate_distance = 0 if entity_id == frame.proposition_id else distance
            distances[entity_id] = min(
                distances.get(entity_id, candidate_distance), candidate_distance
            )
    nodes = await _load_nodes(set(scores))
    for node_id, node in nodes.items():
        node.seed_score = scores[node_id]
        node.distance = distances[node_id]
    return list(nodes.values())


async def load_working_frames(graph: WorkingGraph) -> list[FrameSeed]:
    proposition_ids = {
        node_id
        for node_id, node in graph.nodes.items()
        if node.node_type == "PROPOSITION"
    }
    if not proposition_ids:
        return []
    async with AsyncSessionLocal() as session:
        frames = (
            (
                await session.execute(
                    select(GraphSemanticFrame).where(
                        GraphSemanticFrame.proposition_entity_id.in_(proposition_ids),
                        GraphSemanticFrame.frame_kind == "proposition",
                    )
                )
            )
            .scalars()
            .all()
        )
        rows = (
            await session.execute(
                select(
                    GraphFrameArgument.frame_id,
                    GraphFrameArgument.role,
                    GraphFrameArgument.entity_id,
                    GraphFrameArgument.variable_name,
                    GraphFrameArgument.literal_value,
                )
                .where(GraphFrameArgument.frame_id.in_([frame.id for frame in frames]))
                .order_by(GraphFrameArgument.frame_id, GraphFrameArgument.position)
            )
        ).all()
    arguments: dict[uuid.UUID, list[FrameArgumentPattern]] = defaultdict(list)
    for frame_id, role, entity_id, variable_name, literal_value in rows:
        arguments[frame_id].append(
            FrameArgumentPattern(
                role=str(role),
                entity_id=entity_id,
                variable_name=str(variable_name) if variable_name else None,
                literal_value=literal_value,
            )
        )
    return [
        FrameSeed(
            id=frame.id,
            kind=str(frame.frame_kind),
            title=str(frame.title),
            text=str(frame.frame_text),
            predicate=str(frame.predicate or ""),
            score=0.0,
            proposition_id=frame.proposition_entity_id,
            argument_ids=tuple(
                argument.entity_id
                for argument in arguments[frame.id]
                if argument.entity_id is not None
            ),
            executable_status=str(frame.executable_status),
            conditions=tuple(frame.conditions_json or ()),
            exceptions=tuple(frame.exceptions_json or ()),
            arguments=tuple(arguments[frame.id]),
        )
        for frame in frames
    ]


async def load_landmarks(
    version: GraphVersion, seed_ids: list[uuid.UUID], limit: int
) -> list[uuid.UUID]:
    if not seed_ids or limit <= 0:
        return []
    async with AsyncSessionLocal() as session:
        projection = (
            await session.execute(
                select(GraphProjectionSnapshot)
                .where(
                    GraphProjectionSnapshot.graph_version_id == version.id,
                    GraphProjectionSnapshot.name == "semantic_affinity_undirected",
                    GraphProjectionSnapshot.status == "completed",
                )
                .order_by(GraphProjectionSnapshot.completed_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if projection is None:
            return []
        community_ids = (
            (
                await session.execute(
                    select(GraphCommunityMembership.community_id)
                    .join(
                        GraphCommunity,
                        GraphCommunity.id == GraphCommunityMembership.community_id,
                    )
                    .where(
                        GraphCommunityMembership.entity_id.in_(seed_ids),
                        GraphCommunity.projection_id == projection.id,
                    )
                )
            )
            .scalars()
            .all()
        )
        if not community_ids:
            return []
        rows = (
            await session.execute(
                select(GraphNodeMetric.entity_id)
                .join(
                    GraphCommunityMembership,
                    GraphCommunityMembership.entity_id == GraphNodeMetric.entity_id,
                )
                .where(
                    GraphNodeMetric.projection_id == projection.id,
                    GraphNodeMetric.metric == "pagerank",
                    GraphCommunityMembership.community_id.in_(community_ids),
                    GraphNodeMetric.entity_id.not_in(seed_ids),
                )
                .order_by(GraphNodeMetric.value.desc())
                .limit(limit)
            )
        ).all()
    return list(dict.fromkeys(row.entity_id for row in rows))


async def _load_nodes(node_ids: set[uuid.UUID]) -> dict[uuid.UUID, WorkingNode]:
    if not node_ids:
        return {}
    async with AsyncSessionLocal() as session:
        entity_rows = (
            await session.execute(
                select(
                    GraphEntity.id,
                    GraphEntity.canonical_name,
                    GraphEntity.primary_type,
                ).where(GraphEntity.id.in_(node_ids))
            )
        ).all()
        description_rows = (
            await session.execute(
                select(EntityDescription.entity_id, EntityDescription.description)
                .where(EntityDescription.entity_id.in_(node_ids))
                .order_by(EntityDescription.created_at.desc())
            )
        ).all()
    descriptions: dict[uuid.UUID, str] = {}
    for entity_id, description in description_rows:
        descriptions.setdefault(entity_id, str(description or ""))
    return {
        row.id: WorkingNode(
            id=row.id,
            name=str(row.canonical_name),
            node_type=str(row.primary_type or "").upper(),
            description=descriptions.get(row.id, ""),
        )
        for row in entity_rows
    }


async def _load_incident_edges(
    collection_id: uuid.UUID,
    frontier: set[uuid.UUID],
    limit: int,
) -> list[WorkingEdge]:
    if not frontier or limit <= 0:
        return []
    structural_first = case(
        (GraphRelationship.rel_type.in_(STRUCTURAL_REL_TYPES), 0), else_=1
    )
    async with AsyncSessionLocal() as session:
        relationship_rows = (
            (
                await session.execute(
                    select(GraphRelationship)
                    .where(
                        GraphRelationship.collection_id == collection_id,
                        or_(
                            GraphRelationship.source_entity_id.in_(frontier),
                            GraphRelationship.target_entity_id.in_(frontier),
                        ),
                    )
                    .order_by(structural_first, GraphRelationship.weight.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        relationship_ids = [row.id for row in relationship_rows]
        description_rows = (
            await session.execute(
                select(
                    RelationshipDescription.relationship_id,
                    RelationshipDescription.description,
                )
                .where(RelationshipDescription.relationship_id.in_(relationship_ids))
                .order_by(RelationshipDescription.created_at.desc())
            )
        ).all()
    descriptions: dict[uuid.UUID, str] = {}
    for relationship_id, description in description_rows:
        descriptions.setdefault(relationship_id, str(description or ""))
    return [
        WorkingEdge(
            id=row.id,
            source_id=row.source_entity_id,
            target_id=row.target_entity_id,
            rel_type=str(row.rel_type).upper(),
            weight=int(row.weight or 1),
            description=descriptions.get(row.id, ""),
        )
        for row in relationship_rows
    ]


async def _load_metric_priors(
    version: GraphVersion, node_ids: set[uuid.UUID]
) -> dict[uuid.UUID, float]:
    if not node_ids:
        return {}
    async with AsyncSessionLocal() as session:
        projection = (
            await session.execute(
                select(GraphProjectionSnapshot.id)
                .where(
                    GraphProjectionSnapshot.graph_version_id == version.id,
                    GraphProjectionSnapshot.name == "semantic_entities_directed",
                    GraphProjectionSnapshot.status == "completed",
                )
                .limit(1)
            )
        ).scalar_one_or_none()
        if projection is None:
            return {}
        rows = (
            await session.execute(
                select(GraphNodeMetric.entity_id, GraphNodeMetric.value).where(
                    GraphNodeMetric.projection_id == projection,
                    GraphNodeMetric.metric == "pagerank",
                    GraphNodeMetric.entity_id.in_(node_ids),
                )
            )
        ).all()
    return {row.entity_id: float(row.value) for row in rows}


async def build_working_graph(
    collection: Collection,
    version: GraphVersion,
    seeds: list[WorkingNode],
    limits: Limits,
) -> WorkingGraph:
    graph = WorkingGraph(nodes={node.id: node for node in seeds})
    frontier = {node_id for node_id, node in graph.nodes.items() if node.distance == 0}
    landmark_ids = await load_landmarks(version, list(frontier), limits.max_landmarks)
    landmarks = await _load_nodes(set(landmark_ids))
    for node in landmarks.values():
        node.distance = 1
        graph.nodes.setdefault(node.id, node)
    frontier.update(landmarks)

    for hop in range(limits.max_hops):
        remaining_edges = limits.max_edges - len(graph.edges)
        if remaining_edges <= 0 or not frontier:
            break
        edges = await _load_incident_edges(collection.id, frontier, remaining_edges)
        new_node_ids: set[uuid.UUID] = set()
        for edge in edges:
            graph.edges.setdefault(edge.id, edge)
            for node_id in (edge.source_id, edge.target_id):
                if node_id not in graph.nodes:
                    new_node_ids.add(node_id)
        remaining_nodes = limits.max_nodes - len(graph.nodes)
        if remaining_nodes <= 0:
            break
        candidates = await _load_nodes(new_node_ids)
        ordered_candidates = sorted(
            candidates.values(),
            key=lambda node: (
                0
                if node.node_type in REASONING_TYPES
                else 1
                if node.node_type == "EVIDENCE_CHUNK"
                else 2,
                str(node.id),
            ),
        )
        loaded = {node.id: node for node in ordered_candidates[:remaining_nodes]}
        for node in loaded.values():
            node.distance = hop + 1
            graph.nodes[node.id] = node
        graph.edges = {
            edge_id: edge
            for edge_id, edge in graph.edges.items()
            if edge.source_id in graph.nodes and edge.target_id in graph.nodes
        }
        frontier = set(loaded)

    priors = await _load_metric_priors(version, set(graph.nodes))
    for node_id, value in priors.items():
        graph.nodes[node_id].pagerank = value
    return graph


def _active_text(node: WorkingNode, query_tokens: set[str]) -> bool:
    node_tokens = tokenize(f"{node.name} {node.description}")
    return bool(query_tokens & node_tokens)


def _frame_arguments(frame: FrameSeed) -> tuple[FrameArgumentPattern, ...]:
    if frame.arguments:
        return frame.arguments
    roles = ("subject", "object")
    return tuple(
        FrameArgumentPattern(
            role=roles[index] if index < len(roles) else f"argument_{index}",
            entity_id=entity_id,
        )
        for index, entity_id in enumerate(frame.argument_ids)
    )


def _term(argument: FrameArgumentPattern) -> tuple[str, str] | None:
    if argument.entity_id is not None:
        return "entity", str(argument.entity_id)
    if argument.variable_name:
        return "variable", argument.variable_name.lstrip("?")
    if argument.literal_value is not None:
        return "literal", json.dumps(argument.literal_value, sort_keys=True)
    return None


def _argument_payload(argument: FrameArgumentPattern) -> dict[str, Any]:
    return {
        "role": argument.role,
        "entity_id": str(argument.entity_id) if argument.entity_id else None,
        "variable_name": argument.variable_name,
        "literal_value": argument.literal_value,
    }


def unify_goal(
    goal: GoalPattern,
    frame: FrameSeed,
    bindings: dict[str, str] | None = None,
) -> dict[str, str] | None:
    if normalize_predicate(goal.predicate) != normalize_predicate(frame.predicate):
        return None
    result = dict(bindings or {})
    candidate_by_role = {
        argument.role: argument for argument in _frame_arguments(frame)
    }
    for expected in goal.arguments:
        candidate = candidate_by_role.get(expected.role)
        if candidate is None:
            return None
        expected_term = _term(expected)
        candidate_term = _term(candidate)
        if expected_term is None or candidate_term is None:
            return None
        if expected_term[0] == "variable":
            current = result.get(expected_term[1])
            if current is not None and current != candidate_term[1]:
                return None
            result[expected_term[1]] = candidate_term[1]
        elif candidate_term[0] == "variable":
            current = result.get(candidate_term[1])
            if current is not None and current != expected_term[1]:
                return None
            result[candidate_term[1]] = expected_term[1]
        elif expected_term != candidate_term:
            return None
    return result


def normalize_predicate(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def compile_goal_patterns(
    plan: QueryPlan,
    graph: WorkingGraph,
    frames: list[FrameSeed],
    limits: Limits,
) -> list[GoalPattern]:
    query_tokens = {_goal_token(token) for token in plan.state_tokens}
    query_entity_ids = {
        node_id
        for node_id, node in graph.nodes.items()
        if query_tokens & goal_tokens(node.name)
        and node.node_type not in PROVENANCE_TYPES | REASONING_TYPES
    }
    ranked: list[tuple[int, float, FrameSeed, GoalPattern]] = []
    for frame in frames:
        if frame.kind != "proposition" or not frame.predicate:
            continue
        overlap = len(query_tokens & goal_tokens(f"{frame.title} {frame.predicate}"))
        arguments = tuple(
            argument
            if argument.entity_id in query_entity_ids or argument.variable_name
            else FrameArgumentPattern(
                role=argument.role,
                variable_name=f"goal_{argument.role}",
            )
            for argument in _frame_arguments(frame)
        )
        constant_count = sum(argument.entity_id is not None for argument in arguments)
        if overlap == 0 or (constant_count == 0 and overlap < 2):
            continue
        ranked.append(
            (
                overlap,
                frame.score,
                frame,
                GoalPattern(frame.predicate, arguments),
            )
        )
    ranked.sort(key=lambda item: (-item[0], -item[1], str(item[2].id)))
    goals: list[GoalPattern] = []
    seen: set[tuple[str, tuple[tuple[str, str] | None, ...]]] = set()
    for _, _, _, goal in ranked:
        key = (
            normalize_predicate(goal.predicate),
            tuple(_term(arg) for arg in goal.arguments),
        )
        if key in seen:
            continue
        seen.add(key)
        goals.append(goal)
        if len(goals) >= limits.max_goal_matches:
            break
    return goals


def backward_chain(
    plan: QueryPlan,
    graph: WorkingGraph,
    frames: list[FrameSeed],
    limits: Limits,
    active_nodes: set[uuid.UUID],
) -> dict[str, Any]:
    outgoing: dict[uuid.UUID, list[WorkingEdge]] = defaultdict(list)
    incoming: dict[uuid.UUID, list[WorkingEdge]] = defaultdict(list)
    for edge in graph.edges.values():
        outgoing[edge.source_id].append(edge)
        incoming[edge.target_id].append(edge)
    frame_by_proposition = {
        frame.proposition_id: frame
        for frame in frames
        if frame.proposition_id is not None
    }
    goals = compile_goal_patterns(plan, graph, frames, limits)
    explored = 0

    def prove(
        proposition_id: uuid.UUID,
        bindings: dict[str, str],
        depth: int,
        path: set[uuid.UUID],
    ) -> dict[str, Any]:
        nonlocal explored
        explored += 1
        if explored > limits.max_goal_states:
            return {"status": "state_limit", "proposition_id": str(proposition_id)}
        if depth > limits.max_goal_depth:
            return {"status": "depth_limit", "proposition_id": str(proposition_id)}
        if proposition_id in path:
            return {"status": "cycle", "proposition_id": str(proposition_id)}
        concluding_rules = [
            edge.source_id
            for edge in incoming[proposition_id]
            if edge.rel_type == "CONCLUDES"
        ]
        frame = frame_by_proposition.get(proposition_id)
        unbound_variables = sorted(
            {
                argument.variable_name.lstrip("?")
                for argument in _frame_arguments(frame)
                if argument.variable_name
                and argument.variable_name.lstrip("?") not in bindings
            }
            if frame
            else set()
        )
        if not concluding_rules:
            return {
                "status": "conditional" if unbound_variables else "asserted",
                "proposition_id": str(proposition_id),
                "bindings": bindings,
                "unbound_variables": unbound_variables,
            }
        alternatives: list[dict[str, Any]] = []
        for rule_id in concluding_rules[: limits.max_goal_matches]:
            blockers = [
                edge.source_id
                for edge in incoming[rule_id]
                if edge.rel_type == "BLOCKS"
            ]
            active_blockers = [value for value in blockers if value in active_nodes]
            antecedents = [
                edge.source_id
                for edge in incoming[rule_id]
                if edge.rel_type == "ANTECEDENT_OF"
            ]
            obligations: list[dict[str, Any]] = []
            for antecedent_id in antecedents:
                node = graph.nodes.get(antecedent_id)
                if antecedent_id in active_nodes:
                    obligations.append(
                        {
                            "status": "satisfied",
                            "node_id": str(antecedent_id),
                            "statement": node.description if node else "",
                        }
                    )
                elif node and node.node_type == "PROPOSITION":
                    obligations.append(
                        prove(
                            antecedent_id,
                            bindings,
                            depth + 1,
                            path | {proposition_id},
                        )
                    )
                else:
                    obligations.append(
                        {
                            "status": "unresolved_subgoal",
                            "node_id": str(antecedent_id),
                            "statement": (
                                node.description or node.name if node else ""
                            ),
                        }
                    )
            if active_blockers:
                status = "blocked"
            elif obligations and all(
                item["status"] in {"asserted", "proved", "satisfied"}
                for item in obligations
            ):
                status = "proved"
            elif not antecedents:
                status = "incomplete_rule"
            else:
                status = "conditional"
            alternatives.append(
                {
                    "rule_id": str(rule_id),
                    "status": status,
                    "active_blockers": [str(value) for value in active_blockers],
                    "obligations": obligations,
                }
            )
        status = (
            "proved"
            if any(item["status"] == "proved" for item in alternatives)
            else "blocked"
            if alternatives
            and all(item["status"] == "blocked" for item in alternatives)
            else "conditional"
        )
        return {
            "status": status,
            "proposition_id": str(proposition_id),
            "bindings": bindings,
            "alternatives": alternatives,
        }

    proofs: list[dict[str, Any]] = []
    for goal in goals:
        matches: list[dict[str, Any]] = []
        for proposition_id, frame in frame_by_proposition.items():
            bindings = unify_goal(goal, frame)
            if bindings is None:
                continue
            matches.append(prove(proposition_id, bindings, 0, set()))
            if len(matches) >= limits.max_goal_matches:
                break
        proofs.append(
            {
                "goal": {
                    "predicate": goal.predicate,
                    "arguments": [
                        _argument_payload(argument) for argument in goal.arguments
                    ],
                },
                "status": (
                    "proved"
                    if any(item["status"] in {"asserted", "proved"} for item in matches)
                    else "unresolved"
                ),
                "matches": matches,
            }
        )
    goal_predicates = {normalize_predicate(goal.predicate) for goal in goals}
    goal_coverage_tokens: set[str] = set()
    for frame in frames:
        if normalize_predicate(frame.predicate) in goal_predicates:
            goal_coverage_tokens.update(
                goal_tokens(f"{frame.title} {frame.predicate} {frame.text}")
            )
    unmatched_query_terms = sorted(
        token
        for token in plan.state_tokens
        if _goal_token(token) not in goal_coverage_tokens
    )
    return {
        "status": (
            "insufficient_goal_coverage"
            if unmatched_query_terms
            else "proved"
            if any(item["status"] == "proved" for item in proofs)
            else "unresolved"
        ),
        "unmatched_query_terms": unmatched_query_terms,
        "unresolved_query_goals": [
            {
                "terms": unmatched_query_terms,
                "status": "no_unifiable_proposition_pattern",
            }
        ]
        if unmatched_query_terms
        else [],
        "goals": proofs,
        "explored_states": explored,
        "state_limit": limits.max_goal_states,
        "depth_limit": limits.max_goal_depth,
    }


def execute_operator(
    plan: QueryPlan,
    conclusions: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> dict[str, Any]:
    usable = [
        item for item in conclusions if item["status"] not in {"blocked", "incomplete"}
    ]
    if plan.operator == "evaluate":
        return {
            "supporting": [
                item["proposition_id"]
                for item in usable
                if "polarity=positive" in item["statement"].casefold()
            ],
            "harm_or_limitation": [
                item["proposition_id"]
                for item in usable
                if "polarity=negative" in item["statement"].casefold()
            ],
            "conflicts": conflicts,
        }
    if plan.operator == "choose":
        return {
            "applicable": [
                {
                    "proposition_id": item["proposition_id"],
                    "subjects": item["subjects"],
                    "objects": item["objects"],
                    "conditions": item["conditions"],
                    "exceptions": item["exceptions"],
                }
                for item in usable
                if item["subjects"] or item["objects"]
            ],
            "blocked": [
                item["proposition_id"]
                for item in conclusions
                if item["status"] == "blocked"
            ],
        }
    if plan.operator == "redesign":
        return {
            "constraints": [
                item["proposition_id"]
                for item in usable
                if item["conditions"]
                or "polarity=negative" in item["statement"].casefold()
            ],
            "rewrite_candidates": [],
            "note": "No graph rewrite is emitted without encoded rewrite operators.",
        }
    if plan.operator == "explain":
        return {
            "mechanism_chain_candidates": [item["proposition_id"] for item in usable],
            "unresolved_conflicts": conflicts,
        }
    if plan.operator == "procedure":
        return {
            "procedure_evidence": [
                {
                    "proposition_id": item["proposition_id"],
                    "statement": item["statement"],
                    "conditions": item["conditions"],
                    "exceptions": item["exceptions"],
                }
                for item in usable
            ],
            "requested_outputs": list(plan.desired_outputs),
        }
    return {
        "support": [
            item["proposition_id"]
            for item in usable
            if "polarity=negative" not in item["statement"].casefold()
        ],
        "attack": [
            item["proposition_id"]
            for item in usable
            if "polarity=negative" in item["statement"].casefold()
        ],
        "conflicts": conflicts,
    }


def reason(
    plan: QueryPlan,
    graph: WorkingGraph,
    limits: Limits,
    frame_seeds: list[FrameSeed] | None = None,
) -> dict[str, Any]:
    outgoing: dict[uuid.UUID, list[WorkingEdge]] = defaultdict(list)
    incoming: dict[uuid.UUID, list[WorkingEdge]] = defaultdict(list)
    for edge in graph.edges.values():
        outgoing[edge.source_id].append(edge)
        incoming[edge.target_id].append(edge)

    query_tokens = set(plan.state_tokens)
    frames_by_proposition = {
        frame.proposition_id: frame
        for frame in frame_seeds or []
        if frame.proposition_id is not None
    }
    propositions = {
        node_id
        for node_id, node in graph.nodes.items()
        if node.node_type == "PROPOSITION"
    }
    rules = {
        node_id for node_id, node in graph.nodes.items() if node.node_type == "RULE"
    }
    active_nodes: set[uuid.UUID] = set()
    for node_id, node in graph.nodes.items():
        text_matches_state = _active_text(node, query_tokens)
        if node.node_type in {"CONDITION", "EXCEPTION"}:
            if text_matches_state:
                active_nodes.add(node_id)
        elif node.seed_score > 0 or text_matches_state:
            active_nodes.add(node_id)
    rule_conclusions = {
        edge.target_id for edge in graph.edges.values() if edge.rel_type == "CONCLUDES"
    }
    active_nodes.update(
        frame.proposition_id
        for frame in frame_seeds or []
        if frame.proposition_id is not None
        and frame.proposition_id not in rule_conclusions
    )
    backward = backward_chain(
        plan,
        graph,
        list(frame_seeds or []),
        limits,
        active_nodes,
    )
    backward_match_ids = {
        uuid.UUID(match["proposition_id"])
        for goal in backward["goals"]
        for match in goal["matches"]
        if match.get("proposition_id")
    }
    activated_propositions: set[uuid.UUID] = set()

    # Monotonic forward chaining reaches a fixed point in at most |rules| rounds.
    for _ in range(len(rules) + 1):
        changed = False
        for rule_id in sorted(rules, key=str):
            antecedents = [
                edge.source_id
                for edge in incoming[rule_id]
                if edge.rel_type == "ANTECEDENT_OF"
            ]
            blockers = [
                edge.source_id
                for edge in incoming[rule_id]
                if edge.rel_type == "BLOCKS"
            ]
            conclusions = [
                edge.target_id
                for edge in outgoing[rule_id]
                if edge.rel_type == "CONCLUDES"
            ]
            if any(item in active_nodes for item in blockers):
                continue
            if antecedents and all(item in active_nodes for item in antecedents):
                new_conclusions = set(conclusions) - active_nodes
                if new_conclusions:
                    active_nodes.update(new_conclusions)
                    activated_propositions.update(new_conclusions)
                    changed = True
        if not changed:
            break

    rule_traces: list[dict[str, Any]] = []
    for rule_id in sorted(rules, key=str):
        antecedents = [
            edge.source_id
            for edge in incoming[rule_id]
            if edge.rel_type == "ANTECEDENT_OF"
        ]
        blockers = [
            edge.source_id for edge in incoming[rule_id] if edge.rel_type == "BLOCKS"
        ]
        conclusions = [
            edge.target_id for edge in outgoing[rule_id] if edge.rel_type == "CONCLUDES"
        ]
        active_antecedents = [item for item in antecedents if item in active_nodes]
        active_blockers = [item for item in blockers if item in active_nodes]
        if not antecedents or not conclusions:
            status = "incomplete"
        elif active_blockers:
            status = "blocked"
        elif antecedents and len(active_antecedents) == len(antecedents):
            status = "fired"
            activated_propositions.update(conclusions)
        else:
            status = "conditional"
        rule_traces.append(
            {
                "rule_id": str(rule_id),
                "status": status,
                "antecedents": [str(item) for item in antecedents],
                "active_antecedents": [str(item) for item in active_antecedents],
                "blockers": [str(item) for item in blockers],
                "active_blockers": [str(item) for item in active_blockers],
                "conclusions": [str(item) for item in conclusions],
            }
        )

    conclusion_rule_statuses: dict[uuid.UUID, set[str]] = defaultdict(set)
    for trace in rule_traces:
        for conclusion_id in trace["conclusions"]:
            conclusion_rule_statuses[uuid.UUID(conclusion_id)].add(trace["status"])
    backward_rule_ids = {
        alternative["rule_id"]
        for goal in backward["goals"]
        for match in goal["matches"]
        for alternative in match.get("alternatives", [])
    }
    rule_traces = [
        trace
        for trace in rule_traces
        if trace["rule_id"] in backward_rule_ids
        or trace["status"] in {"fired", "blocked"}
    ][:64]

    conclusions: list[dict[str, Any]] = []
    for proposition_id in propositions:
        node = graph.nodes[proposition_id]
        frame = frames_by_proposition.get(proposition_id)
        subjects = [
            edge.target_id
            for edge in outgoing[proposition_id]
            if edge.rel_type == "SUBJECT"
        ]
        objects = [
            edge.target_id
            for edge in outgoing[proposition_id]
            if edge.rel_type == "OBJECT"
        ]
        conditions = [
            edge.target_id
            for edge in outgoing[proposition_id]
            if edge.rel_type == "CONDITION"
        ]
        exceptions = [
            edge.target_id
            for edge in outgoing[proposition_id]
            if edge.rel_type == "EXCEPTION"
        ]
        support_count = sum(
            edge.rel_type == "SUPPORTS" for edge in incoming[proposition_id]
        )
        related_ids = [*subjects, *objects]
        relevance = max(
            (
                graph.nodes[item].seed_score
                for item in related_ids
                if item in graph.nodes
            ),
            default=node.seed_score,
        )
        proximity = 1.0 / (1.0 + min(node.distance, 10))
        structural_prior = max(
            (graph.nodes[item].pagerank for item in related_ids if item in graph.nodes),
            default=0.0,
        )
        blocked = any(item in active_nodes for item in exceptions)
        rule_statuses = conclusion_rule_statuses.get(proposition_id, set())
        status = "blocked" if blocked or rule_statuses == {"blocked"} else "asserted"
        if "fired" in rule_statuses or proposition_id in activated_propositions:
            status = "derived"
        elif "conditional" in rule_statuses:
            status = "conditional"
        elif "incomplete" in rule_statuses:
            status = "incomplete"
        score = relevance + (0.15 * proximity) + min(support_count, 3) * 0.03
        if frame:
            score += 0.2
        if proposition_id in backward_match_ids:
            score += 0.25
        score += min(structural_prior, 0.1)
        if status == "blocked":
            score *= 0.25
        item = {
            "proposition_id": str(proposition_id),
            "status": status,
            "score": round(score, 6),
            "statement": frame.text if frame else node.description,
            "subjects": [
                graph.nodes[value].name for value in subjects if value in graph.nodes
            ],
            "objects": [
                graph.nodes[value].name for value in objects if value in graph.nodes
            ],
            "conditions": list(frame.conditions)
            if frame and frame.conditions
            else [
                graph.nodes[value].description
                for value in conditions
                if value in graph.nodes
            ],
            "exceptions": list(frame.exceptions)
            if frame and frame.exceptions
            else [
                graph.nodes[value].description
                for value in exceptions
                if value in graph.nodes
            ],
            "support_count": support_count,
        }
        conclusions.append(item)

    conclusions.sort(key=lambda item: (-item["score"], item["proposition_id"]))
    conflicts = [
        {
            "source": str(edge.source_id),
            "target": str(edge.target_id),
            "type": edge.rel_type,
        }
        for edge in graph.edges.values()
        if edge.rel_type in {"ATTACKS", "CONTRADICTS"}
    ]
    selected_conclusions = conclusions[: limits.max_conclusions]
    return {
        "operator": plan.operator,
        "active_node_count": len(active_nodes),
        "rules": rule_traces,
        "conclusions": selected_conclusions,
        "conflicts": conflicts,
        "operator_result": execute_operator(plan, selected_conclusions, conflicts),
        "backward_reasoning": backward,
        "fixed_point": True,
    }


def trace_payload(
    plan: QueryPlan,
    version: GraphVersion,
    seeds: list[WorkingNode],
    frame_seeds: list[FrameSeed],
    graph: WorkingGraph,
    limits: Limits,
    reasoning: dict[str, Any],
    covered_tokens: tuple[str, ...],
) -> dict[str, Any]:
    required_coverage = (
        len(plan.state_tokens)
        if len(plan.state_tokens) <= 3
        else max(2, math.ceil(len(plan.state_tokens) * 0.6))
    )
    has_coverage = len(covered_tokens) >= required_coverage
    return {
        "graph_version": version.version,
        "plan": asdict(plan),
        "limits": asdict(limits),
        "seeds": [
            {
                "id": str(node.id),
                "name": node.name,
                "type": node.node_type,
                "score": round(node.seed_score, 6),
            }
            for node in seeds
        ],
        "frame_seeds": [
            {
                "id": str(frame.id),
                "kind": frame.kind,
                "title": frame.title,
                "predicate": frame.predicate,
                "score": round(frame.score, 6),
                "proposition_id": (
                    str(frame.proposition_id) if frame.proposition_id else None
                ),
                "executable_status": frame.executable_status,
            }
            for frame in frame_seeds
        ],
        "working_graph": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "max_distance": max(
                (node.distance for node in graph.nodes.values()), default=0
            ),
        },
        "reasoning": reasoning,
        "covered_query_tokens": covered_tokens,
        "sufficiency": (
            "sufficient_candidates"
            if has_coverage and reasoning["conclusions"]
            else "insufficient_graph_evidence"
        ),
        "retrieval_note": (
            "Seed vector search has a bounded result count, but the current "
            "4096-dimensional pgvector tables have no ANN index; seed lookup can "
            "still scan at collection scale."
        ),
    }


def _production_artifact_payload(
    collection: Collection,
    artifacts: query_logic.GraphQueryArtifacts,
) -> dict[str, Any]:
    return {
        "collection_id": str(collection.id),
        "collection_name": collection.name,
        "route": asdict(artifacts.route_profile),
        "entities_used": artifacts.entities_used,
        "relationships_used": artifacts.relationships_used,
        "state": {
            "discovered_entity_count": len(artifacts.state.discovered_entity_ids),
            "traversed_relationship_count": len(artifacts.state.traversed_rel_ids),
            "top_entities": sorted(
                artifacts.state.entity_relevance.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:20],
            "top_relationships": artifacts.state.traversed_rel_ids[:30],
        },
        "reasoning": artifacts.reasoning_trace,
    }


async def _grounded_query_plan(
    collection: Collection,
    question: str,
    document_ids: list[uuid.UUID] | None,
) -> tuple[query_logic.GraphQueryPlan, dict[str, Any]]:
    started = time.perf_counter()
    llm = await query_logic._resolve_llm_provider(
        namespace_id=collection.namespace_id,
        llm_profile_id=collection.llm_profile_id,
    )
    if isinstance(llm, query_logic.LocalEchoLLMProvider):
        return query_logic._fallback_graph_query_plan(question), {
            "status": "fallback_no_llm"
        }
    output_values = [
        "definition",
        "steps",
        "timing",
        "conditions",
        "exceptions",
        "contraindications",
        "causes",
        "mechanisms",
        "comparison",
        "evidence",
        "recommendations",
        "constraints",
    ]
    intent_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning_operator": {
                "type": "string",
                "enum": [
                    "describe",
                    "procedure",
                    "explain",
                    "choose",
                    "evaluate",
                    "redesign",
                ],
            },
            "requested_outputs": {
                "type": "array",
                "items": {"type": "string", "enum": output_values},
            },
        },
        "required": ["reasoning_operator", "requested_outputs"],
    }
    intent_started = time.perf_counter()
    intent = await llm.structured_extract(
        (
            "Classify only the reasoning operation and answer obligations. Do not "
            "identify entities, aliases, topics, mechanisms, or domain concepts. "
            "For practice or health-effect questions, include conditions and "
            "contraindications when safety materially affects the answer.\n\n"
            f"Question: {question}"
        ),
        intent_schema,
    )
    intent_elapsed = time.perf_counter() - intent_started
    operator = str(intent.get("reasoning_operator") or "describe").strip().lower()
    if operator not in {
        "describe",
        "procedure",
        "explain",
        "choose",
        "evaluate",
        "redesign",
    }:
        operator = "describe"
    requested_outputs = list(
        dict.fromkeys(
            value
            for value in query_logic._normalise_plan_list(
                intent.get("requested_outputs"), max_items=12
            )
            if value in output_values
        )
    )
    fallback_outputs = [
        value
        for value in compile_query(question).desired_outputs
        if value in output_values
    ]
    requested_outputs = list(dict.fromkeys([*requested_outputs, *fallback_outputs]))
    retrieval_question = question
    if requested_outputs:
        retrieval_question += "\nRequired graph evidence: " + ", ".join(
            requested_outputs
        )
    retrieval_started = time.perf_counter()
    embedding_provider = await query_logic._resolve_embedding_provider(collection)
    entity_embedding = await query_logic._embed_entity_query(
        embedding_provider, retrieval_question
    )
    relationship_embedding = await query_logic._embed_relationship_query(
        embedding_provider, retrieval_question, rel_type=None
    )
    mention_index = await query_logic._get_mention_index(collection)
    candidate_tuples = await query_logic._top_entity_candidates(
        collection,
        entity_embedding,
        question=question,
        top_k=40,
        document_ids=document_ids,
        mention_index=mention_index,
    )
    candidate_names = [name for name, _, _ in candidate_tuples]
    async with AsyncSessionLocal() as session:
        entity_rows = (
            await session.execute(
                select(
                    GraphEntity.id,
                    GraphEntity.canonical_name,
                    GraphEntity.primary_type,
                ).where(
                    GraphEntity.collection_id == collection.id,
                    GraphEntity.canonical_name.in_(candidate_names),
                )
            )
        ).all()
    entity_by_name = {
        str(name): {
            "id": str(entity_id),
            "name": str(name),
            "type": str(primary_type or ""),
        }
        for entity_id, name, primary_type in entity_rows
    }
    entity_candidates = [
        {
            **entity_by_name[name],
            "description": description[:300],
            "score": round(score, 6),
        }
        for name, description, score in candidate_tuples
        if name in entity_by_name
    ][:30]
    frames = await query_logic._semantic_frame_evidence(
        retrieval_question,
        collection,
        entity_embedding,
        document_ids=document_ids,
        top_k=30,
    )
    frame_candidates = [
        {
            "id": str(frame.frame_id),
            "predicate": frame.predicate,
            "title": frame.title,
            "statement": frame.frame_text[:500],
            "arguments": [
                {"role": role, "entity_id": str(entity_id), "name": name}
                for role, entity_id, name in frame.arguments
            ],
            "conditions": list(frame.conditions),
            "exceptions": list(frame.exceptions),
            "score": round(frame.score, 6),
        }
        for frame in frames
        if frame.frame_kind == "proposition"
    ]
    relationship_seeds = await query_logic._search_relationship_seeds(
        collection,
        relationship_embedding,
        top_k=40,
        min_similarity=0.0,
        document_ids=document_ids,
    )
    relationship_ids = [uuid.UUID(rel_id) for rel_id, _ in relationship_seeds]
    relationship_score = dict(relationship_seeds)
    relationship_candidates: list[dict[str, Any]] = []
    if relationship_ids:
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(
                        GraphRelationship.id,
                        GraphRelationship.rel_type,
                        GraphRelationship.source_entity_id,
                        GraphRelationship.target_entity_id,
                        GraphRelationship.keywords,
                        GraphEntity.canonical_name,
                    )
                    .join(
                        GraphEntity,
                        GraphEntity.id == GraphRelationship.source_entity_id,
                    )
                    .where(GraphRelationship.id.in_(relationship_ids))
                )
            ).all()
            target_ids = {row.target_entity_id for row in rows}
            target_rows = (
                await session.execute(
                    select(GraphEntity.id, GraphEntity.canonical_name).where(
                        GraphEntity.id.in_(target_ids)
                    )
                )
            ).all()
        target_names = {entity_id: str(name) for entity_id, name in target_rows}
        relationship_candidates = [
            {
                "id": str(row.id),
                "predicate": str(row.rel_type),
                "source_id": str(row.source_entity_id),
                "source": str(row.canonical_name),
                "target_id": str(row.target_entity_id),
                "target": target_names.get(row.target_entity_id, ""),
                "keywords": list(row.keywords or []),
                "score": round(relationship_score.get(str(row.id), 0.0), 6),
            }
            for row in rows
        ]
        relationship_candidates.sort(key=lambda item: item["score"], reverse=True)
    retrieval_elapsed = time.perf_counter() - retrieval_started

    def retain_diverse(
        candidates: list[dict[str, Any]],
        *,
        text_fields: tuple[str, ...],
        limit: int,
    ) -> list[str]:
        """Keep strong branch roots without collapsing onto one semantic cluster."""
        remaining = list(candidates)
        retained: list[dict[str, Any]] = []
        retained_tokens: list[set[str]] = []
        while remaining and len(retained) < limit:
            best_index = 0
            best_rank = float("-inf")
            for index, candidate in enumerate(remaining):
                candidate_tokens = tokenize(
                    " ".join(str(candidate.get(field) or "") for field in text_fields)
                )
                max_similarity = max(
                    (
                        len(candidate_tokens & existing)
                        / max(len(candidate_tokens | existing), 1)
                        for existing in retained_tokens
                    ),
                    default=0.0,
                )
                rank = float(candidate.get("score") or 0.0) + (
                    0.12 * (1.0 - max_similarity)
                )
                if rank > best_rank:
                    best_index = index
                    best_rank = rank
            chosen = remaining.pop(best_index)
            retained.append(chosen)
            retained_tokens.append(
                tokenize(
                    " ".join(str(chosen.get(field) or "") for field in text_fields)
                )
            )
        return [str(candidate["id"]) for candidate in retained]

    retained_entity_ids = retain_diverse(
        entity_candidates,
        text_fields=("name", "type", "description"),
        limit=4,
    )
    retained_frame_ids = retain_diverse(
        frame_candidates,
        text_fields=("predicate", "title", "statement"),
        limit=6,
    )
    retained_relationship_ids = retain_diverse(
        relationship_candidates,
        text_fields=("predicate", "source", "target", "keywords"),
        limit=6,
    )

    entity_ids = [item["id"] for item in entity_candidates]
    frame_ids = [item["id"] for item in frame_candidates]
    rel_ids = [item["id"] for item in relationship_candidates]
    predicates = sorted(
        {
            item["predicate"]
            for item in [*frame_candidates, *relationship_candidates]
            if item["predicate"]
        }
    )

    def constrained_items(values: list[str]) -> dict[str, Any]:
        return {"type": "string", "enum": values or [""]}

    grounded_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_entity_ids": {
                "type": "array",
                "items": constrained_items(entity_ids),
                "maxItems": 8,
            },
            "excluded_entity_ids": {
                "type": "array",
                "items": constrained_items(entity_ids),
                "maxItems": 6,
            },
            "selected_frame_ids": {
                "type": "array",
                "items": constrained_items(frame_ids),
                "maxItems": 8,
            },
            "selected_relationship_ids": {
                "type": "array",
                "items": constrained_items(rel_ids),
                "maxItems": 8,
            },
            "goal_predicates": {
                "type": "array",
                "items": constrained_items(predicates),
                "maxItems": 8,
            },
            "blocked_predicates": {
                "type": "array",
                "items": constrained_items(predicates),
                "maxItems": 6,
            },
            "preferred_path_properties": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["causal", "temporal", "transitive", "symmetric"],
                },
                "maxItems": 4,
            },
            "max_reasoning_hops": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },
        },
        "required": [
            "selected_entity_ids",
            "excluded_entity_ids",
            "selected_frame_ids",
            "selected_relationship_ids",
            "goal_predicates",
            "blocked_predicates",
            "preferred_path_properties",
            "max_reasoning_hops",
        ],
    }
    candidate_pack = {
        "entities": entity_candidates,
        "proposition_frames": frame_candidates,
        "relationships": relationship_candidates[:30],
    }
    selection_started = time.perf_counter()
    grounded = await llm.structured_extract(
        (
            "Ground a reasoning plan in the supplied graph candidates. Select only "
            "identifiers and predicates present below; the schema enforces this. "
            "Resolve the user's ordinary wording to the closest encoded graph "
            "meaning. Prefer complete propositions that directly satisfy each "
            "requested output. Include negative, risk, exception, blocker, and "
            "safety evidence when contraindications or constraints are requested. "
            "Choose the smallest sufficient set; never fill an array merely because "
            "candidates are available. "
            "Exclude lexical lookalikes with a different meaning. Select no item "
            "when the graph does not support the interpretation.\n\n"
            f"Question: {question}\n"
            f"Intent: {operator}\n"
            f"Required outputs: {requested_outputs}\n\n"
            f"Graph candidates:\n{json.dumps(candidate_pack, ensure_ascii=True)}"
        ),
        grounded_schema,
    )
    selection_elapsed = time.perf_counter() - selection_started

    def allowed_list(key: str, allowed: set[str], limit: int) -> list[str]:
        return [
            value
            for value in query_logic._normalise_plan_list(
                grounded.get(key), max_items=limit
            )
            if value in allowed
        ]

    llm_entity_ids = allowed_list("selected_entity_ids", set(entity_ids), 8)
    excluded_entity_ids = allowed_list("excluded_entity_ids", set(entity_ids), 6)
    llm_frame_ids = allowed_list("selected_frame_ids", set(frame_ids), 8)
    llm_relationship_ids = allowed_list("selected_relationship_ids", set(rel_ids), 8)
    selected_entity_ids = list(dict.fromkeys([*llm_entity_ids, *retained_entity_ids]))[
        :12
    ]
    selected_frame_ids = list(dict.fromkeys([*llm_frame_ids, *retained_frame_ids]))[:12]
    selected_relationship_ids = list(
        dict.fromkeys([*llm_relationship_ids, *retained_relationship_ids])
    )[:12]
    goal_predicates = allowed_list("goal_predicates", set(predicates), 8)
    blocked_predicates = allowed_list("blocked_predicates", set(predicates), 6)
    selected_names = [
        item["name"] for item in entity_candidates if item["id"] in llm_entity_ids
    ]
    excluded_names = [
        item["name"] for item in entity_candidates if item["id"] in excluded_entity_ids
    ]
    aliases: list[str] = []
    if selected_entity_ids:
        async with AsyncSessionLocal() as session:
            alias_rows = (
                await session.execute(
                    select(EntityAlias.alias_name).where(
                        EntityAlias.collection_id == collection.id,
                        EntityAlias.entity_id.in_(
                            [uuid.UUID(value) for value in selected_entity_ids]
                        ),
                    )
                )
            ).scalars()
        aliases = list(dict.fromkeys(str(value) for value in alias_rows))[:20]
    properties = allowed_list(
        "preferred_path_properties",
        {"causal", "temporal", "transitive", "symmetric"},
        4,
    )
    try:
        hops = min(5, max(1, int(grounded.get("max_reasoning_hops") or 3)))
    except (TypeError, ValueError):
        hops = 3
    plan = query_logic.GraphQueryPlan(
        operation="describe",
        scope="anchored" if selected_names else "top_k",
        anchors=selected_names,
        requested_fields=list(requested_outputs),
        output_shape="prose",
        reasoning_operator=operator,
        requested_outputs=requested_outputs,
        aliases=aliases,
        focus_terms=selected_names,
        competing_terms=excluded_names,
        goal_terms=goal_predicates,
        relation_hints=goal_predicates,
        blocked_relation_hints=blocked_predicates,
        preferred_path_properties=properties,
        grounded_entity_ids=selected_entity_ids,
        grounded_frame_ids=selected_frame_ids,
        grounded_relationship_ids=selected_relationship_ids,
        max_reasoning_hops=hops,
    )
    diagnostics = {
        "status": "grounded",
        "intent": intent,
        "candidate_counts": {
            "entities": len(entity_candidates),
            "frames": len(frame_candidates),
            "relationships": len(relationship_candidates),
        },
        "selection": {
            "entity_ids": selected_entity_ids,
            "frame_ids": selected_frame_ids,
            "relationship_ids": selected_relationship_ids,
            "goal_predicates": goal_predicates,
            "blocked_predicates": blocked_predicates,
        },
        "llm_selection": {
            "entity_ids": llm_entity_ids,
            "frame_ids": llm_frame_ids,
            "relationship_ids": llm_relationship_ids,
        },
        "retained_branches": {
            "entity_ids": retained_entity_ids,
            "frame_ids": retained_frame_ids,
            "relationship_ids": retained_relationship_ids,
        },
        "timings_seconds": {
            "intent_llm": round(intent_elapsed, 3),
            "candidate_retrieval": round(retrieval_elapsed, 3),
            "grounded_selection_llm": round(selection_elapsed, 3),
            "total": round(time.perf_counter() - started, 3),
        },
        "candidates": candidate_pack,
    }
    return plan, diagnostics


async def _latest_guidance_projection(collection_id: uuid.UUID) -> uuid.UUID | None:
    async with AsyncSessionLocal() as session:
        return (
            await session.execute(
                select(GraphProjectionSnapshot.id)
                .join(
                    GraphVersion,
                    GraphVersion.id == GraphProjectionSnapshot.graph_version_id,
                )
                .where(
                    GraphVersion.collection_id == collection_id,
                    GraphProjectionSnapshot.name == "semantic_affinity_undirected",
                    GraphProjectionSnapshot.status == "completed",
                )
                .order_by(GraphProjectionSnapshot.completed_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()


async def _guided_candidate_metadata(
    entity_ids: set[uuid.UUID],
    projection_id: uuid.UUID | None,
) -> tuple[
    dict[uuid.UUID, tuple[str, str]],
    dict[uuid.UUID, dict[str, float]],
    dict[uuid.UUID, set[uuid.UUID]],
]:
    if not entity_ids:
        return {}, {}, {}
    async with AsyncSessionLocal() as session:
        entity_rows = (
            await session.execute(
                select(
                    GraphEntity.id,
                    GraphEntity.canonical_name,
                    GraphEntity.primary_type,
                ).where(GraphEntity.id.in_(entity_ids))
            )
        ).all()
        description_rows = (
            await session.execute(
                select(EntityDescription.entity_id, EntityDescription.description)
                .where(EntityDescription.entity_id.in_(entity_ids))
                .order_by(EntityDescription.weight.desc())
            )
        ).all()
        metric_rows = []
        membership_rows = []
        if projection_id is not None:
            metric_rows = (
                await session.execute(
                    select(
                        GraphNodeMetric.entity_id,
                        GraphNodeMetric.metric,
                        GraphNodeMetric.value,
                    ).where(
                        GraphNodeMetric.projection_id == projection_id,
                        GraphNodeMetric.entity_id.in_(entity_ids),
                        GraphNodeMetric.metric.in_(
                            ["betweenness_approx", "is_articulation", "pagerank"]
                        ),
                    )
                )
            ).all()
            membership_rows = (
                await session.execute(
                    select(
                        GraphCommunityMembership.entity_id,
                        GraphCommunityMembership.community_id,
                    )
                    .join(
                        GraphCommunity,
                        GraphCommunity.id == GraphCommunityMembership.community_id,
                    )
                    .where(
                        GraphCommunity.projection_id == projection_id,
                        GraphCommunityMembership.entity_id.in_(entity_ids),
                    )
                )
            ).all()
    descriptions: dict[uuid.UUID, str] = {}
    for entity_id, description in description_rows:
        descriptions.setdefault(entity_id, str(description or ""))
    entities = {
        entity_id: (
            str(name or ""),
            f"{primary_type or ''} {descriptions.get(entity_id, '')}",
        )
        for entity_id, name, primary_type in entity_rows
    }
    metrics: dict[uuid.UUID, dict[str, float]] = defaultdict(dict)
    maxima: dict[str, float] = defaultdict(float)
    for entity_id, metric, value in metric_rows:
        numeric = float(value or 0.0)
        metrics[entity_id][str(metric)] = numeric
        maxima[str(metric)] = max(maxima[str(metric)], numeric)
    for values in metrics.values():
        for metric in ("betweenness_approx", "pagerank"):
            maximum = maxima.get(metric, 0.0)
            values[metric] = values.get(metric, 0.0) / maximum if maximum else 0.0
    communities: dict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)
    for entity_id, community_id in membership_rows:
        communities[entity_id].add(community_id)
    return entities, metrics, communities


async def _guided_branch_state(
    collection: Collection,
    question: str,
    plan: query_logic.GraphQueryPlan,
    planner_diagnostics: dict[str, Any],
    *,
    document_ids: list[uuid.UUID] | None,
    max_nodes: int = 180,
    max_edges: int = 320,
    frontier_batch: int = 24,
    alternatives_per_batch: int = 48,
) -> tuple[query_logic.GraphQueryState, dict[str, Any]]:
    started = time.perf_counter()
    selection = dict(planner_diagnostics.get("selection") or {})
    seed_ids = {
        uuid.UUID(value) for value in selection.get("entity_ids") or [] if value
    }
    frame_ids = {
        uuid.UUID(value) for value in selection.get("frame_ids") or [] if value
    }
    selected_relationship_ids = {
        uuid.UUID(value) for value in selection.get("relationship_ids") or [] if value
    }
    if frame_ids:
        async with AsyncSessionLocal() as session:
            frame_argument_ids = (
                await session.execute(
                    select(GraphFrameArgument.entity_id).where(
                        GraphFrameArgument.frame_id.in_(frame_ids)
                    )
                )
            ).scalars()
        seed_ids.update(frame_argument_ids)
    projection_id = await _latest_guidance_projection(collection.id)
    focus_tokens = goal_tokens(
        " ".join(
            [
                question,
                *plan.requested_outputs,
                *plan.goal_terms,
                *plan.relation_hints,
            ]
        )
    )
    preferred_properties = {
        value.casefold() for value in plan.preferred_path_properties
    }
    node_scores = {entity_id: 1.0 for entity_id in seed_ids}
    node_depth = {entity_id: 0 for entity_id in seed_ids}
    queue = [(-1.0, 0, str(entity_id), entity_id) for entity_id in seed_ids]
    heapq.heapify(queue)
    expanded: set[uuid.UUID] = set()
    retained_edges: dict[uuid.UUID, GraphRelationship] = {}
    rel_scores: dict[str, float] = {}
    rel_combined_scores: dict[str, float] = {}
    relation_types_seen: set[str] = set()
    communities_seen: set[uuid.UUID] = set()
    batches: list[dict[str, Any]] = []

    while queue and len(node_scores) < max_nodes and len(retained_edges) < max_edges:
        frontier: list[uuid.UUID] = []
        while queue and len(frontier) < frontier_batch:
            _, depth, _, entity_id = heapq.heappop(queue)
            if entity_id in expanded or depth >= plan.max_reasoning_hops:
                continue
            expanded.add(entity_id)
            frontier.append(entity_id)
        if not frontier:
            break
        query_started = time.perf_counter()
        relationship_conditions = [
            GraphRelationship.collection_id == collection.id,
            or_(
                GraphRelationship.source_entity_id.in_(frontier),
                GraphRelationship.target_entity_id.in_(frontier),
            ),
        ]
        if document_ids:
            relationship_conditions.append(
                GraphRelationship.id.in_(
                    select(RelationshipDescription.relationship_id).where(
                        RelationshipDescription.document_id.in_(document_ids)
                    )
                )
            )
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(
                        GraphRelationship,
                        GraphRelationshipType.inferred_properties,
                    )
                    .join(
                        GraphRelationshipType,
                        GraphRelationshipType.id
                        == GraphRelationship.relationship_type_id,
                    )
                    .where(*relationship_conditions)
                    .limit(2500)
                )
            ).all()
        query_elapsed = time.perf_counter() - query_started
        candidate_ids = {
            endpoint
            for relationship, _ in rows
            for endpoint in (
                relationship.source_entity_id,
                relationship.target_entity_id,
            )
            if endpoint not in node_scores
        }
        entities, metrics, communities = await _guided_candidate_metadata(
            candidate_ids | set(frontier), projection_id
        )
        transitions: list[tuple[float, str, uuid.UUID, GraphRelationship]] = []
        for relationship, inferred_properties in rows:
            source_id = relationship.source_entity_id
            target_id = relationship.target_entity_id
            if source_id in frontier:
                parent_id, candidate_id = source_id, target_id
            elif target_id in frontier:
                parent_id, candidate_id = target_id, source_id
            else:
                continue
            if candidate_id in expanded:
                continue
            rel_type = str(relationship.rel_type or "").upper()
            rel_overlap = len(focus_tokens & goal_tokens(rel_type.replace("_", " ")))
            name, description = entities.get(candidate_id, ("", ""))
            node_overlap = len(focus_tokens & goal_tokens(f"{name} {description}"))
            properties = dict(inferred_properties or {})
            property_score = 0.0
            if (
                "causal" in preferred_properties
                and properties.get("causal") == "causal"
            ):
                property_score += 2.0
            if (
                "temporal" in preferred_properties
                and properties.get("temporal") == "temporal"
            ):
                property_score += 2.0
            if (
                "transitive" in preferred_properties
                and properties.get("transitivity") == "transitive"
            ):
                property_score += 1.5
            candidate_metrics = metrics.get(candidate_id, {})
            parent_communities = communities.get(parent_id, set())
            candidate_communities = communities.get(candidate_id, set())
            crosses_community = bool(
                parent_communities
                and candidate_communities
                and parent_communities.isdisjoint(candidate_communities)
            )
            novelty = float(rel_type not in relation_types_seen)
            community_novelty = float(bool(candidate_communities - communities_seen))
            parent_score = node_scores.get(parent_id, 0.0)
            score = (
                parent_score * 0.72
                + node_overlap * 2.5
                + rel_overlap * 2.0
                + property_score
                + candidate_metrics.get("betweenness_approx", 0.0) * 2.5
                + candidate_metrics.get("is_articulation", 0.0) * 2.0
                + candidate_metrics.get("pagerank", 0.0) * 0.5
                + float(crosses_community) * 1.5
                + novelty * 0.35
                + community_novelty * 0.35
            )
            if relationship.id in selected_relationship_ids:
                score += 8.0
            transitions.append(
                (score, str(relationship.id), candidate_id, relationship)
            )
        transitions.sort(key=lambda item: (-item[0], item[1]))
        accepted = 0
        accepted_signatures: set[tuple[str, tuple[str, ...]]] = set()
        deferred: list[tuple[float, str, uuid.UUID, GraphRelationship]] = []
        for transition in transitions:
            _, _, candidate_id, relationship = transition
            signature = (
                str(relationship.rel_type or "").upper(),
                tuple(
                    sorted(str(value) for value in communities.get(candidate_id, set()))
                ),
            )
            if signature in accepted_signatures:
                deferred.append(transition)
                continue
            accepted_signatures.add(signature)
            deferred.insert(0, transition)
        for score, _, candidate_id, relationship in deferred:
            if accepted >= alternatives_per_batch or len(node_scores) >= max_nodes:
                break
            if relationship.id in retained_edges:
                continue
            retained_edges[relationship.id] = relationship
            rel_id = str(relationship.id)
            normalized_score = max(0.0, score)
            rel_scores[rel_id] = normalized_score
            rel_combined_scores[rel_id] = normalized_score
            previous = node_scores.get(candidate_id)
            if previous is None or score > previous:
                node_scores[candidate_id] = score
                parent_depth = min(
                    node_depth.get(relationship.source_entity_id, 999),
                    node_depth.get(relationship.target_entity_id, 999),
                )
                depth = min(parent_depth + 1, plan.max_reasoning_hops)
                node_depth[candidate_id] = depth
                heapq.heappush(queue, (-score, depth, str(candidate_id), candidate_id))
            relation_types_seen.add(str(relationship.rel_type or "").upper())
            communities_seen.update(communities.get(candidate_id, set()))
            accepted += 1
        batches.append(
            {
                "frontier": len(frontier),
                "edge_rows": len(rows),
                "candidate_nodes": len(candidate_ids),
                "accepted_alternatives": accepted,
                "sql_seconds": round(query_elapsed, 4),
            }
        )

    traversed_ids = list(retained_edges)
    discovered_ids = {str(value) for value in node_scores}
    diagnostics = {
        "strategy": "burner_guided_best_first",
        "seed_count": len(seed_ids),
        "node_count": len(discovered_ids),
        "edge_count": len(traversed_ids),
        "expanded_count": len(expanded),
        "projection_id": str(projection_id) if projection_id else None,
        "batches": batches,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    return (
        query_logic.GraphQueryState(
            discovered_entity_ids=discovered_ids,
            entity_relevance={str(key): value for key, value in node_scores.items()},
            traversed_rel_ids=[str(value) for value in traversed_ids],
            rel_score_cache=rel_scores,
            rel_combined_score_cache=rel_combined_scores,
        ),
        diagnostics,
    )


async def _production_query_artifacts(
    collection: Collection,
    question: str,
    *,
    max_hops: int | None,
    grounded_plan: bool,
) -> tuple[
    query_logic.GraphQueryPlan,
    query_logic.DocumentRoutingDecision,
    query_logic.GraphQueryArtifacts,
    list[tuple[Collection, query_logic.GraphQueryArtifacts]],
    str,
    dict[str, Any],
]:
    pipeline_started = time.perf_counter()
    routing_started = time.perf_counter()
    routing = await query_logic._resolve_document_routing(
        question,
        collection,
        collection.namespace_id,
        collection.llm_profile_id,
    )
    routing_elapsed = time.perf_counter() - routing_started
    document_ids = routing.document_ids if not routing.use_all_documents else None
    planning_started = time.perf_counter()
    if grounded_plan:
        plan, planner_diagnostics = await _grounded_query_plan(
            collection, question, document_ids
        )
    else:
        plan = await query_logic._plan_graph_query(
            question,
            collection.namespace_id,
            collection.llm_profile_id,
        )
        planner_diagnostics = {"status": "production_one_shot"}
    planning_elapsed = time.perf_counter() - planning_started
    if max_hops is not None:
        plan = replace(plan, max_reasoning_hops=min(5, max(1, max_hops)))
    base_started = time.perf_counter()
    base = await query_logic._build_graph_query_artifacts(
        question,
        collection,
        collection.namespace_id,
        "mix",
        collection.llm_profile_id,
        document_ids=document_ids,
        query_plan=plan,
    )
    base_elapsed = time.perf_counter() - base_started
    meta_started = time.perf_counter()
    meta_artifacts = [
        (
            meta_collection,
            await query_logic._build_graph_query_artifacts(
                question,
                meta_collection,
                collection.namespace_id,
                "mix",
                collection.llm_profile_id,
                document_ids=None,
                query_plan=plan,
            ),
        )
        for meta_collection in await query_logic._load_meta_collections(collection)
    ]
    meta_elapsed = time.perf_counter() - meta_started
    if meta_artifacts:
        projection_state = await query_logic._meta_projection_state(
            question,
            collection,
            meta_artifacts,
            document_ids=document_ids,
        )
        if projection_state.discovered_entity_ids or projection_state.traversed_rel_ids:
            semantic_suffix = query_logic._semantic_frame_suffix(base.context)
            projected_state = query_logic._merge_states(base.state, projection_state)
            (
                context,
                entities,
                relationships,
                rel_context,
            ) = await query_logic._build_context(
                projected_state,
                collection,
                document_ids=document_ids,
            )
            if semantic_suffix:
                context = f"{context}\n\n{semantic_suffix}"
                rel_context = f"{rel_context}\n{semantic_suffix}".strip()
            base = replace(
                base,
                context=context,
                entities_used=entities,
                relationships_used=relationships,
                rel_context=rel_context,
                state=projected_state,
            )
    guided_state, guided_diagnostics = await _guided_branch_state(
        collection,
        question,
        plan,
        planner_diagnostics,
        document_ids=document_ids,
    )
    if guided_state.discovered_entity_ids or guided_state.traversed_rel_ids:
        (
            guided_context,
            guided_entities,
            guided_relationships,
            guided_rel_context,
        ) = await query_logic._build_context(
            guided_state,
            collection,
            document_ids=document_ids,
        )
        base = replace(
            base,
            context=(
                f"{base.context}\n\nBurner Guided Branch Evidence:\n"
                f"{query_logic._strip_context_label(guided_context)}"
            ),
            entities_used=list(dict.fromkeys([*base.entities_used, *guided_entities])),
            relationships_used=list(
                dict.fromkeys([*base.relationships_used, *guided_relationships])
            ),
            rel_context=(
                f"{base.rel_context}\nBurner Guided Branch Evidence:\n"
                f"{guided_rel_context}"
            ).strip(),
            state=query_logic._merge_states(base.state, guided_state),
        )
    planner_diagnostics["guided_search"] = guided_diagnostics
    if meta_artifacts:
        meta_sections = "\n\n".join(
            f"Level {query_logic.meta_collection_level(meta.name)} ({meta.name}):\n"
            f"{query_logic._strip_context_label(artifacts.context)}"
            for meta, artifacts in meta_artifacts
        )
        context = (
            f"Internal Higher-Level Context:\n{meta_sections}\n\n"
            f"Primary Evidence:\n{query_logic._strip_context_label(base.context)}"
        )
    else:
        context = base.context
    planner_diagnostics["pipeline_timings_seconds"] = {
        "document_routing": round(routing_elapsed, 3),
        "planning": round(planning_elapsed, 3),
        "base_artifacts": round(base_elapsed, 3),
        "meta_artifacts": round(meta_elapsed, 3),
        "total": round(time.perf_counter() - pipeline_started, 3),
    }
    return plan, routing, base, meta_artifacts, context, planner_diagnostics


async def main() -> None:
    args = parse_args()
    collection, _ = await load_collection(args.collection_id)
    (
        plan,
        routing,
        base,
        meta_artifacts,
        context,
        planner_diagnostics,
    ) = await _production_query_artifacts(
        collection,
        args.question,
        max_hops=args.max_hops,
        grounded_plan=not args.ungrounded_plan,
    )
    payload = {
        "pipeline": (
            "production_custom_graph_rag_mix_grounded_planner"
            if not args.ungrounded_plan
            else "production_custom_graph_rag_mix"
        ),
        "plan": asdict(plan),
        "planner": planner_diagnostics,
        "routing": {
            "use_all_documents": routing.use_all_documents,
            "document_ids": [str(value) for value in routing.document_ids],
        },
        "base": _production_artifact_payload(collection, base),
        "meta": [
            _production_artifact_payload(meta, artifacts)
            for meta, artifacts in meta_artifacts
        ],
    }
    if args.trace_only:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return
    fallback = base.rel_context or "\n".join(base.entities_used)
    print(
        await query_logic._answer_from_context(
            args.question,
            collection.namespace_id,
            collection.llm_profile_id,
            context,
            fallback,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

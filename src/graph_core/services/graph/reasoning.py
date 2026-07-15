"""Bounded deterministic activation over executable custom-graph structure."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import case, or_, select

from graph_core.database import AsyncSessionLocal
from graph_core.models.graph_rag import (
    EntityDescription,
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
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

_STRUCTURAL_TYPES = {
    "ANTECEDENT_OF",
    "BLOCKS",
    "CONCLUDES",
    "CONDITION",
    "EXCEPTION",
    "OBJECT",
    "SUBJECT",
}
_STOPWORDS = {
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "how",
    "should",
    "the",
    "their",
    "this",
    "what",
    "when",
    "which",
    "with",
    "why",
}


@dataclass(frozen=True, slots=True)
class ReasoningArgument:
    role: str
    entity_id: uuid.UUID | None = None
    variable_name: str | None = None
    literal_value: Any | None = None


@dataclass(frozen=True, slots=True)
class ReasoningSeed:
    proposition_id: uuid.UUID
    frame_text: str
    argument_ids: tuple[uuid.UUID, ...]
    conditions: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    polarity: str = "positive"
    modality: str = "asserted"
    retrieval_score: float = 0.0
    predicate: str = ""
    arguments: tuple[ReasoningArgument, ...] = ()


@dataclass(frozen=True, slots=True)
class _Node:
    id: uuid.UUID
    node_type: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class _Edge:
    source_id: uuid.UUID
    target_id: uuid.UUID
    rel_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Frame:
    proposition_id: uuid.UUID
    predicate: str
    frame_text: str
    modality: str
    conditions: tuple[str, ...]
    arguments: tuple[ReasoningArgument, ...]
    retrieval_score: float = 0.0


@dataclass(frozen=True, slots=True)
class _Goal:
    predicate: str
    arguments: tuple[ReasoningArgument, ...]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{2,}", text.casefold())
        if token not in _STOPWORDS
    }


def _goal_token(value: str) -> str:
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        return value[:-1]
    if value.endswith("ing") and len(value) > 5:
        return value[:-3]
    return value


def _goal_tokens(text: str) -> set[str]:
    return {_goal_token(token) for token in _tokens(text)}


def _matches_anchor(node: _Node, anchor_terms: list[str]) -> bool:
    node_tokens = _goal_tokens(f"{node.name} {node.description}")
    return any(
        anchor_tokens and anchor_tokens <= node_tokens
        for anchor in anchor_terms
        if (anchor_tokens := _goal_tokens(anchor))
    )


def _operator(question: str) -> str:
    lowered = question.casefold()
    if any(value in lowered for value in ("redesign", "refactor", "restructure")):
        return "redesign"
    if any(value in lowered for value in (" vs ", "versus", "when should")):
        return "choose"
    if any(value in lowered for value in ("good", "bad", "ugly", "evaluate")):
        return "evaluate"
    if any(
        value in lowered
        for value in ("how to", "how and when", "steps", "procedure", "instructions")
    ):
        return "procedure"
    if any(value in lowered for value in ("why", "how does", "mechanism")):
        return "explain"
    return "prove_or_disprove"


async def _load_nodes(node_ids: set[uuid.UUID]) -> dict[uuid.UUID, _Node]:
    if not node_ids:
        return {}
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(
                    GraphEntity.id,
                    GraphEntity.primary_type,
                    GraphEntity.canonical_name,
                ).where(GraphEntity.id.in_(node_ids))
            )
        ).all()
        descriptions = (
            await session.execute(
                select(EntityDescription.entity_id, EntityDescription.description)
                .where(EntityDescription.entity_id.in_(node_ids))
                .order_by(EntityDescription.created_at.desc())
            )
        ).all()
    description_by_id: dict[uuid.UUID, str] = {}
    for entity_id, description in descriptions:
        description_by_id.setdefault(entity_id, str(description or ""))
    return {
        row.id: _Node(
            row.id,
            str(row.primary_type or "").upper(),
            str(row.canonical_name or ""),
            description_by_id.get(row.id, ""),
        )
        for row in rows
    }


async def _community_landmarks(
    collection_id: uuid.UUID,
    seed_ids: set[uuid.UUID],
    *,
    limit: int = 24,
) -> set[uuid.UUID]:
    """Select central nodes only from communities touched by retrieval seeds."""
    if not seed_ids or limit <= 0:
        return set()
    async with AsyncSessionLocal() as session:
        version_id = (
            await session.execute(
                select(GraphVersion.id)
                .where(GraphVersion.collection_id == collection_id)
                .order_by(GraphVersion.version.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if version_id is None:
            return set()
        projection_id = (
            await session.execute(
                select(GraphProjectionSnapshot.id)
                .where(
                    GraphProjectionSnapshot.graph_version_id == version_id,
                    GraphProjectionSnapshot.name == "semantic_affinity_undirected",
                    GraphProjectionSnapshot.status == "completed",
                )
                .order_by(GraphProjectionSnapshot.completed_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if projection_id is None:
            return set()
        community_ids = set(
            (
                await session.execute(
                    select(GraphCommunityMembership.community_id)
                    .join(
                        GraphCommunity,
                        GraphCommunity.id == GraphCommunityMembership.community_id,
                    )
                    .where(
                        GraphCommunity.projection_id == projection_id,
                        GraphCommunityMembership.entity_id.in_(seed_ids),
                    )
                )
            ).scalars()
        )
        if not community_ids:
            return set()
        rows = (
            await session.execute(
                select(GraphNodeMetric.entity_id)
                .join(
                    GraphCommunityMembership,
                    GraphCommunityMembership.entity_id == GraphNodeMetric.entity_id,
                )
                .where(
                    GraphNodeMetric.projection_id == projection_id,
                    GraphNodeMetric.metric == "pagerank",
                    GraphCommunityMembership.community_id.in_(community_ids),
                    GraphNodeMetric.entity_id.not_in(seed_ids),
                )
                .order_by(GraphNodeMetric.value.desc())
                .limit(limit)
            )
        ).all()
    return {row.entity_id for row in rows}


async def _bounded_graph(
    collection_id: uuid.UUID,
    seed_ids: set[uuid.UUID],
    *,
    max_hops: int = 3,
    max_nodes: int = 500,
    max_edges: int = 900,
    relation_hints: list[str] | None = None,
    blocked_relation_hints: list[str] | None = None,
    preferred_path_properties: list[str] | None = None,
    timings: dict[str, Any] | None = None,
) -> tuple[dict[uuid.UUID, _Node], list[_Edge]]:
    started = time.perf_counter()
    initial_nodes_started = time.perf_counter()
    nodes = await _load_nodes(seed_ids)
    initial_nodes_elapsed = time.perf_counter() - initial_nodes_started
    landmarks_started = time.perf_counter()
    landmark_ids = await _community_landmarks(collection_id, set(nodes))
    nodes.update(await _load_nodes(landmark_ids))
    landmarks_elapsed = time.perf_counter() - landmarks_started
    frontier = set(nodes)
    edges: dict[tuple[uuid.UUID, uuid.UUID, str], _Edge] = {}
    structural_first = case(
        (GraphRelationship.rel_type.in_(_STRUCTURAL_TYPES), 0), else_=1
    )
    hop_timings: list[dict[str, Any]] = []
    for hop in range(max_hops):
        if not frontier or len(nodes) >= max_nodes or len(edges) >= max_edges:
            break
        remaining_edges = max_edges - len(edges)
        edge_query_started = time.perf_counter()
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
                    .where(
                        GraphRelationship.collection_id == collection_id,
                        or_(
                            GraphRelationship.source_entity_id.in_(frontier),
                            GraphRelationship.target_entity_id.in_(frontier),
                        ),
                    )
                    .order_by(structural_first, GraphRelationship.weight.desc())
                    .limit(min(max(remaining_edges * 4, remaining_edges), 5000))
                )
            ).all()
        edge_query_elapsed = time.perf_counter() - edge_query_started
        candidate_edges: list[_Edge] = []
        for row, predicate_properties in rows:
            candidate_edges.append(
                _Edge(
                    row.source_entity_id,
                    row.target_entity_id,
                    str(row.rel_type or "").upper(),
                    dict(predicate_properties or {}),
                )
            )
        candidate_edges.sort(
            key=lambda edge: (
                -_planned_edge_priority(
                    edge,
                    relation_hints=relation_hints,
                    preferred_path_properties=preferred_path_properties,
                ),
                str(edge.source_id),
                edge.rel_type,
                str(edge.target_id),
            )
        )
        new_ids: set[uuid.UUID] = set()
        for edge in candidate_edges:
            if len(edges) >= max_edges:
                break
            if edge.rel_type not in _STRUCTURAL_TYPES and _matches_relation_hint(
                edge.rel_type, blocked_relation_hints or []
            ):
                continue
            edges[(edge.source_id, edge.target_id, edge.rel_type)] = edge
            if edge.source_id not in nodes:
                new_ids.add(edge.source_id)
            if edge.target_id not in nodes:
                new_ids.add(edge.target_id)
        remaining = max_nodes - len(nodes)
        node_load_started = time.perf_counter()
        loaded = await _load_nodes(set(sorted(new_ids, key=str)[:remaining]))
        node_load_elapsed = time.perf_counter() - node_load_started
        nodes.update(loaded)
        frontier = set(loaded)
        hop_timings.append(
            {
                "hop": hop + 1,
                "frontier": len(frontier),
                "edge_rows": len(rows),
                "edge_query": round(edge_query_elapsed, 3),
                "node_load": round(node_load_elapsed, 3),
            }
        )
    if timings is not None:
        timings.update(
            {
                "initial_nodes": round(initial_nodes_elapsed, 3),
                "community_landmarks": round(landmarks_elapsed, 3),
                "hops": hop_timings,
                "total": round(time.perf_counter() - started, 3),
            }
        )
    return nodes, [
        edge
        for edge in edges.values()
        if edge.source_id in nodes and edge.target_id in nodes
    ]


def _matches_relation_hint(rel_type: str, hints: list[str]) -> bool:
    rel_tokens = _goal_tokens(rel_type.replace("_", " "))
    return any(
        hint_tokens and hint_tokens <= rel_tokens
        for hint in hints
        if (hint_tokens := _goal_tokens(hint))
    )


def _planned_edge_priority(
    edge: _Edge,
    *,
    relation_hints: list[str] | None,
    preferred_path_properties: list[str] | None,
) -> float:
    if edge.rel_type in _STRUCTURAL_TYPES:
        return 100.0
    score = 0.0
    if _matches_relation_hint(edge.rel_type, relation_hints or []):
        score += 10.0
    preferred = {value.casefold() for value in preferred_path_properties or []}
    if "causal" in preferred and edge.properties.get("causal") == "causal":
        score += 6.0
    if "temporal" in preferred and edge.properties.get("temporal") == "temporal":
        score += 6.0
    if (
        "transitive" in preferred
        and edge.properties.get("transitivity") == "transitive"
    ):
        score += 4.0
    if "symmetric" in preferred and edge.properties.get("symmetry") == "symmetric":
        score += 2.0
    return score


def _predicate_derivations(
    edges: list[_Edge],
    nodes: dict[uuid.UUID, _Node] | None = None,
    *,
    limit: int = 64,
) -> list[dict[str, str]]:
    semantic = [edge for edge in edges if edge.rel_type not in _STRUCTURAL_TYPES]
    known = {(edge.source_id, edge.rel_type, edge.target_id) for edge in semantic}
    derived: list[dict[str, str]] = []
    for edge in semantic:
        if edge.properties.get("symmetry") != "symmetric":
            continue
        reverse = (edge.target_id, edge.rel_type, edge.source_id)
        if reverse in known:
            continue
        known.add(reverse)
        derived.append(
            {
                "source": str(edge.target_id),
                **(
                    {"source_name": nodes[edge.target_id].name}
                    if nodes and edge.target_id in nodes
                    else {}
                ),
                "target": str(edge.source_id),
                **(
                    {"target_name": nodes[edge.source_id].name}
                    if nodes and edge.source_id in nodes
                    else {}
                ),
                "predicate": edge.rel_type,
                "derivation": "symmetry",
            }
        )
        if len(derived) >= limit:
            return derived

    transitive_types = {
        edge.rel_type
        for edge in semantic
        if edge.properties.get("transitivity") == "transitive"
    }
    for rel_type in sorted(transitive_types):
        pairs = {
            (source_id, target_id)
            for source_id, candidate_type, target_id in known
            if candidate_type == rel_type
        }
        changed = True
        while changed and len(derived) < limit:
            changed = False
            additions = {
                (source_id, target_id)
                for source_id, middle_id in pairs
                for candidate_middle, target_id in pairs
                if middle_id == candidate_middle
                and source_id != target_id
                and (source_id, target_id) not in pairs
            }
            for source_id, target_id in sorted(
                additions,
                key=lambda item: (str(item[0]), str(item[1])),
            ):
                pairs.add((source_id, target_id))
                known.add((source_id, rel_type, target_id))
                derived.append(
                    {
                        "source": str(source_id),
                        **(
                            {"source_name": nodes[source_id].name}
                            if nodes and source_id in nodes
                            else {}
                        ),
                        "target": str(target_id),
                        **(
                            {"target_name": nodes[target_id].name}
                            if nodes and target_id in nodes
                            else {}
                        ),
                        "predicate": rel_type,
                        "derivation": "transitivity",
                    }
                )
                changed = True
                if len(derived) >= limit:
                    break
    return derived


def _mechanism_edges(
    edges: list[_Edge],
    nodes: dict[uuid.UUID, _Node],
) -> list[dict[str, str]]:
    candidates = [
        edge
        for edge in edges
        if edge.rel_type not in _STRUCTURAL_TYPES
        and (
            edge.properties.get("causal") == "causal"
            or edge.properties.get("temporal") == "temporal"
        )
    ]
    return [
        {
            "source": str(edge.source_id),
            "source_name": nodes[edge.source_id].name,
            "target": str(edge.target_id),
            "target_name": nodes[edge.target_id].name,
            "predicate": edge.rel_type,
            "causal": str(edge.properties.get("causal") or "unknown"),
            "temporal": str(edge.properties.get("temporal") or "unknown"),
        }
        for edge in candidates[:24]
    ]


def _normalize_predicate(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _term(argument: ReasoningArgument) -> tuple[str, str] | None:
    if argument.entity_id is not None:
        return "entity", str(argument.entity_id)
    if argument.variable_name:
        return "variable", argument.variable_name.lstrip("?")
    if argument.literal_value is not None:
        return "literal", json.dumps(
            argument.literal_value, sort_keys=True, default=str
        )
    return None


def _argument_payload(argument: ReasoningArgument) -> dict[str, Any]:
    return {
        "role": argument.role,
        "entity_id": str(argument.entity_id) if argument.entity_id else None,
        "variable_name": argument.variable_name,
        "literal_value": argument.literal_value,
    }


def _seed_arguments(seed: ReasoningSeed) -> tuple[ReasoningArgument, ...]:
    if seed.arguments:
        return seed.arguments
    roles = ("subject", "object")
    return tuple(
        ReasoningArgument(
            role=roles[index] if index < len(roles) else f"argument_{index}",
            entity_id=entity_id,
        )
        for index, entity_id in enumerate(seed.argument_ids)
    )


async def _load_frames(
    collection_id: uuid.UUID,
    proposition_ids: set[uuid.UUID],
    seeds: list[ReasoningSeed],
) -> list[_Frame]:
    frames_by_proposition = {
        seed.proposition_id: _Frame(
            proposition_id=seed.proposition_id,
            predicate=seed.predicate,
            frame_text=seed.frame_text,
            modality=seed.modality,
            conditions=seed.conditions,
            arguments=_seed_arguments(seed),
            retrieval_score=seed.retrieval_score,
        )
        for seed in seeds
    }
    if not proposition_ids:
        return list(frames_by_proposition.values())
    async with AsyncSessionLocal() as session:
        frames = (
            (
                await session.execute(
                    select(GraphSemanticFrame).where(
                        GraphSemanticFrame.collection_id == collection_id,
                        GraphSemanticFrame.frame_kind == "proposition",
                        GraphSemanticFrame.proposition_entity_id.in_(proposition_ids),
                    )
                )
            )
            .scalars()
            .all()
        )
        frame_ids = [frame.id for frame in frames]
        argument_rows = (
            await session.execute(
                select(
                    GraphFrameArgument.frame_id,
                    GraphFrameArgument.role,
                    GraphFrameArgument.entity_id,
                    GraphFrameArgument.variable_name,
                    GraphFrameArgument.literal_value,
                )
                .where(GraphFrameArgument.frame_id.in_(frame_ids))
                .order_by(GraphFrameArgument.frame_id, GraphFrameArgument.position)
            )
        ).all()
    arguments: dict[uuid.UUID, list[ReasoningArgument]] = defaultdict(list)
    for frame_id, role, entity_id, variable_name, literal_value in argument_rows:
        arguments[frame_id].append(
            ReasoningArgument(
                role=str(role),
                entity_id=entity_id,
                variable_name=str(variable_name) if variable_name else None,
                literal_value=literal_value,
            )
        )
    for frame in frames:
        if frame.proposition_entity_id is None:
            continue
        retrieved = frames_by_proposition.get(frame.proposition_entity_id)
        frames_by_proposition[frame.proposition_entity_id] = _Frame(
            proposition_id=frame.proposition_entity_id,
            predicate=str(frame.predicate or ""),
            frame_text=str(frame.frame_text),
            modality=str(frame.modality or "asserted"),
            conditions=tuple(frame.conditions_json or ()),
            arguments=tuple(arguments[frame.id]),
            retrieval_score=retrieved.retrieval_score if retrieved else 0.0,
        )
    return list(frames_by_proposition.values())


def _unify(
    goal: _Goal,
    frame: _Frame,
    bindings: dict[str, str] | None = None,
) -> dict[str, str] | None:
    if _normalize_predicate(goal.predicate) != _normalize_predicate(frame.predicate):
        return None
    result = dict(bindings or {})
    candidate_by_role = {argument.role: argument for argument in frame.arguments}
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


def _compile_goals(
    question: str,
    nodes: dict[uuid.UUID, _Node],
    frames: list[_Frame],
    *,
    anchor_terms: list[str] | None = None,
    goal_terms: list[str] | None = None,
    limit: int = 8,
) -> list[_Goal]:
    query_tokens = _goal_tokens(" ".join(goal_terms or [])) or _goal_tokens(question)
    planned_anchors = list(anchor_terms or [])
    fallback_anchor_tokens = _goal_tokens(question)
    query_entity_ids = {
        node_id
        for node_id, node in nodes.items()
        if node.node_type not in {"CONDITION", "EXCEPTION", "RULE", "SCOPE"}
        and (
            _matches_anchor(node, planned_anchors)
            if planned_anchors
            else fallback_anchor_tokens
            & _goal_tokens(f"{node.name} {node.description}")
        )
    }
    ranked: list[tuple[int, float, _Frame, _Goal]] = []
    for frame in frames:
        if not frame.predicate:
            continue
        overlap = len(
            query_tokens & _goal_tokens(f"{frame.predicate} {frame.frame_text}")
        )
        arguments = tuple(
            argument
            if argument.entity_id in query_entity_ids or argument.variable_name
            else ReasoningArgument(
                role=argument.role,
                variable_name=f"goal_{argument.role}",
            )
            for argument in frame.arguments
        )
        constants = sum(argument.entity_id is not None for argument in arguments)
        if overlap == 0 and not goal_terms:
            continue
        if constants == 0 and overlap < 2:
            continue
        ranked.append(
            (overlap, frame.retrieval_score, frame, _Goal(frame.predicate, arguments))
        )
    ranked.sort(key=lambda item: (-item[0], -item[1], str(item[2].proposition_id)))
    goals: list[_Goal] = []
    seen: set[tuple[str, tuple[tuple[str, str] | None, ...]]] = set()
    for _, _, _, goal in ranked:
        key = (
            _normalize_predicate(goal.predicate),
            tuple(_term(argument) for argument in goal.arguments),
        )
        if key in seen:
            continue
        seen.add(key)
        goals.append(goal)
        if len(goals) >= limit:
            break
    return goals


def _backward_reason(
    question: str,
    nodes: dict[uuid.UUID, _Node],
    edges: list[_Edge],
    frames: list[_Frame],
    active_nodes: set[uuid.UUID],
    *,
    anchor_terms: list[str] | None = None,
    goal_terms: list[str] | None = None,
    max_depth: int = 4,
    max_states: int = 128,
    max_matches: int = 8,
) -> dict[str, Any]:
    incoming: dict[uuid.UUID, list[_Edge]] = defaultdict(list)
    for edge in edges:
        incoming[edge.target_id].append(edge)
    frame_by_proposition = {frame.proposition_id: frame for frame in frames}
    goals = _compile_goals(
        question,
        nodes,
        frames,
        anchor_terms=anchor_terms,
        goal_terms=goal_terms,
        limit=max_matches,
    )
    explored = 0

    def prove(
        proposition_id: uuid.UUID,
        bindings: dict[str, str],
        depth: int,
        path: set[uuid.UUID],
    ) -> dict[str, Any]:
        nonlocal explored
        explored += 1
        if explored > max_states:
            return {"status": "state_limit", "proposition_id": str(proposition_id)}
        if depth > max_depth:
            return {"status": "depth_limit", "proposition_id": str(proposition_id)}
        if proposition_id in path:
            return {"status": "cycle", "proposition_id": str(proposition_id)}
        frame = frame_by_proposition.get(proposition_id)
        concluding_rules = [
            edge.source_id
            for edge in incoming[proposition_id]
            if edge.rel_type == "CONCLUDES"
        ]
        if not concluding_rules:
            unbound = sorted(
                argument.variable_name.lstrip("?")
                for argument in (frame.arguments if frame else ())
                if argument.variable_name
                and argument.variable_name.lstrip("?") not in bindings
            )
            asserted = bool(
                frame
                and frame.modality == "asserted"
                and not frame.conditions
                and not unbound
            )
            return {
                "status": "asserted" if asserted else "supported_hypothesis",
                "proposition_id": str(proposition_id),
                "bindings": bindings,
                "unbound_variables": unbound,
                "conditions": list(frame.conditions if frame else ()),
            }
        alternatives: list[dict[str, Any]] = []
        for rule_id in concluding_rules[:max_matches]:
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
                node = nodes.get(antecedent_id)
                if antecedent_id in active_nodes:
                    obligations.append(
                        {
                            "status": "satisfied",
                            "node_id": str(antecedent_id),
                            "statement": (node.description or node.name)
                            if node
                            else "",
                        }
                    )
                elif node and node.node_type in {"ASSERTION", "PROPOSITION"}:
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
                            "statement": (node.description or node.name)
                            if node
                            else "",
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
                status = "supported_hypothesis"
            alternatives.append(
                {
                    "rule_id": str(rule_id),
                    "status": status,
                    "active_blockers": [str(value) for value in active_blockers],
                    "obligations": obligations,
                }
            )
        if any(item["status"] == "proved" for item in alternatives):
            status = "proved"
        elif alternatives and all(item["status"] == "blocked" for item in alternatives):
            status = "blocked"
        else:
            status = "supported_hypothesis"
        return {
            "status": status,
            "proposition_id": str(proposition_id),
            "bindings": bindings,
            "alternatives": alternatives,
        }

    goal_results: list[dict[str, Any]] = []
    covered_tokens: set[str] = set()
    for goal in goals:
        matches: list[dict[str, Any]] = []
        for frame in frames:
            bindings = _unify(goal, frame)
            if bindings is None:
                continue
            matches.append(prove(frame.proposition_id, bindings, 0, set()))
            covered_tokens.update(_goal_tokens(f"{frame.predicate} {frame.frame_text}"))
            if len(matches) >= max_matches:
                break
        if any(match["status"] in {"asserted", "proved"} for match in matches):
            status = "proved"
        elif any(match["status"] == "supported_hypothesis" for match in matches):
            status = "supported_hypothesis"
        else:
            status = "unresolved"
        goal_results.append(
            {
                "goal": {
                    "predicate": goal.predicate,
                    "arguments": [
                        _argument_payload(argument) for argument in goal.arguments
                    ],
                },
                "status": status,
                "matches": matches,
            }
        )
    unmatched_terms = sorted(_goal_tokens(question) - covered_tokens)

    def unresolved_obligations(value: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(value, dict):
            if value.get("status") == "unresolved" and "goal" in value:
                found.append(
                    {
                        "status": "unresolved_goal",
                        "goal": value["goal"],
                    }
                )
            if value.get("status") == "supported_hypothesis" and value.get(
                "conditions"
            ):
                found.append(
                    {
                        "status": "unresolved_conditions",
                        "proposition_id": value.get("proposition_id"),
                        "conditions": value["conditions"],
                    }
                )
            if value.get("status") in {
                "blocked",
                "cycle",
                "depth_limit",
                "incomplete_rule",
                "state_limit",
                "unresolved_subgoal",
            }:
                found.append(value)
            for child in value.values():
                found.extend(unresolved_obligations(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(unresolved_obligations(child))
        return found

    missing_bridges = unresolved_obligations(goal_results)
    if unmatched_terms:
        missing_bridges.insert(
            0,
            {
                "status": "unmatched_query_terms",
                "terms": unmatched_terms,
            },
        )
    if goal_results and all(goal["status"] == "proved" for goal in goal_results):
        status = "proved"
    elif any(
        goal["status"] in {"proved", "supported_hypothesis"} for goal in goal_results
    ):
        status = "supported_hypothesis"
    else:
        status = "unresolved"
    if unmatched_terms:
        status = "insufficient_goal_coverage"
    return {
        "status": status,
        "goals": goal_results,
        "unmatched_query_terms": unmatched_terms,
        "missing_bridges": missing_bridges[:24],
        "explored_states": explored,
        "depth_limit": max_depth,
        "state_limit": max_states,
    }


async def activate_reasoning(
    collection_id: uuid.UUID,
    question: str,
    seeds: list[ReasoningSeed],
    *,
    navigation_ids: set[uuid.UUID] | None = None,
    operator: str | None = None,
    requested_outputs: list[str] | None = None,
    aliases: list[str] | None = None,
    anchor_terms: list[str] | None = None,
    goal_terms: list[str] | None = None,
    relation_hints: list[str] | None = None,
    blocked_relation_hints: list[str] | None = None,
    preferred_path_properties: list[str] | None = None,
    max_hops: int = 3,
) -> dict[str, Any]:
    resolved_operator = operator or _operator(question)
    resolved_outputs = list(dict.fromkeys(requested_outputs or []))
    resolved_aliases = list(dict.fromkeys(aliases or []))
    reasoning_plan = {
        "anchor_terms": list(dict.fromkeys(anchor_terms or [])),
        "goal_terms": list(dict.fromkeys(goal_terms or [])),
        "relation_hints": list(dict.fromkeys(relation_hints or [])),
        "blocked_relation_hints": list(dict.fromkeys(blocked_relation_hints or [])),
        "preferred_path_properties": list(
            dict.fromkeys(preferred_path_properties or [])
        ),
        "max_hops": min(5, max(1, int(max_hops))),
    }
    if not seeds and not navigation_ids:
        return {
            "operator": resolved_operator,
            "sufficiency": "insufficient_graph_evidence",
            "covered_query_tokens": [],
            "propositions": [],
            "rules": [],
            "conflicts": [],
            "goal_reasoning": {
                "status": "unresolved",
                "goals": [],
                "unmatched_query_terms": sorted(_goal_tokens(question)),
                "missing_bridges": [
                    {
                        "status": "unmatched_query_terms",
                        "terms": sorted(_goal_tokens(question)),
                    }
                ],
            },
            "answer_contract": {
                "status": "unresolved",
                "requested_outputs": resolved_outputs,
                "aliases": resolved_aliases,
                "reasoning_plan": reasoning_plan,
                "proved_claims": [],
                "supported_hypotheses": [],
                "missing_bridges": [
                    {
                        "status": "unmatched_query_terms",
                        "terms": sorted(_goal_tokens(question)),
                    }
                ],
                "policy": "No graph evidence is available for an answer.",
            },
            "fixed_point": True,
        }
    query_tokens = _tokens(question)
    frame_tokens = set().union(*(_tokens(seed.frame_text) for seed in seeds))
    covered = sorted(query_tokens & frame_tokens)
    seed_ids = {
        entity_id
        for seed in seeds
        for entity_id in (seed.proposition_id, *seed.argument_ids)
    }
    seed_ids.update(navigation_ids or set())
    bounded_hops = min(5, max(1, int(max_hops)))
    bounded_graph_timings: dict[str, Any] = {}
    nodes, edges = await _bounded_graph(
        collection_id,
        seed_ids,
        max_hops=bounded_hops,
        relation_hints=relation_hints,
        blocked_relation_hints=blocked_relation_hints,
        preferred_path_properties=preferred_path_properties,
        timings=bounded_graph_timings,
    )

    async def prepare_goal_reasoning(
        current_nodes: dict[uuid.UUID, _Node],
        current_edges: list[_Edge],
        *,
        max_depth: int,
        max_states: int,
    ) -> tuple[list[_Frame], set[uuid.UUID], dict[str, Any]]:
        proposition_ids = {
            node_id
            for node_id, node in current_nodes.items()
            if node.node_type in {"ASSERTION", "PROPOSITION"}
        }
        current_frames = await _load_frames(collection_id, proposition_ids, seeds)
        active_nodes = {entity_id for seed in seeds for entity_id in seed.argument_ids}
        active_nodes.update(navigation_ids or set())
        rule_conclusions = {
            edge.target_id for edge in current_edges if edge.rel_type == "CONCLUDES"
        }
        active_nodes.update(
            seed.proposition_id
            for seed in seeds
            if seed.modality == "asserted"
            and not seed.conditions
            and seed.proposition_id not in rule_conclusions
        )
        for node_id, node in current_nodes.items():
            if node.node_type in {"CONDITION", "EXCEPTION"} and (
                query_tokens & _tokens(f"{node.name} {node.description}")
            ):
                active_nodes.add(node_id)
        return (
            current_frames,
            active_nodes,
            _backward_reason(
                question,
                current_nodes,
                current_edges,
                current_frames,
                active_nodes,
                anchor_terms=anchor_terms,
                goal_terms=goal_terms,
                max_depth=max_depth,
                max_states=max_states,
            ),
        )

    frames, active, backward = await prepare_goal_reasoning(
        nodes,
        edges,
        max_depth=bounded_hops + 1,
        max_states=128,
    )
    search_hops = bounded_hops
    predicate_derivations = _predicate_derivations(edges, nodes)
    mechanism_edges = _mechanism_edges(edges, nodes)
    incoming: dict[uuid.UUID, list[_Edge]] = defaultdict(list)
    outgoing: dict[uuid.UUID, list[_Edge]] = defaultdict(list)
    for edge in edges:
        incoming[edge.target_id].append(edge)
        outgoing[edge.source_id].append(edge)
    rules = {node_id for node_id, node in nodes.items() if node.node_type == "RULE"}
    derived: set[uuid.UUID] = set()
    for _ in range(len(rules) + 1):
        changed = False
        for rule_id in sorted(rules, key=str):
            antecedents = {
                edge.source_id
                for edge in incoming[rule_id]
                if edge.rel_type == "ANTECEDENT_OF"
            }
            blockers = {
                edge.source_id
                for edge in incoming[rule_id]
                if edge.rel_type == "BLOCKS"
            }
            conclusions = {
                edge.target_id
                for edge in outgoing[rule_id]
                if edge.rel_type == "CONCLUDES"
            }
            if antecedents and antecedents <= active and not blockers & active:
                new = conclusions - active
                if new:
                    active.update(new)
                    derived.update(new)
                    changed = True
        if not changed:
            break
    rule_traces: list[dict[str, Any]] = []
    statuses_by_conclusion: dict[uuid.UUID, set[str]] = defaultdict(set)
    for rule_id in sorted(rules, key=str):
        antecedents = {
            edge.source_id
            for edge in incoming[rule_id]
            if edge.rel_type == "ANTECEDENT_OF"
        }
        blockers = {
            edge.source_id for edge in incoming[rule_id] if edge.rel_type == "BLOCKS"
        }
        conclusions = {
            edge.target_id for edge in outgoing[rule_id] if edge.rel_type == "CONCLUDES"
        }
        active_blockers = blockers & active
        if not antecedents or not conclusions:
            status = "incomplete"
        elif active_blockers:
            status = "blocked"
        elif antecedents <= active:
            status = "fired"
        else:
            status = "conditional"
        for conclusion in conclusions:
            statuses_by_conclusion[conclusion].add(status)
        rule_traces.append(
            {
                "rule_id": str(rule_id),
                "status": status,
                "antecedents": [str(value) for value in sorted(antecedents, key=str)],
                "active_antecedents": [
                    str(value) for value in sorted(antecedents & active, key=str)
                ],
                "blockers": [str(value) for value in sorted(blockers, key=str)],
                "active_blockers": [
                    str(value) for value in sorted(active_blockers, key=str)
                ],
                "conclusions": [str(value) for value in sorted(conclusions, key=str)],
            }
        )
    proposition_traces = []
    frames_by_proposition = {frame.proposition_id: frame for frame in frames}
    seeded_propositions: set[uuid.UUID] = set()
    for seed in seeds:
        seeded_propositions.add(seed.proposition_id)
        statuses = statuses_by_conclusion.get(seed.proposition_id, set())
        if "blocked" in statuses:
            status = "blocked"
        elif "fired" in statuses or seed.proposition_id in derived:
            status = "derived"
        elif "conditional" in statuses:
            status = "conditional"
        elif "incomplete" in statuses:
            status = "incomplete"
        elif seed.modality != "asserted" or seed.conditions:
            status = "conditional"
        else:
            status = "asserted"
        proposition_traces.append(
            {
                "proposition_id": str(seed.proposition_id),
                "status": status,
                "statement": seed.frame_text,
                "conditions": list(seed.conditions),
                "exceptions": list(seed.exceptions),
                "polarity": seed.polarity,
                "modality": seed.modality,
                "score": seed.retrieval_score
                + (0.08 * len(query_tokens & _tokens(seed.frame_text))),
            }
        )
    for node_id, node in nodes.items():
        if node_id in seeded_propositions or node.node_type not in {
            "ASSERTION",
            "PROPOSITION",
        }:
            continue
        statuses = statuses_by_conclusion.get(node_id, set())
        if "blocked" in statuses:
            status = "blocked"
        elif "fired" in statuses or node_id in derived:
            status = "derived"
        elif "conditional" in statuses:
            status = "conditional"
        elif "incomplete" in statuses:
            status = "incomplete"
        elif node_id in frames_by_proposition and (
            frames_by_proposition[node_id].modality != "asserted"
            or frames_by_proposition[node_id].conditions
        ):
            status = "conditional"
        else:
            status = "asserted"
        proposition_traces.append(
            {
                "proposition_id": str(node_id),
                "status": status,
                "statement": node.description or node.name,
                "conditions": [],
                "exceptions": [],
                "polarity": "unknown",
                "modality": "asserted",
                "score": (0.08 * len(query_tokens & _tokens(node.description)))
                + (0.15 if status == "derived" else 0.0),
            }
        )
    proposition_traces.sort(
        key=lambda item: (-float(item["score"]), item["proposition_id"])
    )
    proposition_traces = proposition_traces[:24]
    selected_proposition_ids = {item["proposition_id"] for item in proposition_traces}
    rule_traces = [
        item
        for item in rule_traces
        if selected_proposition_ids.intersection(item["conclusions"])
        or item["status"] in {"fired", "blocked"}
    ][:48]
    conflicts = [
        {
            "source": str(edge.source_id),
            "target": str(edge.target_id),
            "type": edge.rel_type,
        }
        for edge in edges
        if edge.rel_type in {"ATTACKS", "CONTRADICTS"}
    ]
    proposition_tokens = set().union(
        *(_tokens(item["statement"]) for item in proposition_traces)
    )
    covered = sorted(query_tokens & (frame_tokens | proposition_tokens))
    operator = resolved_operator
    usable = [
        item
        for item in proposition_traces
        if item["status"] not in {"blocked", "incomplete"}
    ]
    if operator == "choose":
        operator_result: dict[str, Any] = {
            "applicable": [
                {
                    "proposition_id": item["proposition_id"],
                    "conditions": item["conditions"],
                    "exceptions": item["exceptions"],
                }
                for item in usable
            ],
            "blocked": [
                item["proposition_id"]
                for item in proposition_traces
                if item["status"] == "blocked"
            ],
        }
    elif operator == "evaluate":
        operator_result = {
            "supporting": [
                item["proposition_id"]
                for item in usable
                if item["polarity"] != "negative"
            ],
            "harm_or_limitation": [
                item["proposition_id"]
                for item in usable
                if item["polarity"] == "negative"
            ],
        }
    elif operator == "redesign":
        operator_result = {
            "constraints": [
                item["proposition_id"]
                for item in usable
                if item["conditions"] or item["polarity"] == "negative"
            ],
            "rewrite_candidates": [],
        }
    elif operator == "explain":
        operator_result = {
            "mechanism_chain_candidates": mechanism_edges,
            "supporting_propositions": [item["proposition_id"] for item in usable],
        }
    elif operator == "procedure":
        operator_result = {
            "procedure_evidence": [
                {
                    "proposition_id": item["proposition_id"],
                    "statement": item["statement"],
                    "conditions": item["conditions"],
                    "exceptions": item["exceptions"],
                }
                for item in usable
            ],
            "requested_outputs": resolved_outputs,
        }
    else:
        operator_result = {
            "support": [
                item["proposition_id"]
                for item in usable
                if item["polarity"] != "negative"
            ],
            "attack": [
                item["proposition_id"]
                for item in usable
                if item["polarity"] == "negative"
            ],
        }
    proved_ids = {
        match["proposition_id"]
        for goal in backward["goals"]
        for match in goal["matches"]
        if match["status"] in {"asserted", "proved"}
    }
    hypothesis_ids = {
        match["proposition_id"]
        for goal in backward["goals"]
        for match in goal["matches"]
        if match["status"] == "supported_hypothesis"
    } - proved_ids
    traces_by_id = {item["proposition_id"]: item for item in proposition_traces}

    def contract_claim(proposition_id: str) -> dict[str, Any]:
        trace = traces_by_id.get(proposition_id)
        if trace is not None:
            return {
                "proposition_id": proposition_id,
                "statement": trace["statement"],
                "status": trace["status"],
                "conditions": trace["conditions"],
                "exceptions": trace["exceptions"],
            }
        node = nodes.get(uuid.UUID(proposition_id))
        return {
            "proposition_id": proposition_id,
            "statement": (node.description or node.name) if node else "",
            "status": "unknown",
            "conditions": [],
            "exceptions": [],
        }

    answer_contract = {
        "status": backward["status"],
        "operator": operator,
        "requested_outputs": resolved_outputs,
        "aliases": resolved_aliases,
        "reasoning_plan": reasoning_plan,
        "proved_claims": [contract_claim(value) for value in sorted(proved_ids)],
        "supported_hypotheses": [
            contract_claim(value) for value in sorted(hypothesis_ids)
        ],
        "missing_bridges": backward["missing_bridges"],
        "policy": (
            "Only proved_claims may be stated as established. "
            "supported_hypotheses require explicit qualification. "
            "Missing bridges must not be silently inferred."
        ),
    }
    return {
        "operator": operator,
        "sufficiency": (
            "proved_graph_answer"
            if backward["status"] == "proved"
            else "hypothesis_only"
            if backward["status"] == "supported_hypothesis"
            else "insufficient_graph_evidence"
        ),
        "covered_query_tokens": covered,
        "propositions": proposition_traces,
        "rules": rule_traces,
        "conflicts": conflicts,
        "predicate_derivations": predicate_derivations,
        "goal_reasoning": backward,
        "answer_contract": answer_contract,
        "operator_result": operator_result,
        "fixed_point": True,
        "working_graph": {
            "nodes": len(nodes),
            "edges": len(edges),
            "search_hops": search_hops,
        },
        "timings_seconds": {"bounded_graph": bounded_graph_timings},
    }

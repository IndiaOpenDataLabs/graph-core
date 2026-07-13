"""Bounded deterministic activation over executable custom-graph structure."""

from __future__ import annotations

import math
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy import case, or_, select

from graph_core.database import AsyncSessionLocal
from graph_core.models.graph_rag import (
    EntityDescription,
    GraphEntity,
    GraphRelationship,
)
from graph_core.models.incremental_graph import (
    GraphCommunity,
    GraphCommunityMembership,
    GraphNodeMetric,
    GraphProjectionSnapshot,
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
    "take",
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
class ReasoningSeed:
    proposition_id: uuid.UUID
    frame_text: str
    argument_ids: tuple[uuid.UUID, ...]
    conditions: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    polarity: str = "positive"
    modality: str = "asserted"
    retrieval_score: float = 0.0


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


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{2,}", text.casefold())
        if token not in _STOPWORDS
    }


def _operator(question: str) -> str:
    lowered = question.casefold()
    if any(value in lowered for value in ("redesign", "refactor", "restructure")):
        return "redesign"
    if any(value in lowered for value in (" vs ", "versus", "when should")):
        return "choose"
    if any(value in lowered for value in ("good", "bad", "ugly", "evaluate")):
        return "evaluate"
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
) -> tuple[dict[uuid.UUID, _Node], list[_Edge]]:
    nodes = await _load_nodes(seed_ids)
    landmark_ids = await _community_landmarks(collection_id, set(nodes))
    nodes.update(await _load_nodes(landmark_ids))
    frontier = set(nodes)
    edges: dict[tuple[uuid.UUID, uuid.UUID, str], _Edge] = {}
    structural_first = case(
        (GraphRelationship.rel_type.in_(_STRUCTURAL_TYPES), 0), else_=1
    )
    for _ in range(max_hops):
        if not frontier or len(nodes) >= max_nodes or len(edges) >= max_edges:
            break
        async with AsyncSessionLocal() as session:
            rows = (
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
                        .limit(max_edges - len(edges))
                    )
                )
                .scalars()
                .all()
            )
        new_ids: set[uuid.UUID] = set()
        for row in rows:
            edge = _Edge(
                row.source_entity_id,
                row.target_entity_id,
                str(row.rel_type or "").upper(),
            )
            edges[(edge.source_id, edge.target_id, edge.rel_type)] = edge
            if edge.source_id not in nodes:
                new_ids.add(edge.source_id)
            if edge.target_id not in nodes:
                new_ids.add(edge.target_id)
        remaining = max_nodes - len(nodes)
        loaded = await _load_nodes(set(sorted(new_ids, key=str)[:remaining]))
        nodes.update(loaded)
        frontier = set(loaded)
    return nodes, [
        edge
        for edge in edges.values()
        if edge.source_id in nodes and edge.target_id in nodes
    ]


async def activate_reasoning(
    collection_id: uuid.UUID,
    question: str,
    seeds: list[ReasoningSeed],
    *,
    navigation_ids: set[uuid.UUID] | None = None,
) -> dict[str, Any]:
    if not seeds and not navigation_ids:
        return {
            "sufficiency": "insufficient_graph_evidence",
            "covered_query_tokens": [],
            "propositions": [],
            "rules": [],
            "conflicts": [],
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
    nodes, edges = await _bounded_graph(collection_id, seed_ids)
    incoming: dict[uuid.UUID, list[_Edge]] = defaultdict(list)
    outgoing: dict[uuid.UUID, list[_Edge]] = defaultdict(list)
    for edge in edges:
        incoming[edge.target_id].append(edge)
        outgoing[edge.source_id].append(edge)
    rules = {
        node_id for node_id, node in nodes.items() if node.node_type == "RULE"
    }
    active = set(seed_ids)
    for node_id, node in nodes.items():
        if node.node_type in {"CONDITION", "EXCEPTION"} and (
            query_tokens & _tokens(f"{node.name} {node.description}")
        ):
            active.add(node_id)
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
            edge.source_id
            for edge in incoming[rule_id]
            if edge.rel_type == "BLOCKS"
        }
        conclusions = {
            edge.target_id
            for edge in outgoing[rule_id]
            if edge.rel_type == "CONCLUDES"
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
                "conclusions": [
                    str(value) for value in sorted(conclusions, key=str)
                ],
            }
        )
    proposition_traces = []
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
    selected_proposition_ids = {
        item["proposition_id"] for item in proposition_traces
    }
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
    required_coverage = (
        len(query_tokens)
        if len(query_tokens) <= 3
        else max(2, math.ceil(len(query_tokens) * 0.6))
    )
    operator = _operator(question)
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
            "mechanism_chain_candidates": [
                item["proposition_id"] for item in usable
            ]
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
    return {
        "operator": operator,
        "sufficiency": (
            "sufficient_candidates"
            if len(covered) >= required_coverage and proposition_traces
            else "insufficient_graph_evidence"
        ),
        "covered_query_tokens": covered,
        "propositions": proposition_traces,
        "rules": rule_traces,
        "conflicts": conflicts,
        "operator_result": operator_result,
        "fixed_point": True,
        "working_graph": {"nodes": len(nodes), "edges": len(edges)},
    }

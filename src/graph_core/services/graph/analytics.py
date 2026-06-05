"""Offline analytics over the canonical collection graph.

This module builds a lightweight structural view over the merged graph:
- strong-edge communities from a weighted projection
- articulation / connector nodes that bridge graph regions
- bounded connector paths between top anchors

It is intentionally dependency-free for now.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import permutations
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import aliased

from graph_core.config import settings
from graph_core.database import AsyncSessionLocal
from graph_core.llm import LocalEchoLLMProvider
from graph_core.llm.interface import LLMProvider
from graph_core.models.collection import Collection
from graph_core.models.graph_rag import GraphEntity, GraphRelationship


@dataclass(slots=True)
class NodeRecord:
    id: uuid.UUID
    name: str


@dataclass(slots=True)
class RelationshipRecord:
    id: uuid.UUID
    source_id: uuid.UUID
    source_name: str
    target_id: uuid.UUID
    target_name: str
    rel_type: str
    weight: int


@dataclass(slots=True)
class EdgeAggregate:
    node_a: uuid.UUID
    node_b: uuid.UUID
    strength: float
    total_weight: int
    rel_types: tuple[str, ...]
    relationship_ids: tuple[str, ...]


@dataclass(slots=True)
class DirectedEdgeAggregate:
    source_id: uuid.UUID
    target_id: uuid.UUID
    strength: float
    total_weight: int
    rel_type: str
    relationship_ids: tuple[str, ...]


@dataclass(slots=True)
class NodeMetrics:
    node_id: uuid.UUID
    name: str
    community_id: int
    rel_type: str
    pagerank: float
    authority_score: float
    hub_score: float
    eigenvector_score: float
    betweenness: float
    closeness: float
    inbound_strength: float
    outbound_strength: float
    weighted_degree: float
    external_communities: tuple[int, ...]
    external_strength: float
    is_articulation: bool
    anchor_score: float


def _pair_key(left: uuid.UUID, right: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
    return (left, right) if left.hex <= right.hex else (right, left)


def _edge_strength(weight_sum: int, max_weight: int) -> float:
    cap = max(1, int(max_weight))
    return min(1.0, float(weight_sum) / float(cap))


def derived_graph_name(collection_id: uuid.UUID) -> str:
    return f"collection_{str(collection_id).replace('-', '')}_derived"


def _connected_components(
    node_ids: list[uuid.UUID],
    adjacency: dict[uuid.UUID, dict[uuid.UUID, EdgeAggregate]],
) -> list[list[uuid.UUID]]:
    seen: set[uuid.UUID] = set()
    components: list[list[uuid.UUID]] = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        queue: deque[uuid.UUID] = deque([node_id])
        seen.add(node_id)
        component: list[uuid.UUID] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency.get(current, {}):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    return components


def _merge_small_communities(
    components: list[list[uuid.UUID]],
    full_adjacency: dict[uuid.UUID, dict[uuid.UUID, EdgeAggregate]],
    *,
    min_size: int,
) -> list[list[uuid.UUID]]:
    if min_size <= 1 or not components:
        return components

    community_index: dict[uuid.UUID, int] = {}
    for idx, nodes in enumerate(components):
        for node_id in nodes:
            community_index[node_id] = idx

    merged = [list(nodes) for nodes in components]
    changed = True
    while changed:
        changed = False
        for idx, nodes in list(enumerate(merged)):
            if not nodes or len(nodes) >= min_size:
                continue
            neighbor_strengths: dict[int, float] = defaultdict(float)
            for node_id in nodes:
                for neighbor_id, edge in full_adjacency.get(node_id, {}).items():
                    other_idx = community_index.get(neighbor_id)
                    if other_idx is None or other_idx == idx:
                        continue
                    neighbor_strengths[other_idx] += edge.strength
            if not neighbor_strengths:
                continue
            target_idx = max(neighbor_strengths, key=neighbor_strengths.get)
            merged[target_idx].extend(nodes)
            for node_id in nodes:
                community_index[node_id] = target_idx
            merged[idx] = []
            changed = True

    return [nodes for nodes in merged if nodes]


def _articulation_points(
    adjacency: dict[uuid.UUID, dict[uuid.UUID, EdgeAggregate]],
) -> set[uuid.UUID]:
    discovery: dict[uuid.UUID, int] = {}
    low: dict[uuid.UUID, int] = {}
    parent: dict[uuid.UUID, uuid.UUID | None] = {}
    articulation: set[uuid.UUID] = set()
    time = 0

    def dfs(node_id: uuid.UUID) -> None:
        nonlocal time
        time += 1
        discovery[node_id] = time
        low[node_id] = time
        child_count = 0
        for neighbor_id in adjacency.get(node_id, {}):
            if neighbor_id not in discovery:
                parent[neighbor_id] = node_id
                child_count += 1
                dfs(neighbor_id)
                low[node_id] = min(low[node_id], low[neighbor_id])
                if parent.get(node_id) is None and child_count > 1:
                    articulation.add(node_id)
                if (
                    parent.get(node_id) is not None
                    and low[neighbor_id] >= discovery[node_id]
                ):
                    articulation.add(node_id)
            elif neighbor_id != parent.get(node_id):
                low[node_id] = min(low[node_id], discovery[neighbor_id])

    for node_id in adjacency:
        if node_id in discovery:
            continue
        parent[node_id] = None
        dfs(node_id)

    return articulation


def _shortest_directed_path(
    start_id: uuid.UUID,
    end_id: uuid.UUID,
    adjacency: dict[uuid.UUID, dict[uuid.UUID, DirectedEdgeAggregate]],
    *,
    max_depth: int,
) -> list[uuid.UUID] | None:
    if start_id == end_id:
        return [start_id]

    queue: deque[tuple[uuid.UUID, list[uuid.UUID]]] = deque([(start_id, [start_id])])
    seen: set[uuid.UUID] = {start_id}
    while queue:
        current, path = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        neighbors = sorted(
            adjacency.get(current, {}).items(),
            key=lambda item: item[1].strength,
            reverse=True,
        )
        for neighbor_id, _edge in neighbors:
            if neighbor_id == end_id:
                return [*path, neighbor_id]
            if neighbor_id in seen:
                continue
            seen.add(neighbor_id)
            queue.append((neighbor_id, [*path, neighbor_id]))
    return None


def _top_rel_types(relationships: list[RelationshipRecord]) -> list[str]:
    return sorted({rel.rel_type or "RELATES_TO" for rel in relationships})


def _power_pagerank(
    nodes: list[uuid.UUID],
    out_edges: dict[uuid.UUID, list[tuple[uuid.UUID, float]]],
    in_edges: dict[uuid.UUID, list[tuple[uuid.UUID, float]]],
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> dict[uuid.UUID, float]:
    count = len(nodes)
    if count == 0:
        return {}
    pr = {node_id: 1.0 / count for node_id in nodes}
    out_weight = {
        node_id: sum(weight for _target, weight in out_edges.get(node_id, []))
        for node_id in nodes
    }
    for _ in range(max_iter):
        new = {node_id: (1.0 - alpha) / count for node_id in nodes}
        sink = sum(pr[node_id] for node_id in nodes if out_weight.get(node_id, 0.0) == 0.0)
        sink_share = alpha * sink / count
        for node_id in nodes:
            new[node_id] += sink_share
        for node_id in nodes:
            for source_id, weight in in_edges.get(node_id, []):
                denom = out_weight.get(source_id, 0.0)
                if denom > 0:
                    new[node_id] += alpha * pr[source_id] * (weight / denom)
        error = sum(abs(new[node_id] - pr[node_id]) for node_id in nodes)
        pr = new
        if error < tol:
            break
    return pr


def _power_hits(
    nodes: list[uuid.UUID],
    out_edges: dict[uuid.UUID, list[tuple[uuid.UUID, float]]],
    in_edges: dict[uuid.UUID, list[tuple[uuid.UUID, float]]],
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[dict[uuid.UUID, float], dict[uuid.UUID, float]]:
    authority = {node_id: 1.0 for node_id in nodes}
    hub = {node_id: 1.0 for node_id in nodes}
    for _ in range(max_iter):
        new_authority = {
            node_id: sum(hub[src] * weight for src, weight in in_edges.get(node_id, []))
            for node_id in nodes
        }
        norm = sum(value * value for value in new_authority.values()) ** 0.5 or 1.0
        for node_id in nodes:
            new_authority[node_id] /= norm
        new_hub = {
            node_id: sum(
                new_authority[target] * weight
                for target, weight in out_edges.get(node_id, [])
            )
            for node_id in nodes
        }
        norm = sum(value * value for value in new_hub.values()) ** 0.5 or 1.0
        for node_id in nodes:
            new_hub[node_id] /= norm
        error = sum(
            abs(new_authority[node_id] - authority[node_id])
            + abs(new_hub[node_id] - hub[node_id])
            for node_id in nodes
        )
        authority, hub = new_authority, new_hub
        if error < tol:
            break
    return authority, hub


def _power_eigenvector_undirected(
    nodes: list[uuid.UUID],
    neighbors: dict[uuid.UUID, list[tuple[uuid.UUID, float]]],
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> dict[uuid.UUID, float]:
    scores = {node_id: 1.0 for node_id in nodes}
    for _ in range(max_iter):
        new = {
            node_id: sum(scores[other_id] * weight for other_id, weight in neighbors.get(node_id, []))
            for node_id in nodes
        }
        norm = sum(value * value for value in new.values()) ** 0.5 or 1.0
        for node_id in nodes:
            new[node_id] /= norm
        error = sum(abs(new[node_id] - scores[node_id]) for node_id in nodes)
        scores = new
        if error < tol:
            break
    return scores


def _shortest_paths_unweighted(
    adjacency: dict[uuid.UUID, list[uuid.UUID]],
    source_id: uuid.UUID,
) -> tuple[
    list[uuid.UUID],
    dict[uuid.UUID, list[uuid.UUID]],
    dict[uuid.UUID, float],
    dict[uuid.UUID, int],
]:
    distance = {source_id: 0}
    sigma: dict[uuid.UUID, float] = defaultdict(float)
    sigma[source_id] = 1.0
    predecessors: dict[uuid.UUID, list[uuid.UUID]] = defaultdict(list)
    queue: deque[uuid.UUID] = deque([source_id])
    order: list[uuid.UUID] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for neighbor_id in adjacency.get(node_id, []):
            if neighbor_id not in distance:
                distance[neighbor_id] = distance[node_id] + 1
                queue.append(neighbor_id)
            if distance[neighbor_id] == distance[node_id] + 1:
                sigma[neighbor_id] += sigma[node_id]
                predecessors[neighbor_id].append(node_id)
    return order, predecessors, sigma, distance


def _betweenness_directed(
    nodes: list[uuid.UUID],
    adjacency: dict[uuid.UUID, list[uuid.UUID]],
) -> dict[uuid.UUID, float]:
    scores = {node_id: 0.0 for node_id in nodes}
    for source_id in nodes:
        order, predecessors, sigma, _distance = _shortest_paths_unweighted(
            adjacency,
            source_id,
        )
        delta = {node_id: 0.0 for node_id in nodes}
        for node_id in reversed(order):
            for predecessor_id in predecessors[node_id]:
                if sigma[node_id]:
                    delta[predecessor_id] += (
                        sigma[predecessor_id] / sigma[node_id]
                    ) * (1.0 + delta[node_id])
            if node_id != source_id:
                scores[node_id] += delta[node_id]
    count = len(nodes)
    if count > 2:
        scale = 1.0 / ((count - 1) * (count - 2))
        for node_id in scores:
            scores[node_id] *= scale
    return scores


def _harmonic_closeness(
    nodes: list[uuid.UUID],
    adjacency: dict[uuid.UUID, list[uuid.UUID]],
) -> dict[uuid.UUID, float]:
    scores: dict[uuid.UUID, float] = {}
    for source_id in nodes:
        _order, _predecessors, _sigma, distance = _shortest_paths_unweighted(
            adjacency,
            source_id,
        )
        scores[source_id] = sum(
            1.0 / dist
            for node_id, dist in distance.items()
            if node_id != source_id and dist > 0
        )
    return scores


def _normalize_scores(scores: dict[uuid.UUID, float]) -> dict[uuid.UUID, float]:
    if not scores:
        return {}
    max_value = max(scores.values()) or 0.0
    if max_value <= 0.0:
        return {node_id: 0.0 for node_id in scores}
    return {node_id: float(value) / float(max_value) for node_id, value in scores.items()}


async def _load_graph_records(
    collection_id: uuid.UUID,
) -> tuple[Collection, list[NodeRecord], list[RelationshipRecord]]:
    async with AsyncSessionLocal() as session:
        collection = await session.get(Collection, collection_id)
        if not collection:
            raise ValueError(f"Collection {collection_id} not found")

        nodes = (
            await session.execute(
                select(GraphEntity.id, GraphEntity.canonical_name).where(
                    GraphEntity.collection_id == collection_id
                )
            )
        ).all()

        source_entity = aliased(GraphEntity)
        target_entity = aliased(GraphEntity)
        relationships = (
            await session.execute(
                select(
                    GraphRelationship.id,
                    GraphRelationship.source_entity_id,
                    source_entity.canonical_name,
                    GraphRelationship.target_entity_id,
                    target_entity.canonical_name,
                    GraphRelationship.rel_type,
                    GraphRelationship.weight,
                )
                .join(
                    source_entity,
                    source_entity.id == GraphRelationship.source_entity_id,
                )
                .join(
                    target_entity,
                    target_entity.id == GraphRelationship.target_entity_id,
                )
                .where(GraphRelationship.collection_id == collection_id)
            )
        ).all()

    return (
        collection,
        [NodeRecord(id=node_id, name=name) for node_id, name in nodes],
        [
            RelationshipRecord(
                id=rel_id,
                source_id=source_id,
                source_name=source_name,
                target_id=target_id,
                target_name=target_name,
                rel_type=rel_type,
                weight=int(weight or 0),
            )
            for (
                rel_id,
                source_id,
                source_name,
                target_id,
                target_name,
                rel_type,
                weight,
            ) in relationships
        ],
    )


def _build_rel_type_analysis(
    nodes: list[NodeRecord],
    relationships: list[RelationshipRecord],
    *,
    rel_type: str,
    min_edge_strength: float = 0.2,
    min_community_size: int = 2,
    max_anchors: int = 12,
    max_path_depth: int = 4,
    max_connector_paths: int = 20,
    max_relationship_weight: int | None = None,
) -> dict[str, Any]:
    max_weight = max_relationship_weight or settings.graph_rag_max_relationship_weight
    node_names = {node.id: node.name for node in nodes}
    rels = [rel for rel in relationships if (rel.rel_type or "RELATES_TO") == rel_type]
    if not rels:
        return {
            "rel_type": rel_type,
            "parameters": {
                "min_edge_strength": min_edge_strength,
                "min_community_size": min_community_size,
                "max_anchors": max_anchors,
                "max_path_depth": max_path_depth,
                "max_connector_paths": max_connector_paths,
                "max_relationship_weight": max_weight,
            },
            "totals": {
                "entities": 0,
                "relationships": 0,
                "directed_pairs": 0,
                "undirected_pairs": 0,
                "strong_pairs": 0,
                "communities": 0,
            },
            "node_metrics": [],
            "communities": [],
            "top_anchors": [],
            "bridge_nodes": [],
            "connector_paths": [],
        }

    by_pair: dict[tuple[uuid.UUID, uuid.UUID], dict[str, Any]] = {}
    by_direction: dict[tuple[uuid.UUID, uuid.UUID], dict[str, Any]] = {}
    for rel in rels:
        pair = _pair_key(rel.source_id, rel.target_id)
        bucket = by_pair.setdefault(
            pair,
            {
                "total_weight": 0,
                "rel_types": set(),
                "relationship_ids": [],
            },
        )
        bucket["total_weight"] += max(0, int(rel.weight))
        bucket["rel_types"].add(rel.rel_type)
        bucket["relationship_ids"].append(str(rel.id))
        directed = by_direction.setdefault(
            (rel.source_id, rel.target_id),
            {
                "total_weight": 0,
                "relationship_ids": [],
            },
        )
        directed["total_weight"] += max(0, int(rel.weight))
        directed["relationship_ids"].append(str(rel.id))

    full_adjacency: dict[uuid.UUID, dict[uuid.UUID, EdgeAggregate]] = defaultdict(dict)
    directed_adjacency: dict[
        uuid.UUID, dict[uuid.UUID, DirectedEdgeAggregate]
    ] = defaultdict(dict)
    all_edges: list[EdgeAggregate] = []
    for (left_id, right_id), bucket in by_pair.items():
        aggregate = EdgeAggregate(
            node_a=left_id,
            node_b=right_id,
            strength=_edge_strength(bucket["total_weight"], max_weight),
            total_weight=int(bucket["total_weight"]),
            rel_types=tuple(sorted(bucket["rel_types"])),
            relationship_ids=tuple(sorted(bucket["relationship_ids"])),
        )
        full_adjacency[left_id][right_id] = aggregate
        full_adjacency[right_id][left_id] = aggregate
        all_edges.append(aggregate)
    for (source_id, target_id), bucket in by_direction.items():
        directed_adjacency[source_id][target_id] = DirectedEdgeAggregate(
            source_id=source_id,
            target_id=target_id,
            strength=_edge_strength(bucket["total_weight"], max_weight),
            total_weight=int(bucket["total_weight"]),
            rel_type=rel_type,
            relationship_ids=tuple(sorted(bucket["relationship_ids"])),
        )

    strong_adjacency: dict[
        uuid.UUID, dict[uuid.UUID, EdgeAggregate]
    ] = defaultdict(dict)
    for edge in all_edges:
        if edge.strength < min_edge_strength:
            continue
        strong_adjacency[edge.node_a][edge.node_b] = edge
        strong_adjacency[edge.node_b][edge.node_a] = edge

    active_node_ids = sorted(
        {rel.source_id for rel in rels} | {rel.target_id for rel in rels},
        key=lambda node_id: node_id.hex,
    )
    active_nodes = [node for node in nodes if node.id in set(active_node_ids)]
    node_ids = active_node_ids
    out_edges_weighted: dict[uuid.UUID, list[tuple[uuid.UUID, float]]] = defaultdict(list)
    in_edges_weighted: dict[uuid.UUID, list[tuple[uuid.UUID, float]]] = defaultdict(list)
    directed_neighbors: dict[uuid.UUID, list[uuid.UUID]] = defaultdict(list)
    undirected_neighbors_weighted: dict[
        uuid.UUID, list[tuple[uuid.UUID, float]]
    ] = defaultdict(list)
    for source_id, targets in directed_adjacency.items():
        for target_id, edge in targets.items():
            weight = max(edge.strength, 1e-6)
            out_edges_weighted[source_id].append((target_id, weight))
            in_edges_weighted[target_id].append((source_id, weight))
            directed_neighbors[source_id].append(target_id)
    for node_id, neighbors in full_adjacency.items():
        for neighbor_id, edge in neighbors.items():
            undirected_neighbors_weighted[node_id].append(
                (neighbor_id, max(edge.strength, 1e-6))
            )

    pagerank = _power_pagerank(node_ids, out_edges_weighted, in_edges_weighted)
    authority_score, hub_score = _power_hits(
        node_ids,
        out_edges_weighted,
        in_edges_weighted,
    )
    eigenvector_score = _power_eigenvector_undirected(
        node_ids,
        undirected_neighbors_weighted,
    )
    betweenness = _betweenness_directed(node_ids, directed_neighbors)
    closeness = _harmonic_closeness(node_ids, directed_neighbors)
    closeness_norm = _normalize_scores(closeness)

    communities = _connected_components(node_ids, strong_adjacency)
    communities = _merge_small_communities(
        communities,
        full_adjacency,
        min_size=min_community_size,
    )

    community_of: dict[uuid.UUID, int] = {}
    for idx, members in enumerate(communities):
        for node_id in members:
            community_of[node_id] = idx

    articulation = _articulation_points(strong_adjacency)

    metrics: list[NodeMetrics] = []
    for node in active_nodes:
        outbound_strength = sum(
            edge.strength for edge in directed_adjacency.get(node.id, {}).values()
        )
        inbound_strength = sum(
            edge.strength
            for neighbors in directed_adjacency.values()
            for target_id, edge in neighbors.items()
            if target_id == node.id
        )
        weighted_degree = sum(
            edge.strength for edge in full_adjacency.get(node.id, {}).values()
        )
        external_strength = 0.0
        external_communities: set[int] = set()
        for neighbor_id, edge in full_adjacency.get(node.id, {}).items():
            neighbor_community = community_of.get(neighbor_id, -1)
            if neighbor_community != community_of.get(node.id, -1):
                external_communities.add(neighbor_community)
                external_strength += edge.strength
        anchor_score = (
            (2.0 * pagerank.get(node.id, 0.0))
            + (2.0 * authority_score.get(node.id, 0.0))
            + (2.0 * hub_score.get(node.id, 0.0))
            + (1.5 * eigenvector_score.get(node.id, 0.0))
            + (6.0 * betweenness.get(node.id, 0.0))
            + (2.5 * closeness_norm.get(node.id, 0.0))
            + outbound_strength
            + inbound_strength
            + weighted_degree
            + (2.0 * len(external_communities))
            + (1.5 * external_strength)
            + (3.0 if node.id in articulation else 0.0)
        )
        metrics.append(
            NodeMetrics(
                node_id=node.id,
                name=node.name,
                community_id=community_of.get(node.id, -1),
                rel_type=rel_type,
                pagerank=round(pagerank.get(node.id, 0.0), 6),
                authority_score=round(authority_score.get(node.id, 0.0), 6),
                hub_score=round(hub_score.get(node.id, 0.0), 6),
                eigenvector_score=round(eigenvector_score.get(node.id, 0.0), 6),
                betweenness=round(betweenness.get(node.id, 0.0), 6),
                closeness=round(closeness_norm.get(node.id, 0.0), 6),
                inbound_strength=round(inbound_strength, 4),
                outbound_strength=round(outbound_strength, 4),
                weighted_degree=round(weighted_degree, 4),
                external_communities=tuple(sorted(external_communities)),
                external_strength=round(external_strength, 4),
                is_articulation=node.id in articulation,
                anchor_score=round(anchor_score, 4),
            )
        )

    top_anchors = sorted(
        metrics,
        key=lambda item: (
            item.anchor_score,
            item.betweenness,
            item.closeness,
            item.hub_score + item.authority_score,
            item.outbound_strength + item.inbound_strength,
            item.weighted_degree,
            len(item.external_communities),
            item.name,
        ),
        reverse=True,
    )[:max_anchors]

    connector_paths: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, ...]] = set()
    for left_anchor, right_anchor in permutations(top_anchors, 2):
        if len(connector_paths) >= max_connector_paths:
            break
        path = _shortest_directed_path(
            left_anchor.node_id,
            right_anchor.node_id,
            directed_adjacency,
            max_depth=max_path_depth,
        )
        if not path or len(path) < 2:
            continue
        path_names = tuple(node_names[node_id] for node_id in path)
        if path_names in seen_paths:
            continue
        seen_paths.add(path_names)
        hops: list[dict[str, Any]] = []
        path_score = 0.0
        for current_id, next_id in zip(path, path[1:], strict=False):
            edge = directed_adjacency[current_id][next_id]
            path_score += edge.strength
            hops.append(
                {
                    "source_id": str(current_id),
                    "source": node_names[current_id],
                    "target_id": str(next_id),
                    "target": node_names[next_id],
                    "strength": round(edge.strength, 4),
                    "rel_types": [edge.rel_type],
                    "relationship_ids": list(edge.relationship_ids),
                }
            )
        connector_paths.append(
            {
                "rel_type": rel_type,
                "from_anchor": left_anchor.name,
                "to_anchor": right_anchor.name,
                "source_community": left_anchor.community_id,
                "target_community": right_anchor.community_id,
                "nodes": list(path_names),
                "hop_count": len(path) - 1,
                "path_score": round(path_score, 4),
                "hops": hops,
            }
        )

    connector_paths.sort(
        key=lambda item: (
            item["source_community"] != item["target_community"],
            item["path_score"],
            -item["hop_count"],
        ),
        reverse=True,
    )

    community_summaries: list[dict[str, Any]] = []
    for idx, members in enumerate(communities):
        member_metrics = [metric for metric in metrics if metric.community_id == idx]
        internal_edges: list[EdgeAggregate] = []
        member_set = set(members)
        for node_id in members:
            for neighbor_id, edge in strong_adjacency.get(node_id, {}).items():
                if neighbor_id not in member_set or node_id.hex > neighbor_id.hex:
                    continue
                internal_edges.append(edge)
        anchor_preview = [
            metric.name
            for metric in sorted(
                member_metrics,
                key=lambda metric: metric.anchor_score,
                reverse=True,
            )[:3]
        ]
        community_summaries.append(
            {
                "rel_type": rel_type,
                "community_id": idx,
                "size": len(members),
                "node_ids": [str(node_id) for node_id in members],
                "node_names": sorted(node_names[node_id] for node_id in members),
                "strong_edge_count": len(internal_edges),
                "average_strong_edge_strength": round(
                    (
                        sum(edge.strength for edge in internal_edges)
                        / len(internal_edges)
                    )
                    if internal_edges
                    else 0.0,
                    4,
                ),
                "anchor_preview": anchor_preview,
            }
        )

    return {
        "rel_type": rel_type,
        "parameters": {
            "min_edge_strength": min_edge_strength,
            "min_community_size": min_community_size,
            "max_anchors": max_anchors,
            "max_path_depth": max_path_depth,
            "max_connector_paths": max_connector_paths,
            "max_relationship_weight": max_weight,
        },
        "totals": {
            "entities": len(active_nodes),
            "relationships": len(rels),
            "directed_pairs": len(by_direction),
            "undirected_pairs": len(all_edges),
            "strong_pairs": sum(
                1 for edge in all_edges if edge.strength >= min_edge_strength
            ),
            "communities": len(communities),
        },
        "node_metrics": [
            {
                "node_id": str(metric.node_id),
                "name": metric.name,
                "community_id": metric.community_id,
                "rel_type": metric.rel_type,
                "anchor_score": metric.anchor_score,
                "pagerank": metric.pagerank,
                "authority_score": metric.authority_score,
                "hub_score": metric.hub_score,
                "eigenvector_score": metric.eigenvector_score,
                "betweenness": metric.betweenness,
                "closeness": metric.closeness,
                "inbound_strength": metric.inbound_strength,
                "outbound_strength": metric.outbound_strength,
                "weighted_degree": metric.weighted_degree,
                "external_communities": list(metric.external_communities),
                "external_strength": metric.external_strength,
                "is_articulation": metric.is_articulation,
            }
            for metric in metrics
        ],
        "communities": community_summaries,
        "top_anchors": [
            {
                "node_id": str(metric.node_id),
                "name": metric.name,
                "community_id": metric.community_id,
                "rel_type": metric.rel_type,
                "anchor_score": metric.anchor_score,
                "pagerank": metric.pagerank,
                "authority_score": metric.authority_score,
                "hub_score": metric.hub_score,
                "eigenvector_score": metric.eigenvector_score,
                "betweenness": metric.betweenness,
                "closeness": metric.closeness,
                "inbound_strength": metric.inbound_strength,
                "outbound_strength": metric.outbound_strength,
                "weighted_degree": metric.weighted_degree,
                "external_communities": list(metric.external_communities),
                "external_strength": metric.external_strength,
                "is_articulation": metric.is_articulation,
            }
            for metric in top_anchors
        ],
        "bridge_nodes": [
            {
                "node_id": str(metric.node_id),
                "name": metric.name,
                "community_id": metric.community_id,
                "rel_type": metric.rel_type,
                "anchor_score": metric.anchor_score,
                "pagerank": metric.pagerank,
                "authority_score": metric.authority_score,
                "hub_score": metric.hub_score,
                "eigenvector_score": metric.eigenvector_score,
                "betweenness": metric.betweenness,
                "closeness": metric.closeness,
                "inbound_strength": metric.inbound_strength,
                "outbound_strength": metric.outbound_strength,
                "weighted_degree": metric.weighted_degree,
                "external_communities": list(metric.external_communities),
                "external_strength": metric.external_strength,
                "is_articulation": metric.is_articulation,
            }
            for metric in sorted(
                (
                    metric
                    for metric in metrics
                    if metric.is_articulation or metric.external_communities
                ),
                key=lambda metric: (
                    metric.is_articulation,
                    len(metric.external_communities),
                    metric.anchor_score,
                ),
                reverse=True,
            )[: max(10, max_anchors)]
        ],
        "connector_paths": connector_paths[:max_connector_paths],
    }


def build_collection_analysis(
    nodes: list[NodeRecord],
    relationships: list[RelationshipRecord],
    *,
    min_edge_strength: float = 0.2,
    min_community_size: int = 2,
    max_anchors: int = 12,
    max_path_depth: int = 4,
    max_connector_paths: int = 20,
    max_relationship_weight: int | None = None,
) -> dict[str, Any]:
    rel_types = _top_rel_types(relationships)
    analyses = [
        _build_rel_type_analysis(
            nodes,
            relationships,
            rel_type=rel_type,
            min_edge_strength=min_edge_strength,
            min_community_size=min_community_size,
            max_anchors=max_anchors,
            max_path_depth=max_path_depth,
            max_connector_paths=max_connector_paths,
            max_relationship_weight=max_relationship_weight,
        )
        for rel_type in rel_types
    ]

    anchor_totals: dict[str, dict[str, Any]] = {}
    bridge_totals: dict[str, dict[str, Any]] = {}
    communities: list[dict[str, Any]] = []
    connector_paths: list[dict[str, Any]] = []
    total_strong_pairs = 0
    total_directed_pairs = 0
    total_undirected_pairs = 0
    for analysis in analyses:
        total_strong_pairs += int(analysis["totals"]["strong_pairs"])
        total_directed_pairs += int(analysis["totals"]["directed_pairs"])
        total_undirected_pairs += int(analysis["totals"]["undirected_pairs"])
        communities.extend(analysis["communities"])
        connector_paths.extend(analysis["connector_paths"])
        for bucket, items in (
            (anchor_totals, analysis["top_anchors"]),
            (bridge_totals, analysis["bridge_nodes"]),
        ):
            for item in items:
                entry = bucket.setdefault(
                    item["node_id"],
                    {
                        **item,
                        "rel_types": [],
                        "anchor_score": 0.0,
                        "pagerank": 0.0,
                        "authority_score": 0.0,
                        "hub_score": 0.0,
                        "eigenvector_score": 0.0,
                        "betweenness": 0.0,
                        "closeness": 0.0,
                        "inbound_strength": 0.0,
                        "outbound_strength": 0.0,
                        "weighted_degree": 0.0,
                        "external_strength": 0.0,
                    },
                )
                entry["rel_types"].append(item["rel_type"])
                entry["anchor_score"] += float(item["anchor_score"])
                entry["pagerank"] += float(item.get("pagerank", 0.0))
                entry["authority_score"] += float(item.get("authority_score", 0.0))
                entry["hub_score"] += float(item.get("hub_score", 0.0))
                entry["eigenvector_score"] += float(item.get("eigenvector_score", 0.0))
                entry["betweenness"] += float(item.get("betweenness", 0.0))
                entry["closeness"] += float(item.get("closeness", 0.0))
                entry["inbound_strength"] += float(item["inbound_strength"])
                entry["outbound_strength"] += float(item["outbound_strength"])
                entry["weighted_degree"] += float(item["weighted_degree"])
                entry["external_strength"] += float(item["external_strength"])
                entry["is_articulation"] = (
                    bool(entry.get("is_articulation")) or bool(item["is_articulation"])
                )
                entry["external_communities"] = sorted(
                    set(entry.get("external_communities", []))
                    | set(item.get("external_communities", []))
                )

    def _finalize_entries(items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = list(items.values())
        for row in rows:
            row["rel_types"] = sorted(set(row["rel_types"]))
            for key in (
                "anchor_score",
                "pagerank",
                "authority_score",
                "hub_score",
                "eigenvector_score",
                "betweenness",
                "closeness",
                "inbound_strength",
                "outbound_strength",
                "weighted_degree",
                "external_strength",
            ):
                row[key] = round(float(row[key]), 4)
        rows.sort(
            key=lambda item: (
                item["anchor_score"],
                item["betweenness"],
                item["closeness"],
                item["hub_score"] + item["authority_score"],
                item["outbound_strength"] + item["inbound_strength"],
                item["weighted_degree"],
                len(item["rel_types"]),
                item["name"],
            ),
            reverse=True,
        )
        return rows

    communities.sort(
        key=lambda item: (
            item["strong_edge_count"],
            item["average_strong_edge_strength"],
            item["size"],
        ),
        reverse=True,
    )
    connector_paths.sort(
        key=lambda item: (
            item["source_community"] != item["target_community"],
            item["path_score"],
            -item["hop_count"],
        ),
        reverse=True,
    )

    return {
        "parameters": {
            "min_edge_strength": min_edge_strength,
            "min_community_size": min_community_size,
            "max_anchors": max_anchors,
            "max_path_depth": max_path_depth,
            "max_connector_paths": max_connector_paths,
            "max_relationship_weight": (
                max_relationship_weight or settings.graph_rag_max_relationship_weight
            ),
        },
        "totals": {
            "entities": len(nodes),
            "relationships": len(relationships),
            "directed_pairs": total_directed_pairs,
            "undirected_pairs": total_undirected_pairs,
            "strong_pairs": total_strong_pairs,
            "communities": len(communities),
            "rel_types": len(rel_types),
        },
        "rel_type_analyses": analyses,
        "communities": communities,
        "top_anchors": _finalize_entries(anchor_totals)[:max_anchors],
        "bridge_nodes": _finalize_entries(bridge_totals)[: max(10, max_anchors)],
        "connector_paths": connector_paths[:max_connector_paths],
    }


async def analyze_collection_graph(
    collection_id: uuid.UUID,
    *,
    min_edge_strength: float = 0.2,
    min_community_size: int = 2,
    max_anchors: int = 12,
    max_path_depth: int = 4,
    max_connector_paths: int = 20,
) -> dict[str, Any]:
    collection, nodes, relationships = await _load_graph_records(collection_id)
    analysis = build_collection_analysis(
        nodes,
        relationships,
        min_edge_strength=min_edge_strength,
        min_community_size=min_community_size,
        max_anchors=max_anchors,
        max_path_depth=max_path_depth,
        max_connector_paths=max_connector_paths,
    )
    analysis["collection"] = {
        "id": str(collection.id),
        "name": collection.name,
        "namespace_id": str(collection.namespace_id),
        "strategy": str(collection.strategy),
    }
    return analysis


async def build_collection_understanding(
    analysis: dict[str, Any],
    llm_provider: LLMProvider | None = None,
) -> dict[str, list[dict[str, Any]]]:
    collection = analysis["collection"]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    seen_entity_refs: set[str] = set()
    semantic_edge_types = {
        "SUPPORTS",
        "JUSTIFIES",
        "QUALIFIES",
        "LIMITS",
        "DISTINGUISHES",
        "CONTRADICTS",
        "CAUSES",
        "IMPLIES",
        "DEPENDS_ON",
    }

    def derived_chunk_hash(*parts: object) -> str:
        raw = "::".join(str(part) for part in parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def entity_ref_node(name: str) -> str:
        normalized = "_".join(name.strip().lower().split())
        return f"derived:entity:{normalized}"

    def ensure_entity_ref(
        name: str,
        *,
        supporting_ids: list[str] | None = None,
    ) -> str:
        node_id = entity_ref_node(name)
        if node_id in seen_entity_refs:
            return node_id
        seen_entity_refs.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "name": name,
                "collection_id": collection["id"],
                "type": "base_entity_ref",
                "description": f"Reference to base graph entity: {name}.",
                "source_ids": supporting_ids or [],
            }
        )
        return node_id

    chunk_index = 0

    def next_chunk_index() -> int:
        nonlocal chunk_index
        current = chunk_index
        chunk_index += 1
        return current

    def slug(text: str) -> str:
        return "_".join(part for part in text.strip().lower().split() if part)[:96]

    def add_edge(
        source_id: str,
        target_id: str,
        *,
        rel_type: str,
        description: str,
        source_ids: list[str] | None = None,
    ) -> None:
        edge_id = f"{source_id}__{rel_type}__{target_id}"
        if any(edge["id"] == edge_id for edge in edges):
            return
        edges.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "id": edge_id,
                "collection_id": collection["id"],
                "rel_type": rel_type,
                "description": description,
                "source_ids": source_ids or [],
            }
        )

    def normalize_map(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_value = max(scores.values()) or 0.0
        if max_value <= 0.0:
            return {key: 0.0 for key in scores}
        return {key: float(value) / float(max_value) for key, value in scores.items()}

    candidate_regions: list[dict[str, Any]] = []
    for idx, bridge in enumerate(analysis.get("bridge_nodes", [])[:8], start=1):
        rel_types = list(bridge.get("rel_types") or [])
        candidate_regions.append(
            {
                "region_id": f"bridge_{idx}",
                "kind": "bridge",
                "title": bridge["name"],
                "description": (
                    f"{bridge['name']} bridges relation types {', '.join(rel_types) or 'unknown'} "
                    f"with betweenness {bridge['betweenness']} and closeness {bridge['closeness']}. "
                    f"It touches external communities {bridge['external_communities']}."
                ),
                "source_ids": [str(bridge.get("node_id"))] if bridge.get("node_id") else [],
                "entity_names": [bridge["name"]],
                "rel_types": rel_types,
            }
        )

    for idx, path in enumerate(analysis.get("connector_paths", [])[:8], start=1):
        path_nodes = [str(node) for node in path.get("nodes", []) if str(node).strip()]
        supporting_rel_ids = [
            str(rel_id)
            for hop in path.get("hops", [])
            for rel_id in hop.get("relationship_ids", [])
        ]
        candidate_regions.append(
            {
                "region_id": f"path_{idx}",
                "kind": "path",
                "title": f"{path.get('from_anchor', '?')} -> {path.get('to_anchor', '?')}",
                "description": (
                    f"{path.get('rel_type', 'RELATES_TO')} path across {path.get('hop_count', 0)} hops: "
                    f"{' -> '.join(path_nodes[:8])}. Path score {path.get('path_score', 0.0)}."
                ),
                "source_ids": supporting_rel_ids,
                "entity_names": path_nodes[:12],
                "rel_types": [str(path.get("rel_type") or "RELATES_TO")],
            }
        )

    rel_type_strengths: dict[str, float] = {}
    for rel_analysis in analysis.get("rel_type_analyses", []):
        rel_type = str(rel_analysis.get("rel_type") or "RELATES_TO")
        rows = rel_analysis.get("node_metrics", [])
        rel_type_strengths[rel_type] = sum(
            (
                float(row.get("pagerank", 0.0))
                + float(row.get("hub_score", 0.0))
                + float(row.get("authority_score", 0.0))
                + float(row.get("betweenness", 0.0))
            )
            for row in rows
        )
    for idx, (rel_type, score) in enumerate(
        sorted(rel_type_strengths.items(), key=lambda item: item[1], reverse=True)[:6],
        start=1,
    ):
        rel_analysis = next(
            (
                item
                for item in analysis.get("rel_type_analyses", [])
                if str(item.get("rel_type") or "RELATES_TO") == rel_type
            ),
            {},
        )
        top_names = [
            str(row.get("name") or "")
            for row in (rel_analysis.get("top_anchors") or [])[:6]
            if str(row.get("name") or "").strip()
        ]
        candidate_regions.append(
            {
                "region_id": f"reltype_{idx}",
                "kind": "rel_type",
                "title": rel_type,
                "description": (
                    f"Relation type {rel_type} has structural weight {round(score, 6)} "
                    f"with anchors: {', '.join(top_names) or 'none'}."
                ),
                "source_ids": [],
                "entity_names": top_names,
                "rel_types": [rel_type],
            }
        )

    fallback_concepts = []
    for region in candidate_regions[:8]:
        fallback_concepts.append(
            {
                "label": region["title"],
                "concept_type": region["kind"],
                "description": region["description"],
                "importance_reason": f"Derived from {region['kind']} candidate {region['region_id']}.",
                "evidence_region_ids": [region["region_id"]],
                "member_entity_names": region["entity_names"][:8],
            }
        )

    induced: dict[str, Any] = {"concepts": fallback_concepts, "meta_edges": []}
    if llm_provider and not isinstance(llm_provider, LocalEchoLLMProvider) and candidate_regions:
        schema = {
            "type": "object",
            "properties": {
                "concepts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "concept_type": {"type": "string"},
                            "description": {"type": "string"},
                            "importance_reason": {"type": "string"},
                            "evidence_region_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "member_entity_names": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "label",
                            "concept_type",
                            "description",
                            "importance_reason",
                            "evidence_region_ids",
                            "member_entity_names",
                        ],
                    },
                },
                "meta_edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_label": {"type": "string"},
                            "target_label": {"type": "string"},
                            "rel_type": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": [
                            "source_label",
                            "target_label",
                            "rel_type",
                            "description",
                        ],
                    },
                },
            },
            "required": ["concepts", "meta_edges"],
        }
        candidate_text = "\n".join(
            (
                f"- {region['region_id']} [{region['kind']}] {region['title']}: "
                f"{region['description']} "
                f"Entities: {', '.join(region['entity_names'][:8]) or 'none'}. "
                f"Rel types: {', '.join(region['rel_types']) or 'none'}."
            )
            for region in candidate_regions[:20]
        )
        prompt = (
            "You are inducing reusable semantic concepts from structural graph candidates.\n"
            "Do not return mechanical labels like community, connector, or bridge.\n"
            "Create a small set of higher-level concepts that explain what these regions collectively mean.\n"
            "Then create semantic meta-edges between concepts using only these relation types: "
            + ", ".join(sorted(semantic_edge_types))
            + ".\n"
            "Favor abstractions, roles, flows, tensions, and constraints over graph mechanics.\n\n"
            f"Collection: {collection['name']}\n\n"
            f"Candidates:\n{candidate_text}\n"
        )
        try:
            induced = await llm_provider.structured_extract(prompt=prompt, schema=schema)
        except Exception:
            induced = {"concepts": fallback_concepts, "meta_edges": []}

    region_lookup = {region["region_id"]: region for region in candidate_regions}
    concept_id_by_label: dict[str, str] = {}
    concept_source_ids: dict[str, list[str]] = {}
    for concept in induced.get("concepts", [])[:12]:
        label = str(concept.get("label") or "").strip()
        if not label:
            continue
        node_id = f"derived:concept:{slug(label)}"
        evidence_ids = [
            str(value)
            for value in concept.get("evidence_region_ids", [])
            if str(value).strip() in region_lookup
        ]
        source_ids = sorted(
            {
                source_id
                for region_id in evidence_ids
                for source_id in region_lookup[region_id]["source_ids"]
                if str(source_id).strip()
            }
        )
        description = (
            f"{str(concept.get('description') or '').strip()} "
            f"Why it matters: {str(concept.get('importance_reason') or '').strip()}"
        ).strip()
        nodes.append(
            {
                "id": node_id,
                "name": label,
                "collection_id": collection["id"],
                "type": "derived_concept",
                "description": description,
                "source_ids": source_ids,
            }
        )
        chunks.append(
            {
                "chunk_hash": derived_chunk_hash(collection["id"], "concept", label),
                "chunk_index": next_chunk_index(),
                "content": description,
                "metadata": {
                    "memory_type": "derived_graph",
                    "derived_kind": "concept",
                    "derived_id": node_id,
                    "concept_type": str(concept.get("concept_type") or "concept"),
                    "collection_id": collection["id"],
                },
            }
        )
        concept_id_by_label[label] = node_id
        concept_source_ids[node_id] = source_ids
        member_names = [
            str(value).strip()
            for value in concept.get("member_entity_names", [])
            if str(value).strip()
        ]
        for name in member_names[:10]:
            ref_id = ensure_entity_ref(name, supporting_ids=source_ids)
            add_edge(
                node_id,
                ref_id,
                rel_type="EVIDENCED_BY",
                description=f"Concept {label} is evidenced by entity {name}.",
                source_ids=source_ids,
            )

    for edge in induced.get("meta_edges", [])[:24]:
        source_label = str(edge.get("source_label") or "").strip()
        target_label = str(edge.get("target_label") or "").strip()
        rel_type = str(edge.get("rel_type") or "").strip().upper()
        if (
            not source_label
            or not target_label
            or rel_type not in semantic_edge_types
            or source_label not in concept_id_by_label
            or target_label not in concept_id_by_label
        ):
            continue
        source_id = concept_id_by_label[source_label]
        target_id = concept_id_by_label[target_label]
        edge_source_ids = sorted(
            set(concept_source_ids.get(source_id, []))
            | set(concept_source_ids.get(target_id, []))
        )
        add_edge(
            source_id,
            target_id,
            rel_type=rel_type,
            description=str(edge.get("description") or "").strip(),
            source_ids=edge_source_ids,
        )

    return {"nodes": nodes, "edges": edges, "chunks": chunks}

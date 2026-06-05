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


def build_collection_understanding(
    analysis: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    collection = analysis["collection"]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    seen_entity_refs: set[str] = set()

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

    def ensure_meta_node(
        node_id: str,
        *,
        name: str,
        node_type: str,
        description: str,
        source_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not any(node["id"] == node_id for node in nodes):
            nodes.append(
                {
                    "id": node_id,
                    "name": name,
                    "collection_id": collection["id"],
                    "type": node_type,
                    "description": description,
                    "source_ids": source_ids or [],
                }
            )
            chunks.append(
                {
                    "chunk_hash": derived_chunk_hash(collection["id"], "meta", node_id),
                    "chunk_index": next_chunk_index(),
                    "content": description,
                    "metadata": {
                        "memory_type": "derived_graph",
                        "derived_kind": "meta",
                        "derived_id": node_id,
                        "collection_id": collection["id"],
                        **(metadata or {}),
                    },
                }
            )
        return node_id

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

    for community in analysis["communities"]:
        rel_type = str(community.get("rel_type") or "RELATES_TO")
        node_id = f"derived:community:{rel_type.lower()}:{community['community_id']}"
        anchor_preview = community.get("anchor_preview") or []
        node_names = community.get("node_names") or []
        description = (
            f"{rel_type} community {community['community_id']} contains "
            f"{community['size']} entities with {community['strong_edge_count']} "
            f"strong {rel_type} edges. "
            f"Representative anchors: {', '.join(anchor_preview) or 'none'}. "
            f"This region appears centered on: "
            f"{', '.join(node_names[:6]) or 'no nodes'}."
        )
        source_ids = community.get("node_ids", [])
        nodes.append(
            {
                "id": node_id,
                "name": f"{rel_type} Community {community['community_id']}",
                "collection_id": collection["id"],
                "type": "derived_community",
                "description": description,
                "source_ids": source_ids,
            }
        )
        chunks.append(
            {
                "chunk_hash": derived_chunk_hash(
                    collection["id"],
                    "community",
                    rel_type,
                    community["community_id"],
                ),
                "chunk_index": next_chunk_index(),
                "content": description,
                "metadata": {
                    "memory_type": "derived_graph",
                    "derived_kind": "community",
                    "rel_type": rel_type,
                    "derived_id": node_id,
                    "collection_id": collection["id"],
                },
            }
        )
        for name in anchor_preview:
            ref_id = ensure_entity_ref(name, supporting_ids=source_ids)
            edges.append(
                {
                    "source_id": node_id,
                    "target_id": ref_id,
                    "id": f"{node_id}__{ref_id}",
                    "collection_id": collection["id"],
                    "rel_type": "SUMMARIZES",
                    "description": (
                        f"{rel_type} community {community['community_id']} is "
                        f"summarized in part by anchor entity {name}."
                    ),
                    "source_ids": source_ids,
                }
            )

    for bridge in analysis["bridge_nodes"]:
        rel_type_list = bridge.get("rel_types") or [bridge.get("rel_type") or "RELATES_TO"]
        rel_type_text = ", ".join(rel_type_list)
        node_id = (
            f"derived:bridge:{'_'.join(bridge['name'].strip().lower().split())}:"
            f"{'_'.join(rel_type.lower() for rel_type in rel_type_list)}"
        )
        source_ids = [bridge["node_id"]] if bridge.get("node_id") else []
        external_list = ", ".join(
            str(cid) for cid in bridge["external_communities"]
        ) or "none"
        description = (
            f"{bridge['name']} acts as a bridge node in community "
            f"{bridge['community_id']} across relation types {rel_type_text}. "
            f"It connects to external communities {external_list} "
            f"with external strength {bridge['external_strength']} and weighted "
            f"degree {bridge['weighted_degree']}. Inbound strength "
            f"{bridge['inbound_strength']} and outbound strength "
            f"{bridge['outbound_strength']}. Betweenness {bridge['betweenness']}, "
            f"closeness {bridge['closeness']}, hub {bridge['hub_score']}, "
            f"authority {bridge['authority_score']}."
        )
        nodes.append(
            {
                "id": node_id,
                "name": f"Bridge: {bridge['name']}",
                "collection_id": collection["id"],
                "type": "derived_bridge",
                "description": description,
                "source_ids": source_ids,
            }
        )
        chunks.append(
            {
                "chunk_hash": derived_chunk_hash(
                    collection["id"],
                    "bridge",
                    bridge["name"],
                    "_".join(rel_type_list),
                ),
                "chunk_index": next_chunk_index(),
                "content": description,
                "metadata": {
                    "memory_type": "derived_graph",
                    "derived_kind": "bridge",
                    "rel_types": rel_type_list,
                    "derived_id": node_id,
                    "collection_id": collection["id"],
                },
            }
        )
        ref_id = ensure_entity_ref(bridge["name"], supporting_ids=source_ids)
        edges.append(
            {
                "source_id": node_id,
                "target_id": ref_id,
                "id": f"{node_id}__{ref_id}",
                "collection_id": collection["id"],
                "rel_type": "FOCUSES_ON",
                "description": f"Bridge summary focuses on entity {bridge['name']}.",
                "source_ids": source_ids,
            }
        )

    for idx, path in enumerate(analysis["connector_paths"]):
        rel_type = str(path.get("rel_type") or "RELATES_TO")
        node_id = f"derived:path:{rel_type.lower()}:{idx}"
        path_nodes = path.get("nodes") or []
        supporting_rel_ids = [
            rel_id
            for hop in path.get("hops", [])
            for rel_id in hop.get("relationship_ids", [])
        ]
        flow_parts = [
            f"{hop['source']} -[{rel_type}]-> {hop['target']}"
            for hop in path.get("hops", [])
        ]
        description = (
            f"Directed {rel_type} connector path from {path['from_anchor']} to "
            f"{path['to_anchor']} crosses {path['hop_count']} hops with score "
            f"{path['path_score']}. Flow: "
            f"{'; '.join(flow_parts) or ' -> '.join(path_nodes)}."
        )
        nodes.append(
            {
                "id": node_id,
                "name": (
                    f"{rel_type} Connector {idx}: {path['from_anchor']} -> "
                    f"{path['to_anchor']}"
                ),
                "collection_id": collection["id"],
                "type": "derived_connector",
                "description": description,
                "source_ids": supporting_rel_ids,
            }
        )
        chunks.append(
            {
                "chunk_hash": derived_chunk_hash(
                    collection["id"],
                    "connector",
                    rel_type,
                    idx,
                ),
                "chunk_index": next_chunk_index(),
                "content": description,
                "metadata": {
                    "memory_type": "derived_graph",
                    "derived_kind": "connector",
                    "rel_type": rel_type,
                    "derived_id": node_id,
                    "collection_id": collection["id"],
                },
            }
        )
        for name in path_nodes:
            ref_id = ensure_entity_ref(name)
            edges.append(
                {
                    "source_id": node_id,
                    "target_id": ref_id,
                    "id": f"{node_id}__{ref_id}",
                    "collection_id": collection["id"],
                    "rel_type": "USES",
                    "description": (
                        f"Connector path between {path['from_anchor']} and "
                        f"{path['to_anchor']} uses entity {name}."
                    ),
                    "source_ids": supporting_rel_ids,
                }
            )
        if path["source_community"] != path["target_community"]:
            for community_id in (
                path["source_community"],
                path["target_community"],
            ):
                target_id = f"derived:community:{rel_type.lower()}:{community_id}"
                edges.append(
                    {
                        "source_id": node_id,
                        "target_id": target_id,
                        "id": f"{node_id}__{target_id}",
                        "collection_id": collection["id"],
                        "rel_type": "CONNECTS",
                        "description": (
                            f"Connector path links into community {community_id}."
                        ),
                        "source_ids": supporting_rel_ids,
                    }
                )

    rel_family_map = {
        "CALLS": "behavior",
        "USES": "behavior",
        "DEPENDS_ON": "behavior",
        "REFERENCES": "behavior",
        "IMPORTS": "behavior",
        "DEFINES": "structure",
        "EXTENDS": "structure",
        "IMPLEMENTS": "structure",
        "DOCUMENTS": "evidence",
        "TESTS": "evidence",
        "RELATES_TO": "generic",
    }

    def rel_family(rel_type: str) -> str:
        return rel_family_map.get(rel_type, "other")

    rel_type_totals: dict[str, float] = {}
    route_scores = {
        "hub": 0.0,
        "authority": 0.0,
        "bridge": 0.0,
        "central": 0.0,
        "importance": 0.0,
    }
    for rel_analysis in analysis.get("rel_type_analyses", []):
        rel_type = str(rel_analysis.get("rel_type") or "RELATES_TO")
        rows = rel_analysis.get("node_metrics", [])
        if not rows:
            continue
        total = 0.0
        for row in rows:
            hub = float(row.get("hub_score", 0.0))
            authority = float(row.get("authority_score", 0.0))
            bridge = float(row.get("betweenness", 0.0))
            central = float(row.get("closeness", 0.0))
            importance = (
                float(row.get("pagerank", 0.0))
                + float(row.get("eigenvector_score", 0.0))
            ) / 2.0
            route_scores["hub"] += hub
            route_scores["authority"] += authority
            route_scores["bridge"] += bridge
            route_scores["central"] += central
            route_scores["importance"] += importance
            total += hub + authority + bridge + central + importance
        rel_type_totals[rel_type] = total

    def normalize_map(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_value = max(scores.values()) or 0.0
        if max_value <= 0.0:
            return {key: 0.0 for key in scores}
        return {key: float(value) / float(max_value) for key, value in scores.items()}

    normalized_routes = normalize_map(route_scores)
    normalized_rel_types = normalize_map(rel_type_totals)
    sorted_routes = sorted(
        normalized_routes.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    primary_route = sorted_routes[0][0] if sorted_routes else "central"

    route_nodes: dict[str, str] = {}
    for route_name, score in normalized_routes.items():
        node_id = f"derived:meta:route:{route_name}"
        route_nodes[route_name] = ensure_meta_node(
            node_id,
            name=f"Route {route_name.title()}",
            node_type="derived_meta_route",
            description=(
                f"Meta route archetype {route_name} with normalized score {round(score, 6)}. "
                "Represents one lens for expanding the derived graph."
            ),
            metadata={"route_name": route_name},
        )

    rel_type_nodes: dict[str, str] = {}
    for rel_type, score in normalized_rel_types.items():
        rel_type_nodes[rel_type] = ensure_meta_node(
            f"derived:meta:rel_type:{rel_type.lower()}",
            name=f"Relation Type {rel_type}",
            node_type="derived_meta_rel_type",
            description=(
                f"Meta node for relation type {rel_type} in family {rel_family(rel_type)} "
                f"with normalized weight {round(score, 6)} in this collection analysis."
            ),
            metadata={"rel_type": rel_type},
        )

    top_rel_types = sorted(
        normalized_rel_types.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    for rel_type, score in top_rel_types:
        add_edge(
            route_nodes[primary_route],
            rel_type_nodes[rel_type],
            rel_type="SUPPORTS",
            description=(
                f"Primary route {primary_route} is supported by high-signal relation type "
                f"{rel_type} with normalized weight {round(score, 6)}."
            ),
        )

    if len(sorted_routes) > 1 and (sorted_routes[0][1] - sorted_routes[1][1]) <= 0.2:
        add_edge(
            route_nodes[sorted_routes[1][0]],
            route_nodes[sorted_routes[0][0]],
            rel_type="QUALIFIES",
            description=(
                f"Secondary route {sorted_routes[1][0]} nearly matches primary route "
                f"{sorted_routes[0][0]}, so it qualifies the expansion strategy."
            ),
        )

    rel_rank = sorted(
        normalized_rel_types.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if rel_rank:
        top_rel_type, top_rel_score = rel_rank[0]
        if top_rel_type == "RELATES_TO" and top_rel_score >= 0.75:
            add_edge(
                rel_type_nodes[top_rel_type],
                route_nodes[primary_route],
                rel_type="LIMITS",
                description=(
                    "Generic RELATES_TO edges dominate this collection slice, which limits "
                    "the interpretability of the primary route."
                ),
            )

        non_generic = [
            (rel_type, score)
            for rel_type, score in rel_rank
            if rel_family(rel_type) != "generic"
        ]
        if non_generic:
            top_non_generic_type, top_non_generic_score = non_generic[0]
            if float(top_non_generic_score) >= 0.35:
                add_edge(
                    rel_type_nodes[top_non_generic_type],
                    route_nodes[primary_route],
                    rel_type="QUALIFIES",
                    description=(
                        f"Non-generic relation type {top_non_generic_type} sharpens the "
                        f"otherwise broad {primary_route} route."
                    ),
                )
        if len(non_generic) > 1:
            a_rel, a_score = non_generic[0]
            b_rel, b_score = non_generic[1]
            if rel_family(a_rel) != rel_family(b_rel):
                add_edge(
                    rel_type_nodes[a_rel],
                    rel_type_nodes[b_rel],
                    rel_type="DISTINGUISHES",
                    description=(
                        f"{a_rel} and {b_rel} are both strong but come from different semantic "
                        "families, so they distinguish different structural explanations."
                    ),
                )
            elif abs(float(a_score) - float(b_score)) <= 0.05:
                add_edge(
                    rel_type_nodes[a_rel],
                    rel_type_nodes[b_rel],
                    rel_type="CONTRADICTS",
                    description=(
                        f"{a_rel} and {b_rel} carry nearly equal weight yet imply competing "
                        "interpretations of the collection structure."
                    ),
                )

    bridge_lookup = {
        str(node.get("name") or ""): str(node["id"])
        for node in nodes
        if str(node.get("type") or "") == "derived_bridge"
    }
    for bridge in analysis.get("bridge_nodes", [])[: max(10, len(analysis.get("bridge_nodes", [])))]:
        if (
            len(bridge.get("external_communities", [])) >= 4
            and float(bridge.get("betweenness", 0.0)) > 0.0
        ):
            bridge_name = f"Bridge: {bridge['name']}"
            bridge_node_id = bridge_lookup.get(bridge_name)
            if bridge_node_id:
                add_edge(
                    bridge_node_id,
                    route_nodes["bridge"],
                    rel_type="JUSTIFIES",
                    description=(
                        f"{bridge['name']} connects many external communities and therefore "
                        "justifies a bridge-oriented reading of the graph."
                    ),
                    source_ids=[str(bridge.get("node_id") or "")] if bridge.get("node_id") else [],
                )

    connector_counter: dict[tuple[str, str], dict[str, Any]] = {}
    for path in analysis.get("connector_paths", []):
        rel_type = str(path.get("rel_type") or "RELATES_TO")
        motif = " -> ".join((path.get("nodes") or [])[:3])
        if not motif:
            continue
        key = (rel_type, motif)
        bucket = connector_counter.setdefault(
            key,
            {
                "count": 0,
                "examples": [],
                "source_ids": [],
            },
        )
        bucket["count"] += 1
        example = f"{path.get('from_anchor', '?')} -> {path.get('to_anchor', '?')}"
        if len(bucket["examples"]) < 3:
            bucket["examples"].append(example)
        for hop in path.get("hops", []):
            for rel_id in hop.get("relationship_ids", []):
                if rel_id not in bucket["source_ids"]:
                    bucket["source_ids"].append(rel_id)

    for (rel_type, motif), bucket in sorted(
        connector_counter.items(),
        key=lambda item: item[1]["count"],
        reverse=True,
    )[:8]:
        meta_node_id = ensure_meta_node(
            f"derived:meta:connector:{rel_type.lower()}:{hashlib.md5(motif.encode('utf-8')).hexdigest()[:12]}",
            name=f"Meta Connector {rel_type}",
            node_type="derived_meta_connector",
            description=(
                f"Recurring connector motif for {rel_type}: {motif}. "
                f"Observed {bucket['count']} times with examples: "
                f"{', '.join(bucket['examples']) or 'none'}."
            ),
            source_ids=bucket["source_ids"],
            metadata={"rel_type": rel_type, "motif": motif},
        )
        add_edge(
            meta_node_id,
            route_nodes[primary_route],
            rel_type="SUPPORTS",
            description=(
                f"Recurring {rel_type} connector motif {motif} supports the "
                f"{primary_route} route."
            ),
            source_ids=bucket["source_ids"],
        )

        if rel_type in rel_type_nodes:
            add_edge(
                meta_node_id,
                rel_type_nodes[rel_type],
                rel_type="JUSTIFIES",
                description=(
                    f"Recurring connector motif {motif} justifies the importance of "
                    f"relation type {rel_type}."
                ),
                source_ids=bucket["source_ids"],
            )

    return {"nodes": nodes, "edges": edges, "chunks": chunks}

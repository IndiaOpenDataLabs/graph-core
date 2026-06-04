"""Offline analytics over the canonical collection graph.

This module builds a lightweight structural view over the merged graph:
- strong-edge communities from a weighted projection
- articulation / connector nodes that bridge graph regions
- bounded connector paths between top anchors

It is intentionally dependency-free for now.
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
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
class NodeMetrics:
    node_id: uuid.UUID
    name: str
    community_id: int
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


def _shortest_path(
    start_id: uuid.UUID,
    end_id: uuid.UUID,
    adjacency: dict[uuid.UUID, dict[uuid.UUID, EdgeAggregate]],
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
    max_weight = max_relationship_weight or settings.graph_rag_max_relationship_weight
    node_names = {node.id: node.name for node in nodes}

    by_pair: dict[
        tuple[uuid.UUID, uuid.UUID],
        dict[str, Any],
    ] = {}
    for rel in relationships:
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

    full_adjacency: dict[uuid.UUID, dict[uuid.UUID, EdgeAggregate]] = defaultdict(dict)
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

    strong_adjacency: dict[
        uuid.UUID, dict[uuid.UUID, EdgeAggregate]
    ] = defaultdict(dict)
    for edge in all_edges:
        if edge.strength < min_edge_strength:
            continue
        strong_adjacency[edge.node_a][edge.node_b] = edge
        strong_adjacency[edge.node_b][edge.node_a] = edge

    node_ids = [node.id for node in nodes]
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
    for node in nodes:
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
            weighted_degree
            + (2.0 * len(external_communities))
            + (1.5 * external_strength)
            + (3.0 if node.id in articulation else 0.0)
        )
        metrics.append(
            NodeMetrics(
                node_id=node.id,
                name=node.name,
                community_id=community_of.get(node.id, -1),
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
            item.weighted_degree,
            len(item.external_communities),
            item.name,
        ),
        reverse=True,
    )[:max_anchors]

    connector_paths: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, ...]] = set()
    for left_anchor, right_anchor in combinations(top_anchors, 2):
        if len(connector_paths) >= max_connector_paths:
            break
        path = _shortest_path(
            left_anchor.node_id,
            right_anchor.node_id,
            full_adjacency,
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
            edge = full_adjacency[current_id][next_id]
            path_score += edge.strength
            hops.append(
                {
                    "source_id": str(current_id),
                    "source": node_names[current_id],
                    "target_id": str(next_id),
                    "target": node_names[next_id],
                    "strength": round(edge.strength, 4),
                    "rel_types": list(edge.rel_types),
                    "relationship_ids": list(edge.relationship_ids),
                }
            )
        connector_paths.append(
            {
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
        "parameters": {
            "min_edge_strength": min_edge_strength,
            "min_community_size": min_community_size,
            "max_anchors": max_anchors,
            "max_path_depth": max_path_depth,
            "max_connector_paths": max_connector_paths,
            "max_relationship_weight": max_weight,
        },
        "totals": {
            "entities": len(nodes),
            "relationships": len(relationships),
            "undirected_pairs": len(all_edges),
            "strong_pairs": sum(
                1 for edge in all_edges if edge.strength >= min_edge_strength
            ),
            "communities": len(communities),
        },
        "communities": community_summaries,
        "top_anchors": [
            {
                "node_id": str(metric.node_id),
                "name": metric.name,
                "community_id": metric.community_id,
                "anchor_score": metric.anchor_score,
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
                "anchor_score": metric.anchor_score,
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
    for community in analysis["communities"]:
        node_id = f"derived:community:{community['community_id']}"
        anchor_preview = community.get("anchor_preview") or []
        node_names = community.get("node_names") or []
        description = (
            f"Community {community['community_id']} contains {community['size']} "
            f"entities with {community['strong_edge_count']} strong edges. "
            f"Representative anchors: {', '.join(anchor_preview) or 'none'}. "
            f"This region appears centered on: "
            f"{', '.join(node_names[:6]) or 'no nodes'}."
        )
        source_ids = community.get("node_ids", [])
        nodes.append(
            {
                "id": node_id,
                "name": f"Community {community['community_id']}",
                "collection_id": collection["id"],
                "type": "derived_community",
                "description": description,
                "source_ids": source_ids,
            }
        )
        chunks.append(
            {
                "chunk_hash": (
                    f"{collection['id']}::derived::community::"
                    f"{community['community_id']}"
                ),
                "chunk_index": chunk_index,
                "content": description,
                "metadata": {
                    "memory_type": "derived_graph",
                    "derived_kind": "community",
                    "derived_id": node_id,
                    "collection_id": collection["id"],
                },
            }
        )
        chunk_index += 1
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
                        f"Community {community['community_id']} is summarized in part "
                        f"by anchor entity {name}."
                    ),
                    "source_ids": source_ids,
                }
            )

    for bridge in analysis["bridge_nodes"]:
        node_id = f"derived:bridge:{'_'.join(bridge['name'].strip().lower().split())}"
        source_ids = [bridge["node_id"]] if bridge.get("node_id") else []
        external_list = ", ".join(
            str(cid) for cid in bridge["external_communities"]
        ) or "none"
        description = (
            f"{bridge['name']} acts as a bridge node in community "
            f"{bridge['community_id']}. It connects to external communities "
            f"{external_list} "
            f"with external strength {bridge['external_strength']} and weighted "
            f"degree {bridge['weighted_degree']}."
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
                "chunk_hash": f"{collection['id']}::derived::bridge::{bridge['name']}",
                "chunk_index": chunk_index,
                "content": description,
                "metadata": {
                    "memory_type": "derived_graph",
                    "derived_kind": "bridge",
                    "derived_id": node_id,
                    "collection_id": collection["id"],
                },
            }
        )
        chunk_index += 1
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
        node_id = f"derived:path:{idx}"
        path_nodes = path.get("nodes") or []
        supporting_rel_ids = [
            rel_id
            for hop in path.get("hops", [])
            for rel_id in hop.get("relationship_ids", [])
        ]
        description = (
            f"Connector path from {path['from_anchor']} to {path['to_anchor']} "
            f"crosses {path['hop_count']} hops with score {path['path_score']}. "
            f"Flow: {' -> '.join(path_nodes)}."
        )
        nodes.append(
            {
                "id": node_id,
                "name": (
                    f"Connector {idx}: {path['from_anchor']} -> "
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
                "chunk_hash": f"{collection['id']}::derived::connector::{idx}",
                "chunk_index": chunk_index,
                "content": description,
                "metadata": {
                    "memory_type": "derived_graph",
                    "derived_kind": "connector",
                    "derived_id": node_id,
                    "collection_id": collection["id"],
                },
            }
        )
        chunk_index += 1
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
                target_id = f"derived:community:{community_id}"
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

    return {"nodes": nodes, "edges": edges, "chunks": chunks}

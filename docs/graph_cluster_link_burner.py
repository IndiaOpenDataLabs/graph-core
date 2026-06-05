#!/usr/bin/env python3
"""Inspect how two typed Louvain communities connect through the base graph.

Examples:
  PYTHONPATH=src .venv/bin/python docs/graph_cluster_link_burner.py yoga
  PYTHONPATH=src .venv/bin/python docs/graph_cluster_link_burner.py yoga --first RELATES_TO:27 --second CHARACTERIZES:12
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from collections import Counter, defaultdict, deque
from pathlib import Path

from sqlalchemy import select  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_core.database import AsyncSessionLocal  # noqa: E402
from graph_core.models.collection import Collection  # noqa: E402
from graph_core.services.graph.analytics import (  # noqa: E402
    _build_louvain_communities,
    _load_graph_records,
)


def shortest_paths_between_sets(
    adjacency: dict[str, list[tuple[str, str, float]]],
    starts: set[str],
    targets: set[str],
    *,
    max_depth: int = 4,
    limit: int = 20,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for start in starts:
        queue: deque[tuple[str, list[str], list[str], float]] = deque(
            [(start, [start], [], 0.0)]
        )
        seen_depth: dict[str, int] = {start: 0}
        while queue:
            current, path_nodes, path_rels, score = queue.popleft()
            depth = len(path_rels)
            if depth >= max_depth:
                continue
            for neighbor, rel_type, weight in adjacency.get(current, []):
                next_nodes = [*path_nodes, neighbor]
                next_rels = [*path_rels, rel_type]
                next_score = score + float(weight)
                if neighbor in targets:
                    results.append(
                        {
                            "start": start,
                            "end": neighbor,
                            "nodes": next_nodes,
                            "rel_types": next_rels,
                            "hop_count": len(next_rels),
                            "path_score": round(next_score, 6),
                        }
                    )
                    continue
                next_depth = depth + 1
                if next_depth >= max_depth:
                    continue
                if neighbor not in seen_depth or next_depth < seen_depth[neighbor]:
                    seen_depth[neighbor] = next_depth
                    queue.append((neighbor, next_nodes, next_rels, next_score))
    results.sort(
        key=lambda item: (
            -float(item["path_score"]),
            int(item["hop_count"]),
            tuple(item["rel_types"]),
            tuple(item["nodes"]),
        )
    )
    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()
    for result in results:
        key = tuple(result["nodes"]) + tuple(result["rel_types"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
        if len(deduped) >= limit:
            break
    return deduped


def pick_default_pair(communities: list[dict[str, object]]) -> tuple[str, str]:
    if len(communities) < 2:
        raise ValueError("Need at least two communities to compare")
    return str(communities[0]["community_id"]), str(communities[1]["community_id"])


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("collection")
    parser.add_argument("--first")
    parser.add_argument("--second")
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    async with AsyncSessionLocal() as session:
        collection = (
            await session.execute(select(Collection).where(Collection.name == args.collection))
        ).scalar_one()

    _, nodes, relationships, _aliases_by_entity_id = await _load_graph_records(
        collection.id
    )
    communities = _build_louvain_communities(nodes, relationships)
    community_by_id = {str(item["community_id"]): item for item in communities}

    first_id = args.first
    second_id = args.second
    if not first_id or not second_id:
        first_id, second_id = pick_default_pair(communities)

    if first_id not in community_by_id:
        raise ValueError(f"Unknown community id: {first_id}")
    if second_id not in community_by_id:
        raise ValueError(f"Unknown community id: {second_id}")

    first = community_by_id[first_id]
    second = community_by_id[second_id]

    first_nodes = set(str(node_id) for node_id in first["node_ids"])
    second_nodes = set(str(node_id) for node_id in second["node_ids"])
    name_by_id = {str(node.id): node.name for node in nodes}

    adjacency: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    direct_cross_edges: list[dict[str, object]] = []
    rel_counter: Counter[str] = Counter()
    boundary_counter: Counter[str] = Counter()

    for rel in relationships:
        source_id = str(rel.source_id)
        target_id = str(rel.target_id)
        rel_type = str(rel.rel_type or "RELATES_TO")
        weight = float(rel.weight or 0)
        adjacency[source_id].append((target_id, rel_type, weight))

        crosses_forward = source_id in first_nodes and target_id in second_nodes
        crosses_backward = source_id in second_nodes and target_id in first_nodes
        if crosses_forward or crosses_backward:
            direct_cross_edges.append(
                {
                    "source_name": rel.source_name,
                    "target_name": rel.target_name,
                    "rel_type": rel_type,
                    "weight": weight,
                    "direction": "first->second" if crosses_forward else "second->first",
                }
            )
            rel_counter[rel_type] += 1
            boundary_counter[rel.source_name] += 1
            boundary_counter[rel.target_name] += 1

    forward_paths = shortest_paths_between_sets(
        adjacency,
        first_nodes,
        second_nodes,
        max_depth=args.max_depth,
        limit=12,
    )
    backward_paths = shortest_paths_between_sets(
        adjacency,
        second_nodes,
        first_nodes,
        max_depth=args.max_depth,
        limit=12,
    )

    bridge_counter: Counter[str] = Counter()
    path_rel_counter: Counter[str] = Counter()
    for path in [*forward_paths, *backward_paths]:
        internal_nodes = list(path["nodes"])[1:-1]
        for node_id in internal_nodes:
            bridge_counter[name_by_id.get(node_id, node_id)] += 1
        for rel_type in list(path["rel_types"]):
            path_rel_counter[rel_type] += 1

    print(f"collection: {collection.name} ({collection.id})")
    print(f"first:  {first['community_id']} rel_type={first['rel_type']} size={first['size']} score={first['score']}")
    print(f"second: {second['community_id']} rel_type={second['rel_type']} size={second['size']} score={second['score']}")
    print()
    print("first top entities:")
    for name in list(first.get("node_names", []))[:12]:
        print(f"  - {name}")
    print("second top entities:")
    for name in list(second.get("node_names", []))[:12]:
        print(f"  - {name}")
    print()
    print(f"direct cross edges: {len(direct_cross_edges)}")
    for edge in sorted(
        direct_cross_edges,
        key=lambda item: (-float(item["weight"]), str(item["rel_type"]), str(item["source_name"]), str(item["target_name"])),
    )[:20]:
        print(
            f"  - {edge['direction']} {edge['source_name']} -[{edge['rel_type']}]-> "
            f"{edge['target_name']} (weight={edge['weight']})"
        )
    print()
    print("cross-edge rel types:")
    for rel_type, count in rel_counter.most_common(12):
        print(f"  - {rel_type}: {count}")
    print()
    print("boundary entities:")
    for name, count in boundary_counter.most_common(12):
        print(f"  - {name}: {count}")
    print()
    print("forward paths (first -> second):")
    for path in forward_paths[:12]:
        print(
            f"  - score={path['path_score']} hops={path['hop_count']} "
            f"{' -> '.join(name_by_id.get(node_id, node_id) for node_id in path['nodes'])} "
            f"rels={list(path['rel_types'])}"
        )
    print()
    print("backward paths (second -> first):")
    for path in backward_paths[:12]:
        print(
            f"  - score={path['path_score']} hops={path['hop_count']} "
            f"{' -> '.join(name_by_id.get(node_id, node_id) for node_id in path['nodes'])} "
            f"rels={list(path['rel_types'])}"
        )
    print()
    print("bridge/intermediate entities from short paths:")
    for name, count in bridge_counter.most_common(12):
        print(f"  - {name}: {count}")
    print()
    print("path rel types:")
    for rel_type, count in path_rel_counter.most_common(12):
        print(f"  - {rel_type}: {count}")


if __name__ == "__main__":
    asyncio.run(main())

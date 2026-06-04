#!/usr/bin/env python3
"""Experimental third-layer burner over the derived graph.

This script takes the current rel-type-aware analysis output and
compresses it into more abstract meta-patterns:

- route profile from matched node metrics
- repeated role entities across rel_types
- repeated bridge entities across rel_types
- recurring typed connector flows

Examples:
  PYTHONPATH=src .venv/bin/python docs/graph_meta_burner.py rlm exception
  PYTHONPATH=src .venv/bin/python docs/graph_meta_burner.py rlm architecture
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import select  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_core.database import AsyncSessionLocal  # noqa: E402
from graph_core.models.collection import Collection  # noqa: E402
from graph_core.services.graph.analytics import analyze_collection_graph  # noqa: E402

QUESTION_TERMS: dict[str, tuple[str, ...]] = {
    "exception": ("error", "exception"),
    "architecture": (
        "api",
        "client",
        "manager",
        "handler",
        "service",
        "gateway",
        "server",
        "repl",
        "environment",
        "config",
        "query",
        "auth",
        "tool",
        "broker",
        "cache",
        "logger",
        "provider",
        "core",
        "module",
    ),
}

REL_FAMILIES: dict[str, str] = {
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


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_value = max(scores.values()) or 0.0
    if max_value <= 0.0:
        return {key: 0.0 for key in scores}
    return {key: float(value) / float(max_value) for key, value in scores.items()}


def family_for_rel_type(rel_type: str) -> str:
    return REL_FAMILIES.get(rel_type, "other")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("collection")
    parser.add_argument("question_type", choices=sorted(QUESTION_TERMS))
    args = parser.parse_args()

    terms = QUESTION_TERMS[args.question_type]
    async with AsyncSessionLocal() as session:
        coll = await session.scalar(
            select(Collection).where(Collection.name == args.collection)
        )
        if not coll:
            raise ValueError(f"Collection {args.collection!r} not found")

    analysis = await analyze_collection_graph(coll.id)

    matched_rows: list[dict[str, object]] = []
    for rel_analysis in analysis.get("rel_type_analyses", []):
        for row in rel_analysis.get("node_metrics", []):
            if contains_any(str(row.get("name") or ""), terms):
                matched_rows.append(row)

    route_scores = {
        "hub": 0.0,
        "authority": 0.0,
        "bridge": 0.0,
        "central": 0.0,
        "importance": 0.0,
    }
    rel_type_scores: dict[str, float] = defaultdict(float)
    for row in matched_rows:
        route_scores["hub"] += float(row.get("hub_score", 0.0))
        route_scores["authority"] += float(row.get("authority_score", 0.0))
        route_scores["bridge"] += float(row.get("betweenness", 0.0))
        route_scores["central"] += float(row.get("closeness", 0.0))
        route_scores["importance"] += (
            float(row.get("pagerank", 0.0))
            + float(row.get("eigenvector_score", 0.0))
        ) / 2.0
        rel_type_scores[str(row.get("rel_type") or "RELATES_TO")] += (
            float(row.get("hub_score", 0.0))
            + float(row.get("authority_score", 0.0))
            + float(row.get("betweenness", 0.0))
            + float(row.get("closeness", 0.0))
        )

    normalized_routes = normalize(route_scores)
    normalized_rel_types = normalize(dict(rel_type_scores))

    role_entities: dict[str, dict[str, object]] = {}
    for row in matched_rows:
        name = str(row.get("name") or "")
        entry = role_entities.setdefault(
            name,
            {
                "name": name,
                "rel_types": set(),
                "hub": 0.0,
                "authority": 0.0,
                "bridge": 0.0,
                "central": 0.0,
                "importance": 0.0,
            },
        )
        entry["rel_types"].add(str(row.get("rel_type") or "RELATES_TO"))
        entry["hub"] = max(float(entry["hub"]), float(row.get("hub_score", 0.0)))
        entry["authority"] = max(
            float(entry["authority"]), float(row.get("authority_score", 0.0))
        )
        entry["bridge"] = max(
            float(entry["bridge"]), float(row.get("betweenness", 0.0))
        )
        entry["central"] = max(
            float(entry["central"]), float(row.get("closeness", 0.0))
        )
        entry["importance"] = max(
            float(entry["importance"]),
            (
                float(row.get("pagerank", 0.0))
                + float(row.get("eigenvector_score", 0.0))
            )
            / 2.0,
        )

    bridge_entities: dict[str, dict[str, object]] = {}
    for row in analysis.get("bridge_nodes", []):
        name = str(row.get("name") or "")
        if not contains_any(name, terms):
            continue
        entry = bridge_entities.setdefault(
            name,
            {
                "name": name,
                "rel_types": set(),
                "external_communities": set(),
                "betweenness": 0.0,
                "closeness": 0.0,
            },
        )
        for rel_type in row.get("rel_types", []):
            entry["rel_types"].add(str(rel_type))
        for cid in row.get("external_communities", []):
            entry["external_communities"].add(int(cid))
        entry["betweenness"] = max(
            float(entry["betweenness"]), float(row.get("betweenness", 0.0))
        )
        entry["closeness"] = max(
            float(entry["closeness"]), float(row.get("closeness", 0.0))
        )

    bridge_rows = sorted(
        bridge_entities.values(),
        key=lambda item: (
            len(item["rel_types"]),
            len(item["external_communities"]),
            float(item["betweenness"]),
        ),
        reverse=True,
    )

    connector_motifs: Counter[tuple[str, str]] = Counter()
    connector_examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in analysis.get("connector_paths", []):
        nodes = path.get("nodes") or []
        if not any(contains_any(str(node), terms) for node in nodes):
            continue
        rel_type = str(path.get("rel_type") or "RELATES_TO")
        from_anchor = str(path.get("from_anchor") or "")
        to_anchor = str(path.get("to_anchor") or "")
        key = (rel_type, " -> ".join(nodes[:3]))
        connector_motifs[key] += 1
        if len(connector_examples[key]) < 3:
            connector_examples[key].append(f"{from_anchor} -> {to_anchor}")

    community_patterns: Counter[tuple[str, int]] = Counter()
    for community in analysis.get("communities", []):
        blob = " ".join(community.get("node_names") or [])
        if not contains_any(blob, terms):
            continue
        community_patterns[(str(community.get("rel_type") or "RELATES_TO"), int(community.get("size") or 0))] += 1

    meta_edges: list[dict[str, object]] = []
    sorted_routes = sorted(
        normalized_routes.items(), key=lambda item: item[1], reverse=True
    )
    primary_route = sorted_routes[0][0] if sorted_routes else "central"
    if sorted_routes:
        for rel_type, value in sorted(
            normalized_rel_types.items(), key=lambda item: item[1], reverse=True
        )[:5]:
            meta_edges.append(
                {
                    "source": f"route:{primary_route}",
                    "rel_type": "SUPPORTS",
                    "target": f"rel_type:{rel_type}",
                    "score": round(float(value), 6),
                    "reason": "Top relation type aligned with the selected route.",
                }
            )
        if len(sorted_routes) > 1 and (sorted_routes[0][1] - sorted_routes[1][1]) <= 0.2:
            meta_edges.append(
                {
                    "source": f"route:{sorted_routes[1][0]}",
                    "rel_type": "QUALIFIES",
                    "target": f"route:{sorted_routes[0][0]}",
                    "score": round(float(sorted_routes[1][1]), 6),
                    "reason": "Secondary route score is close to the primary route.",
                }
            )

    rel_rel = sorted(
        normalized_rel_types.items(), key=lambda item: item[1], reverse=True
    )
    if rel_rel:
        top_rel_type, top_rel_score = rel_rel[0]
        if top_rel_type == "RELATES_TO" and top_rel_score >= 0.75:
            meta_edges.append(
                {
                    "source": "rel_type:RELATES_TO",
                    "rel_type": "LIMITS",
                    "target": f"route:{primary_route}",
                    "score": round(float(top_rel_score), 6),
                    "reason": "Generic relationships dominate this slice, reducing interpretability.",
                }
            )
        non_generic = [
            (rel_type, score)
            for rel_type, score in rel_rel
            if family_for_rel_type(rel_type) != "generic"
        ]
        if non_generic:
            top_non_generic_type, top_non_generic_score = non_generic[0]
            if float(top_non_generic_score) >= 0.35:
                meta_edges.append(
                    {
                        "source": f"rel_type:{top_non_generic_type}",
                        "rel_type": "QUALIFIES",
                        "target": f"route:{primary_route}",
                        "score": round(float(top_non_generic_score), 6),
                        "reason": "A strong non-generic relation family sharpens the otherwise broad route.",
                    }
                )
        if len(non_generic) > 1:
            a_rel, a_score = non_generic[0]
            b_rel, b_score = non_generic[1]
            if family_for_rel_type(a_rel) != family_for_rel_type(b_rel):
                meta_edges.append(
                    {
                        "source": f"rel_type:{a_rel}",
                        "rel_type": "DISTINGUISHES",
                        "target": f"rel_type:{b_rel}",
                        "score": round(min(float(a_score), float(b_score)), 6),
                        "reason": "Top non-generic relation types come from different semantic families.",
                    }
                )
            elif abs(float(a_score) - float(b_score)) <= 0.05:
                meta_edges.append(
                    {
                        "source": f"rel_type:{a_rel}",
                        "rel_type": "CONTRADICTS",
                        "target": f"rel_type:{b_rel}",
                        "score": round(min(float(a_score), float(b_score)), 6),
                        "reason": "Top non-generic relation types compete with nearly equal weight but imply different interpretations.",
                    }
                )
        if len(rel_rel) > 1:
            a_rel, a_score = rel_rel[0]
            b_rel, b_score = rel_rel[1]
            if (
                family_for_rel_type(a_rel) != family_for_rel_type(b_rel)
                and family_for_rel_type(a_rel) != "generic"
                and family_for_rel_type(b_rel) != "generic"
            ):
                meta_edges.append(
                    {
                        "source": f"rel_type:{a_rel}",
                        "rel_type": "DISTINGUISHES",
                        "target": f"rel_type:{b_rel}",
                        "score": round(min(float(a_score), float(b_score)), 6),
                        "reason": "Top relation types come from different semantic families.",
                    }
                )
            elif abs(float(a_score) - float(b_score)) <= 0.05:
                meta_edges.append(
                    {
                        "source": f"rel_type:{a_rel}",
                        "rel_type": "CONTRADICTS",
                        "target": f"rel_type:{b_rel}",
                        "score": round(min(float(a_score), float(b_score)), 6),
                        "reason": "Top relation types compete with nearly equal weight but imply different interpretations.",
                    }
                )

    for row in bridge_rows[:5]:
        if len(row["external_communities"]) >= 4 and float(row["betweenness"]) > 0.0:
            meta_edges.append(
                {
                    "source": f"bridge:{row['name']}",
                    "rel_type": "JUSTIFIES",
                    "target": "route:bridge",
                    "score": round(float(row["betweenness"]), 6),
                    "reason": "Bridge entity connects many external communities.",
                }
            )

    for (rel_type, motif), count in connector_motifs.most_common(5):
        meta_edges.append(
            {
                "source": f"connector:{rel_type}:{motif}",
                "rel_type": "SUPPORTS",
                "target": f"route:{primary_route}",
                "score": round(float(count), 6),
                "reason": "Recurring connector motif reinforces the selected route.",
            }
        )

    print(f"collection={coll.name} id={coll.id}")
    print(f"question_type={args.question_type}")
    print(f"matched_metric_rows={len(matched_rows)}")

    print("\n[route profile]")
    for label, value in sorted(
        normalized_routes.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {label}: {value:.6f}")

    print("\n[dominant rel types]")
    for rel_type, value in sorted(
        normalized_rel_types.items(), key=lambda item: item[1], reverse=True
    )[:10]:
        print(f"  {rel_type}: {value:.6f}")

    print("\n[role entities]")
    role_rows = sorted(
        role_entities.values(),
        key=lambda item: (
            len(item["rel_types"]),
            max(
                float(item["hub"]),
                float(item["authority"]),
                float(item["bridge"]),
                float(item["central"]),
                float(item["importance"]),
            ),
            item["name"],
        ),
        reverse=True,
    )
    for row in role_rows[:12]:
        print(
            {
                "name": row["name"],
                "rel_types": sorted(row["rel_types"]),
                "hub": round(float(row["hub"]), 6),
                "authority": round(float(row["authority"]), 6),
                "bridge": round(float(row["bridge"]), 6),
                "central": round(float(row["central"]), 6),
                "importance": round(float(row["importance"]), 6),
            }
        )

    print("\n[bridge motifs]")
    for row in bridge_rows[:12]:
        print(
            {
                "name": row["name"],
                "rel_types": sorted(row["rel_types"]),
                "external_communities": sorted(row["external_communities"]),
                "betweenness": round(float(row["betweenness"]), 6),
                "closeness": round(float(row["closeness"]), 6),
            }
        )

    print("\n[connector motifs]")
    for (rel_type, motif), count in connector_motifs.most_common(12):
        print(
            {
                "rel_type": rel_type,
                "motif": motif,
                "count": count,
                "examples": connector_examples[(rel_type, motif)],
            }
        )

    print("\n[community patterns]")
    for (rel_type, size), count in community_patterns.most_common(12):
        print({"rel_type": rel_type, "size": size, "count": count})

    print("\n[meta edges]")
    for edge in meta_edges[:20]:
        print(edge)


if __name__ == "__main__":
    asyncio.run(main())

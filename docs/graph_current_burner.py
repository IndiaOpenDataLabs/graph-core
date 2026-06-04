#!/usr/bin/env python3
"""Burner for what the current derived/analytics layer surfaces.

Examples:
  PYTHONPATH=src .venv/bin/python docs/graph_current_burner.py rlm exception
  PYTHONPATH=src .venv/bin/python docs/graph_current_burner.py rlm architecture
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqlalchemy import select  # noqa: E402

from graph_core.database import AsyncSessionLocal  # noqa: E402
from graph_core.models.collection import Collection  # noqa: E402
from graph_core.services.graph.analytics import (  # noqa: E402
    analyze_collection_graph,
    build_collection_understanding,
)

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


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("collection")
    parser.add_argument("question_type", choices=sorted(QUESTION_TERMS))
    args = parser.parse_args()

    async with AsyncSessionLocal() as session:
        coll = await session.scalar(select(Collection).where(Collection.name == args.collection))
        if not coll:
            raise ValueError(f"Collection {args.collection!r} not found")

    terms = QUESTION_TERMS[args.question_type]
    analysis = await analyze_collection_graph(coll.id)
    understanding = build_collection_understanding(analysis)

    print(f"collection={coll.name} id={coll.id}")
    print(f"question_type={args.question_type}")
    print(
        "analysis_summary",
        {
            "communities": len(analysis["communities"]),
            "top_anchors": len(analysis["top_anchors"]),
            "bridge_nodes": len(analysis["bridge_nodes"]),
            "connector_paths": len(analysis["connector_paths"]),
            "derived_nodes": len(understanding["nodes"]),
            "derived_edges": len(understanding["edges"]),
            "derived_chunks": len(understanding["chunks"]),
        },
    )

    print("\n[top anchors matching question]")
    found = 0
    for item in analysis["top_anchors"]:
        if contains_any(item["name"], terms):
            found += 1
            print(item)
    if not found:
        print("(none)")

    print("\n[bridge nodes matching question]")
    found = 0
    for item in analysis["bridge_nodes"]:
        if contains_any(item["name"], terms):
            found += 1
            print(item)
    if not found:
        print("(none)")

    print("\n[communities matching question]")
    found = 0
    for community in analysis["communities"]:
        blob = " ".join(community.get("node_names") or [])
        if contains_any(blob, terms):
            found += 1
            print(community)
    if not found:
        print("(none)")

    print("\n[connector paths matching question]")
    found = 0
    for path in analysis["connector_paths"]:
        blob = " ".join(path.get("nodes") or [])
        if contains_any(blob, terms):
            found += 1
            print(path)
    if not found:
        print("(none)")

    print("\n[derived chunks matching question]")
    found = 0
    for chunk in understanding["chunks"]:
        if contains_any(chunk["content"], terms):
            found += 1
            print({"metadata": chunk["metadata"], "content": chunk["content"]})
    if not found:
        print("(none)")


if __name__ == "__main__":
    asyncio.run(main())

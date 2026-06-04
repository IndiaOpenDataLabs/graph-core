"""Build and persist the derived understanding layer for a collection graph.

Usage:
    uv run python -m graph_core.scripts.graph_understanding \
        <collection_id> <namespace_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from graph_core.services.graph import GraphService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build collection understanding")
    parser.add_argument("collection_id", type=uuid.UUID)
    parser.add_argument("namespace_id", type=uuid.UUID)
    parser.add_argument("--min-edge-strength", type=float, default=0.2)
    parser.add_argument("--min-community-size", type=int, default=2)
    parser.add_argument("--max-anchors", type=int, default=12)
    parser.add_argument("--max-path-depth", type=int, default=4)
    parser.add_argument("--max-connector-paths", type=int, default=20)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    service = GraphService()
    result = await service.build_collection_understanding(
        args.collection_id,
        args.namespace_id,
        min_edge_strength=args.min_edge_strength,
        min_community_size=args.min_community_size,
        max_anchors=args.max_anchors,
        max_path_depth=args.max_path_depth,
        max_connector_paths=args.max_connector_paths,
    )
    print(json.dumps(result, indent=2, sort_keys=False))


if __name__ == "__main__":
    asyncio.run(main())

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
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    service = GraphService()
    result = await service.build_collection_understanding(
        args.collection_id,
        args.namespace_id,
    )
    print(json.dumps(result, indent=2, sort_keys=False))


if __name__ == "__main__":
    asyncio.run(main())

"""Inspect structural communities and connector paths for a collection graph.

Usage:
    uv run python -m graph_core.scripts.graph_analysis <collection_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from graph_core.services.graph.analytics import analyze_collection_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a collection graph")
    parser.add_argument("collection_id", type=uuid.UUID)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    analysis = await analyze_collection_graph(
        args.collection_id,
    )
    print(json.dumps(analysis, indent=2, sort_keys=False))


if __name__ == "__main__":
    asyncio.run(main())

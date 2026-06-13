"""Rename FalkorDB graphs to the collection-name-based naming scheme."""

from __future__ import annotations

import asyncio

from graph_core.services.graph import GraphService


async def _main() -> None:
    service = GraphService()
    results = await service.migrate_all_collection_graph_names()
    for row in results:
        print(
            f"{row['collection_id']} {row['collection_name']} -> {row['graph_name']}"
        )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()

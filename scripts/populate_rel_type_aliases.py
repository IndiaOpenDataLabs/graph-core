"""Populate relationship_type_aliases from clustering results.

Reads /tmp/rel_type_clusters_avg_0.80.json (average-linkage at 0.80 threshold)
and inserts one row per (canonical_type, alias_type) pair per collection.

For each cluster, the longest rel_type is chosen as the canonical form
and all other members become aliases pointing to it.
"""

import json
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

DB_URL = "postgresql+asyncpg://graphcore:graphcore@localhost:5432/graphcore"


async def populate():
    engine = create_async_engine(DB_URL)

    with open("/tmp/rel_type_clusters_avg_0.80.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    clusters = data["clusters"]
    multi = [c for c in clusters if len(c) > 1]

    async with AsyncSession(engine) as session:
        # Get all collections
        result = await session.execute(text("SELECT id, name FROM collections"))
        collections = [(row[0], row[1]) for row in result]

        total = 0
        for coll_id, coll_name in collections:
            print(f"\nProcessing collection: {coll_name} ({len(multi)} clusters)")
            for cluster in multi:
                # Canonical = longest rel_type in cluster
                canonical = max(cluster, key=len)
                aliases = [a for a in cluster if a != canonical]
                
                if aliases:
                    values = [
                        {"collection_id": coll_id, "canonical_type": canonical, "alias_type": a}
                        for a in aliases
                    ]
                    stmt = text(
                        "INSERT INTO relationship_type_aliases (collection_id, canonical_type, alias_type) "
                        "VALUES (:collection_id, :canonical_type, :alias_type) "
                        "ON CONFLICT (collection_id, alias_type) DO NOTHING"
                    )
                    for v in values:
                        await session.execute(stmt, v)
                    total += len(values)
                    print(f"  Inserted {len(values)} aliases")

        await session.commit()

    print(f"\nTotal aliases inserted: {total}")
    await engine.dispose()


if __name__ == "__main__":
    import asyncio
    asyncio.run(populate())

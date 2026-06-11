"""Populate relationship_type_aliases from clustering results.

Reads /tmp/rel_type_clusters_avg_0.80.json (average-linkage at 0.80 threshold)
and inserts one row per (canonical_type, alias_type) pair per collection.

For each cluster, the rel_type with the most edges in graph_relationships
is chosen as the canonical form and all other members become aliases.
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

        # Get edge counts per rel_type across all collections
        count_result = await session.execute(text(
            "SELECT rel_type, COUNT(*) FROM graph_relationships GROUP BY rel_type"
        ))
        edge_counts = {r[0]: r[1] for r in count_result}
        print(f"Loaded edge counts for {len(edge_counts)} rel_types\n")

        total = 0
        for coll_id, coll_name in collections:
            print(f"\nProcessing collection: {coll_name} ({len(multi)} clusters)")
            for cluster in multi:
                # Canonical = most common rel_type in cluster (highest edge count)
                counts = [(rt, edge_counts.get(rt, 0)) for rt in cluster]
                counts.sort(key=lambda x: x[1], reverse=True)
                canonical = counts[0][0]
                aliases = [a for a, _ in counts[1:]]
                
                if aliases:
                    canonical_row = await session.execute(
                        text(
                            """
                            INSERT INTO graph_relationship_types (collection_id, canonical_type)
                            VALUES (:collection_id, :canonical_type)
                            ON CONFLICT (collection_id, canonical_type) DO UPDATE
                            SET canonical_type = EXCLUDED.canonical_type
                            RETURNING id
                            """
                        ),
                        {"collection_id": coll_id, "canonical_type": canonical},
                    )
                    relationship_type_id = canonical_row.scalar_one()
                    values = [
                        {
                            "collection_id": coll_id,
                            "relationship_type_id": relationship_type_id,
                            "canonical_type": canonical,
                            "alias_type": a,
                        }
                        for a in aliases
                    ]
                    stmt = text(
                        "INSERT INTO relationship_type_aliases "
                        "(collection_id, relationship_type_id, canonical_type, alias_type) "
                        "VALUES (:collection_id, :relationship_type_id, :canonical_type, :alias_type) "
                        "ON CONFLICT (collection_id, alias_type) DO NOTHING"
                    )
                    for v in values:
                        await session.execute(stmt, v)
                    total += len(values)
                    print(f"  Canonical: {canonical} ({counts[0][1]} edges), {len(aliases)} aliases")

        await session.commit()

    print(f"\nTotal aliases inserted: {total}")
    await engine.dispose()


if __name__ == "__main__":
    import asyncio
    asyncio.run(populate())

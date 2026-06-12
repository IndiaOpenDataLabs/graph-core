"""Retrieve top relationship candidates for a question without rel-type gating.

Usage:
    uv run python -m graph_core.scripts.relationship_burner \
        --collection vedas \
        --namespace-id <namespace_uuid> \
        --question "Explain the mechanism by which the Spark In The Mind acts as a kindling agent to manifest the internal seat of Inner Agni."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from sqlalchemy import select

from graph_core.database import AsyncSessionLocal
from graph_core.models.graph_rag import GraphEntity, GraphRelationship
from graph_core.scripts.query_burner import _resolve_collection
from graph_core.services.graph.query import graph_rag as query_logic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Burner for ungated relationship retrieval"
    )
    parser.add_argument("--collection", required=True, help="Collection UUID or name.")
    parser.add_argument(
        "--namespace-id",
        type=uuid.UUID,
        help="Namespace UUID. Required when --collection is a name.",
    )
    parser.add_argument("--question", required=True, help="Question/subquery to test.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of relationship candidates to return.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    collection = await _resolve_collection(args.collection, args.namespace_id)
    embedding_provider = await query_logic._resolve_embedding_provider(collection)
    query_embedding = await query_logic._embed_relationship_query(
        embedding_provider,
        args.question,
        rel_type=None,
    )
    rel_seeds = await query_logic._search_relationship_seeds(
        collection,
        query_embedding,
        top_k=args.top_k,
    )

    async with AsyncSessionLocal() as session:
        entity_rows = await session.execute(
            select(GraphEntity.id, GraphEntity.canonical_name).where(
                GraphEntity.collection_id == collection.id
            )
        )
        entity_name_by_id = {
            str(entity_id): canonical_name
            for entity_id, canonical_name in entity_rows.all()
        }

        relationship_rows: list[dict[str, object]] = []
        for rel_id, score in rel_seeds:
            rel = await session.get(GraphRelationship, uuid.UUID(rel_id))
            if rel is None:
                continue
            relationship_rows.append(
                {
                    "relationship_id": rel_id,
                    "score": score,
                    "source_id": str(rel.source_entity_id),
                    "source_name": entity_name_by_id.get(str(rel.source_entity_id)),
                    "target_id": str(rel.target_entity_id),
                    "target_name": entity_name_by_id.get(str(rel.target_entity_id)),
                    "stored_rel_type": rel.rel_type,
                    "weight": rel.weight,
                    "keywords": rel.keywords,
                }
            )

    print(
        json.dumps(
            {
                "collection": {
                    "id": str(collection.id),
                    "name": collection.name,
                    "namespace_id": str(collection.namespace_id),
                },
                "question": args.question,
                "top_relationship_candidates": relationship_rows,
            },
            indent=2,
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

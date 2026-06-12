"""Compare relationship-first endpoints against entity relevance for a question.

Usage:
    uv run python -m graph_core.scripts.rel_entity_compare_burner \
        --collection vedas \
        --namespace-id <namespace_uuid> \
        --question "Analyze the energetic depletion of the Ego and how its reduction allows the true essence of Ātmā to be revealed through Inner Agni."
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
        description="Relationship-first endpoint/entity comparison burner"
    )
    parser.add_argument("--collection", required=True, help="Collection UUID or name.")
    parser.add_argument(
        "--namespace-id",
        type=uuid.UUID,
        help="Namespace UUID. Required when --collection is a name.",
    )
    parser.add_argument("--question", required=True, help="Question/subquery to test.")
    parser.add_argument(
        "--rel-top-k",
        type=int,
        default=15,
        help="Number of relationship candidates to retrieve.",
    )
    parser.add_argument(
        "--entity-top-k",
        type=int,
        default=30,
        help="Number of entity candidates to retrieve.",
    )
    parser.add_argument(
        "--entity-threshold",
        type=float,
        default=0.5,
        help="Keep endpoint entities whose entity score is at least this threshold.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    collection = await _resolve_collection(args.collection, args.namespace_id)
    embedding_provider = await query_logic._resolve_embedding_provider(collection)

    relationship_query_embedding = await query_logic._embed_relationship_query(
        embedding_provider,
        args.question,
        rel_type=None,
    )
    entity_query_embedding = await query_logic._embed_entity_query(
        embedding_provider,
        args.question,
    )

    rel_seeds = await query_logic._search_relationship_seeds(
        collection,
        relationship_query_embedding,
        top_k=args.rel_top_k,
    )
    entity_candidates = await query_logic._top_entity_candidates(
        collection,
        entity_query_embedding,
        top_k=args.entity_top_k,
    )
    entity_score_by_name = {
        name.strip().lower(): score for name, _, score in entity_candidates
    }

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

        top_relationship_candidates: list[dict[str, object]] = []
        endpoint_summary: dict[str, dict[str, object]] = {}
        for rel_id, rel_score in rel_seeds:
            rel = await session.get(GraphRelationship, uuid.UUID(rel_id))
            if rel is None:
                continue
            source_id = str(rel.source_entity_id)
            target_id = str(rel.target_entity_id)
            source_name = entity_name_by_id.get(source_id)
            target_name = entity_name_by_id.get(target_id)
            top_relationship_candidates.append(
                {
                    "relationship_id": rel_id,
                    "rel_score": rel_score,
                    "source_id": source_id,
                    "source_name": source_name,
                    "target_id": target_id,
                    "target_name": target_name,
                    "stored_rel_type": rel.rel_type,
                    "weight": rel.weight,
                    "keywords": rel.keywords,
                }
            )
            for entity_id, entity_name in (
                (source_id, source_name),
                (target_id, target_name),
            ):
                if not entity_name:
                    continue
                existing = endpoint_summary.get(entity_id)
                entity_score = entity_score_by_name.get(entity_name.strip().lower())
                if existing is None:
                    endpoint_summary[entity_id] = {
                        "entity_id": entity_id,
                        "entity_name": entity_name,
                        "entity_score": entity_score,
                        "best_rel_score": rel_score,
                        "seed_relationship_ids": [rel_id],
                    }
                else:
                    existing["best_rel_score"] = max(
                        float(existing["best_rel_score"]),
                        rel_score,
                    )
                    seed_rel_ids = list(existing["seed_relationship_ids"])
                    if rel_id not in seed_rel_ids:
                        seed_rel_ids.append(rel_id)
                    existing["seed_relationship_ids"] = seed_rel_ids
                    if entity_score is not None:
                        current_score = existing.get("entity_score")
                        if current_score is None or entity_score > float(current_score):
                            existing["entity_score"] = entity_score

        endpoint_entities = sorted(
            endpoint_summary.values(),
            key=lambda row: (
                row["entity_score"] is not None,
                float(row["entity_score"] or 0.0),
                float(row["best_rel_score"]),
            ),
            reverse=True,
        )
        kept_entity_ids = {
            str(row["entity_id"])
            for row in endpoint_entities
            if row["entity_score"] is not None
            and float(row["entity_score"]) >= args.entity_threshold
        }

        relationships_among_kept: list[dict[str, object]] = []
        if kept_entity_ids:
            kept_rel_rows = await session.execute(
                select(GraphRelationship).where(
                    GraphRelationship.collection_id == collection.id,
                    GraphRelationship.source_entity_id.in_(
                        [uuid.UUID(entity_id) for entity_id in kept_entity_ids]
                    ),
                    GraphRelationship.target_entity_id.in_(
                        [uuid.UUID(entity_id) for entity_id in kept_entity_ids]
                    ),
                )
            )
            for rel in kept_rel_rows.scalars().all():
                rel_id = str(rel.id)
                relationships_among_kept.append(
                    {
                        "relationship_id": rel_id,
                        "source_id": str(rel.source_entity_id),
                        "source_name": entity_name_by_id.get(str(rel.source_entity_id)),
                        "target_id": str(rel.target_entity_id),
                        "target_name": entity_name_by_id.get(str(rel.target_entity_id)),
                        "stored_rel_type": rel.rel_type,
                        "weight": rel.weight,
                        "keywords": rel.keywords,
                        "was_seed_hit": any(
                            candidate["relationship_id"] == rel_id
                            for candidate in top_relationship_candidates
                        ),
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
                "thresholds": {
                    "rel_top_k": args.rel_top_k,
                    "entity_top_k": args.entity_top_k,
                    "entity_threshold": args.entity_threshold,
                },
                "top_entity_candidates": [
                    {
                        "name": name,
                        "entity_score": score,
                        "description": description,
                    }
                    for name, description, score in entity_candidates
                ],
                "top_relationship_candidates": top_relationship_candidates,
                "endpoint_entities": endpoint_entities,
                "kept_entity_ids": sorted(kept_entity_ids),
                "relationships_among_kept": relationships_among_kept,
            },
            indent=2,
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

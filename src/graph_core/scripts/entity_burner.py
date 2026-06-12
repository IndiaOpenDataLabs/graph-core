"""Retrieve entity candidates for a question without dimension gating.

Usage:
    uv run python -m graph_core.scripts.entity_burner \
        --collection vedas \
        --namespace-id <namespace_uuid> \
        --question "Explain the mechanism by which the Spark In The Mind acts as a kindling agent to manifest the internal seat of Inner Agni."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from graph_core.scripts.query_burner import (
    _diagnostic_entity_text,
    _exact_entity_hits,
    _resolve_collection,
)
from graph_core.services.graph.query import graph_rag as query_logic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Burner for entity retrieval only")
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
        default=20,
        help="Number of entity candidates to return.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    collection = await _resolve_collection(args.collection, args.namespace_id)
    embedding_provider = await query_logic._resolve_embedding_provider(collection)
    query_embedding = await query_logic._embed_entity_query(
        embedding_provider,
        args.question,
    )
    diagnostic_entity_text = _diagnostic_entity_text(args.question)
    exact_hits = await _exact_entity_hits(collection.id, diagnostic_entity_text)
    top_candidates = await query_logic._top_entity_candidates(
        collection,
        query_embedding,
        top_k=args.top_k,
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
                "diagnostic_entity_text": diagnostic_entity_text,
                "exact_db_hits": exact_hits,
                "top_entity_candidates": [
                    {
                        "name": name,
                        "score": score,
                        "description": description,
                    }
                    for name, description, score in top_candidates
                ],
            },
            indent=2,
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

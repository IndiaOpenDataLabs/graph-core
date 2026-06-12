"""Debug the current graph query pipeline for a collection/question pair.

Usage:
    uv run python -m graph_core.scripts.query_burner \
        --collection vedas \
        --namespace-id <namespace_uuid> \
        --question "What is inner agni?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import asdict
from typing import Any

from sqlalchemy import or_, select

from graph_core.database import AsyncSessionLocal
from graph_core.models.collection import Collection
from graph_core.models.graph_rag import EntityAlias, EntityDescription, GraphEntity
from graph_core.services.graph.query import graph_rag as query_logic


def _diagnostic_entity_text(question: str) -> str:
    normalized = question.strip().strip("?!.,:; ").lower()
    prefixes = (
        "what is ",
        "who is ",
        "what are ",
        "who are ",
        "tell me about ",
        "explain ",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            candidate = normalized[len(prefix):].strip()
            if candidate:
                return candidate
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug current graph query logic")
    parser.add_argument(
        "--collection",
        required=True,
        help="Collection UUID or collection name.",
    )
    parser.add_argument(
        "--namespace-id",
        type=uuid.UUID,
        help="Namespace UUID. Required when --collection is a name.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to run through the current graph query logic.",
    )
    parser.add_argument(
        "--mode",
        default="mix",
        help="Query mode to debug. Defaults to mix.",
    )
    parser.add_argument(
        "--llm-profile-id",
        type=uuid.UUID,
        help="Optional LLM profile override.",
    )
    parser.add_argument(
        "--skip-final-answer",
        action="store_true",
        help="Stop after retrieval diagnostics instead of generating the final answer.",
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Skip the second-pass artifact build and print only retrieval diagnostics.",
    )
    return parser.parse_args()


async def _resolve_collection(
    collection_ref: str,
    namespace_id: uuid.UUID | None,
) -> Collection:
    async with AsyncSessionLocal() as session:
        try:
            collection_id = uuid.UUID(collection_ref)
        except ValueError:
            collection_id = None

        if collection_id is not None:
            collection = await session.get(Collection, collection_id)
            if collection is None:
                raise ValueError(f"Collection {collection_id} not found")
            return collection

        if namespace_id is None:
            raise ValueError("--namespace-id is required when --collection is a name")

        result = await session.execute(
            select(Collection).where(
                Collection.namespace_id == namespace_id,
                Collection.name == collection_ref,
            )
        )
        collection = result.scalar_one_or_none()
        if collection is None:
            raise ValueError(
                f"Collection named {collection_ref!r} not found in namespace {namespace_id}"
            )
        return collection


async def _exact_entity_hits(
    collection_id: uuid.UUID,
    text: str,
) -> dict[str, list[dict[str, Any]]]:
    normalized = text.strip().lower()
    async with AsyncSessionLocal() as session:
        entity_rows = (
            await session.execute(
                select(
                    GraphEntity.id,
                    GraphEntity.canonical_name,
                    GraphEntity.primary_type,
                    GraphEntity.description_count,
                ).where(
                    GraphEntity.collection_id == collection_id,
                    GraphEntity.canonical_name.ilike(normalized),
                )
            )
        ).all()
        alias_rows = (
            await session.execute(
                select(
                    EntityAlias.id,
                    EntityAlias.entity_id,
                    EntityAlias.alias_name,
                ).where(
                    EntityAlias.collection_id == collection_id,
                    EntityAlias.alias_name.ilike(normalized),
                )
            )
        ).all()
        desc_rows = (
            await session.execute(
                select(
                    GraphEntity.id,
                    GraphEntity.canonical_name,
                    EntityDescription.description,
                )
                .join(
                    EntityDescription,
                    EntityDescription.entity_id == GraphEntity.id,
                )
                .where(
                    GraphEntity.collection_id == collection_id,
                    or_(
                        GraphEntity.canonical_name.ilike(normalized),
                        GraphEntity.id.in_(
                            select(EntityAlias.entity_id).where(
                                EntityAlias.collection_id == collection_id,
                                EntityAlias.alias_name.ilike(normalized),
                            )
                        ),
                    ),
                )
            )
        ).all()

    return {
        "entities": [
            {
                "id": str(entity_id),
                "canonical_name": canonical_name,
                "primary_type": primary_type,
                "description_count": description_count,
            }
            for entity_id, canonical_name, primary_type, description_count in entity_rows
        ],
        "aliases": [
            {
                "id": str(alias_id),
                "entity_id": str(entity_id),
                "alias_name": alias_name,
            }
            for alias_id, entity_id, alias_name in alias_rows
        ],
        "descriptions": [
            {
                "entity_id": str(entity_id),
                "canonical_name": canonical_name,
                "description": description,
            }
            for entity_id, canonical_name, description in desc_rows
        ],
    }


async def main() -> None:
    args = parse_args()
    print("Resolving collection...", file=sys.stderr)
    collection = await _resolve_collection(args.collection, args.namespace_id)
    namespace_id = collection.namespace_id
    mode = query_logic._MODE_ALIASES.get((args.mode or "mix").lower(), "mix")

    print("Resolving embedding provider...", file=sys.stderr)
    embedding_provider = await query_logic._resolve_embedding_provider(collection)
    print("Embedding entity query...", file=sys.stderr)
    entity_query_embedding = await query_logic._embed_entity_query(
        embedding_provider,
        args.question,
    )
    print("Embedding relationship query...", file=sys.stderr)
    relationship_query_embedding = await query_logic._embed_relationship_query(
        embedding_provider,
        args.question,
        rel_type=None,
    )
    print("Checking exact DB hits...", file=sys.stderr)
    diagnostic_entity_text = _diagnostic_entity_text(args.question)
    exact_hits = await _exact_entity_hits(collection.id, diagnostic_entity_text)
    print("Ranking dimensions...", file=sys.stderr)
    active_dimensions = await query_logic._active_dimensions(collection)
    ranked_dimensions = await query_logic._rank_dimensions(
        collection,
        embedding_provider,
        args.question,
        active_dimensions,
    )
    print("Fetching top entity candidates...", file=sys.stderr)
    top_candidates = await query_logic._top_entity_candidates(
        collection,
        entity_query_embedding,
        top_k=20,
    )
    top_entity_score = top_candidates[0][2] if top_candidates else 0.0

    llm_provider = await query_logic._resolve_llm_provider(
        namespace_id=namespace_id,
        llm_profile_id=args.llm_profile_id or collection.llm_profile_id,
    )
    mix_interpretation = None
    subquery_states: list[dict[str, Any]] = []
    if mode == "mix" and top_candidates and top_entity_score >= query_logic._MIX_REWRITE_MIN_SCORE:
        print("Resolving LLM provider for mix interpretation...", file=sys.stderr)
        mix_interpretation = await query_logic._interpret_mix_queries(
            args.question,
            top_candidates,
            llm_provider,
        )
        rel_type_for_sub = None
        if mix_interpretation.retrieval_subqueries:
            print("Building subquery states...", file=sys.stderr)
            subquery_embeddings = await query_logic._embed_relationship_queries_batch(
                embedding_provider,
                mix_interpretation.retrieval_subqueries,
                [rel_type_for_sub] * len(mix_interpretation.retrieval_subqueries),
            )
            for subquery, embedding in zip(
                mix_interpretation.retrieval_subqueries,
                subquery_embeddings,
            ):
                state = await query_logic._relationship_seed_state(
                    collection,
                    embedding,
                    top_k=8,
                    max_endpoints=16,
                    max_pairs=20,
                    query_tokens=query_logic._query_token_set(args.question)
                    | query_logic._query_token_set(subquery),
                    rel_types=None,
                    dimension_weight=1.0,
                )
                subquery_states.append(
                    {
                        "subquery": subquery,
                        "discovered_entity_ids": sorted(state.discovered_entity_ids),
                        "entity_relevance": state.entity_relevance,
                        "traversed_rel_ids": state.traversed_rel_ids[:20],
                    }
                )

    artifacts = None
    if not args.skip_artifacts:
        print("Building graph query artifacts...", file=sys.stderr)
        artifacts = await query_logic._build_graph_query_artifacts(
            question=args.question,
            collection=collection,
            namespace_id=namespace_id,
            mode=mode,
            llm_profile_id=args.llm_profile_id or collection.llm_profile_id,
        )
    final_result = None
    if not args.skip_final_answer:
        print("Generating final answer...", file=sys.stderr)
        final_result = await query_logic.graph_rag_query(
            question=args.question,
            collection=collection,
            namespace_id=namespace_id,
            mode=mode,
            llm_profile_id=args.llm_profile_id or collection.llm_profile_id,
        )

    payload = {
        "collection": {
            "id": str(collection.id),
            "name": collection.name,
            "namespace_id": str(collection.namespace_id),
            "default_query_mode": collection.default_query_mode,
            "llm_profile_id": str(collection.llm_profile_id)
            if collection.llm_profile_id
            else None,
        },
        "question": args.question,
        "mode": mode,
        "mix_rewrite_min_score": query_logic._MIX_REWRITE_MIN_SCORE,
        "diagnostic_entity_text": diagnostic_entity_text,
        "exact_db_hits": exact_hits,
        "ranked_dimensions": ranked_dimensions,
        "top_entity_score": top_entity_score,
        "top_entity_candidates": [
            {
                "name": name,
                "score": score,
                "description": description,
            }
            for name, description, score in top_candidates
        ],
        "mix_interpretation": asdict(mix_interpretation)
        if mix_interpretation is not None
        else None,
        "mix_subquery_states": subquery_states,
        "route_profile": asdict(artifacts.route_profile)
        if artifacts is not None
        else None,
        "entities_used": artifacts.entities_used if artifacts is not None else [],
        "relationships_used": (
            artifacts.relationships_used if artifacts is not None else []
        ),
        "context_preview": artifacts.context[:8000] if artifacts is not None else None,
        "final_result": asdict(final_result) if final_result is not None else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    asyncio.run(main())

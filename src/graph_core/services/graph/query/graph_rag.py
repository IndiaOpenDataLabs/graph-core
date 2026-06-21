"""Graph RAG query functions extracted from GraphService."""

from __future__ import annotations

import asyncio
import logging
import string
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from sqlalchemy import distinct, or_, select

from graph_core.config import settings
from graph_core.database import AsyncSessionLocal
from graph_core.embedding import get_embedding_provider
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.llm import LocalEchoLLMProvider, get_llm_provider
from graph_core.llm.interface import LLMProvider
from graph_core.models.collection import Collection
from graph_core.models.credential import Credential
from graph_core.models.graph_rag import (
    EntityAlias,
    EntityDescription,
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
    RelationshipDescription,
    RelationshipTypeAlias,
)
from graph_core.models.profile import Profile
from graph_core.models.rel_types import (
    normalize_rel_type as normalize_dim,
)
from graph_core.models.rel_types import (
    relationship_embedding_text,
)
from graph_core.services.crypto import CredentialCrypto
from graph_core.services.graph.query.vector import QueryResult
from graph_core.storage.graph_names import collection_graph_name
from graph_core.storage.graph_rag_vectors import GraphRAGVectorStore
from graph_core.storage.meta_collections import (
    base_collection_name,
    meta_collection_level,
    meta_collection_name,
    parse_meta_collection_name,
)

_graph_rag_vectors = GraphRAGVectorStore()
_crypto = CredentialCrypto()
logger = logging.getLogger(__name__)
_ENTITY_RETRIEVAL_INSTRUCTION = (
    "Retrieve ontology entities whose descriptions best explain the user's "
    "state, process, causal mechanism, or source of exhaustion."
)
_RELATIONSHIP_RETRIEVAL_INSTRUCTION = (
    "Retrieve relationship descriptions that best explain the user's question, "
    "especially causes, mechanisms, tensions, and energy depletion."
)
_MIX_REWRITE_MIN_SCORE = 0.55
_REL_ENDPOINT_ENTITY_SCORE_MIN = 0.0
_MODE_ALIASES = {
    "local": "entity-first",
    "ent": "entity-first",
    "entity": "entity-first",
    "entity-first": "entity-first",
    "rel": "relationship-first",
    "relationship": "relationship-first",
    "relationship-first": "relationship-first",
    "hyb": "hybrid",
    "hybrid": "hybrid",
    "mix": "mix",
}
_MAX_QUERY_DIMENSIONS = 25

async def _active_dimensions(collection: Collection) -> list[str]:
    """Active graph dimensions for this collection, in priority order.

    Falls back to the configured subset when the operator has pinned one.
    Otherwise uses only rel_types that actually exist in the collection,
    avoiding a fan-out across every domain's vocabulary on each query.
    """
    configured = list(settings.graph_rag_active_dimensions or [])
    if configured:
        return [normalize_dim(d) for d in configured]
    from graph_core.models.rel_types import DEFAULT_REL_TYPE

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(
                distinct(GraphRelationshipType.canonical_type)
            )
            .select_from(GraphRelationship)
            .join(
                GraphRelationshipType,
                GraphRelationshipType.id == GraphRelationship.relationship_type_id,
            )
            .where(GraphRelationship.collection_id == collection.id)
        )
        found = [
            normalize_dim(str(value))
            for value in result.scalars().all()
            if value is not None and str(value).strip()
        ]
    types: list[str] = []
    for rel_type in found:
        if rel_type not in types:
            types.append(rel_type)
    if DEFAULT_REL_TYPE not in types:
        types.insert(0, DEFAULT_REL_TYPE)
    return types


async def _load_rel_type_alias_map(collection_id: uuid.UUID) -> dict[str, str]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(
                RelationshipTypeAlias.alias_type,
                RelationshipTypeAlias.canonical_type,
            ).where(RelationshipTypeAlias.collection_id == collection_id)
        )
        return {
            normalize_dim(alias_type): normalize_dim(canonical_type)
            for alias_type, canonical_type in result.all()
            if alias_type and canonical_type
        }


def _canonicalize_rel_type(
    rel_type: str | None,
    alias_map: dict[str, str] | None = None,
) -> str:
    normalized = normalize_dim(rel_type)
    if not alias_map:
        return normalized
    return alias_map.get(normalized, normalized)


def _dimension_weight(rel_type: str | None) -> float:
    weights = settings.graph_rag_dimension_weights or {}
    if not rel_type:
        return 1.0
    return float(weights.get(rel_type, 1.0))


async def _fan_out_per_dimension(
    build_state, dimensions: list[str] | None = None
) -> GraphQueryState | None:
    """Run ``build_state(rel_type)`` once per active dimension and merge.

    Each dimension is traversed as an independent sub-graph: a node may
    appear in multiple dimensions with different neighbours and the
    merged ``GraphQueryState`` keeps the highest per-node and per-edge
    scores across them. The result is the same shape as a single-dim
    traversal, so downstream context assembly is dimension-agnostic.
    """
    dims = dimensions if dimensions is not None else []
    if not dims:
        return await build_state(None)

    concurrency = settings.graph_rag_query_embedding_concurrency
    sem = asyncio.Semaphore(concurrency)

    async def _gated(rel_type: str):
        async with sem:
            return await build_state(rel_type)

    results = await asyncio.gather(*(_gated(rel_type) for rel_type in dims))
    states = [state for state in results if state is not None]
    if not states:
        return None
    return _merge_states(*states)


async def _rank_dimensions(
    collection: Collection,
    embedding_provider: EmbeddingProvider,
    question: str,
    dimensions: list[str],
    *,
    top_k: int = _MAX_QUERY_DIMENSIONS,
) -> list[str]:
    if len(dimensions) <= top_k:
        return dimensions

    dimension_set = set(dimensions)

    # Step 1: Find top entities relevant to the question (across all rel_types).
    entity_query_embedding = await _embed_entity_query(
        embedding_provider, question
    )
    seed_entity_ids, entity_relevance = await _search_entity_seeds(
        question, collection, entity_query_embedding
    )
    top_entities = seed_entity_ids[:10]

    if not top_entities:
        return dimensions[:top_k]

    # Step 2: Collect rel_types from edges incident to the top entities.
    graph_storage = get_graph_storage(collection)
    rel_type_alias_map = await _load_rel_type_alias_map(collection.id)
    rel_type_counts: dict[str, int] = defaultdict(int)

    for entity_id in top_entities:
        edges = await graph_storage.get_node_edges_with_types(entity_id)
        for _, _, rel_type in edges:
            normalized = _canonicalize_rel_type(rel_type, rel_type_alias_map)
            if normalized in dimension_set:
                rel_type_counts[normalized] += 1

    # Step 3: Find paths between pairs of top entities and collect rel_types
    # from those paths. This captures the "highway" rel_types that connect
    # relevant entities.
    rel_query_embedding = await _embed_relationship_query(
        embedding_provider, question, rel_type=None
    )
    query_tokens = _query_token_set(question)
    rel_score_cache: dict[str, float] = {}

    pair_count = 0
    max_pairs = 20
    for source_id, target_id in combinations(top_entities, 2):
        if pair_count >= max_pairs:
            break
        pair_count += 1

        path = await _find_relevant_path_for_ranking(
            graph_storage,
            collection,
            rel_query_embedding,
            source_id,
            target_id,
            rel_score_cache,
            query_tokens=query_tokens,
        )
        if not path:
            continue
        _, path_rels = path
        for rel_id in path_rels:
            rel_type_from_id = await _get_rel_type_from_id(rel_id)
            if rel_type_from_id:
                normalized = _canonicalize_rel_type(
                    rel_type_from_id,
                    rel_type_alias_map,
                )
                if normalized in dimension_set:
                    rel_type_counts[normalized] += 2

    if not rel_type_counts:
        return dimensions[:top_k]

    # Step 4: Rank candidates by graph-grounded frequency, then score with
    # vector search for fine-grained ranking.
    sorted_by_count = sorted(
        rel_type_counts.items(), key=lambda x: x[1], reverse=True
    )
    candidate_dims = [rt for rt, _ in sorted_by_count[:50]]

    embeddings = await _embed_relationship_queries_batch(
        embedding_provider,
        [question] * len(candidate_dims),
        candidate_dims,
    )

    async def _score_dimension(
        rel_type: str, embedding: list[float]
    ) -> tuple[str, float, float]:
        hits = await _graph_rag_vectors.search_relationship_embeddings(
            collection_id=collection.id,
            query_embedding=embedding,
            top_k=3,
        )
        sims = [1.0 - hit.distance for hit in hits]
        top1 = sims[0] if sims else 0.0
        top3_mean = sum(sims[:3]) / min(len(sims), 3) if sims else 0.0
        return rel_type, top1 + top3_mean, top1

    scored = await asyncio.gather(
        *(_score_dimension(rt, emb) for rt, emb in zip(candidate_dims, embeddings))
    )
    scored.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
    selected = [rel_type for rel_type, _, _ in scored[:top_k]]
    logger.info(
        "graph_rag dimension gating collection=%s total=%d entities=%d edge_rels=%d path_rels=%d candidates=%d selected=%d top=%s",
        collection.name,
        len(dimensions),
        len(top_entities),
        sum(1 for _, c in rel_type_counts.items() if c > 0),
        pair_count,
        len(candidate_dims),
        len(selected),
        [(rel_type, round(score, 6)) for rel_type, score, _ in scored[:top_k]],
    )
    return selected


async def _get_rel_type_from_id(rel_id: str) -> str | None:
    """Look up the canonical rel_type for a given relationship ID from the DB."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(GraphRelationshipType.canonical_type)
            .select_from(GraphRelationship)
            .join(
                GraphRelationshipType,
                GraphRelationshipType.id == GraphRelationship.relationship_type_id,
            )
            .where(GraphRelationship.id == uuid.UUID(rel_id))
        )
        return result.scalar_one_or_none()


async def _find_relevant_path_for_ranking(
    graph_storage,
    collection: Collection,
    relationship_query_embedding: list[float],
    source_id: str,
    target_id: str,
    rel_score_cache: dict[str, float],
    *,
    max_depth: int = 3,
    beam_width: int = 4,
    query_tokens: set[str] | None = None,
) -> tuple[list[str], list[str]] | None:
    """Beam-search path between two entities, collecting all rel_type edges.

    Variant of _find_relevant_path that doesn't filter by rel_types,
    used during dimension ranking to discover relevant relationship types.
    """
    if query_tokens is None:
        query_tokens = set()

    queue: deque[tuple[str, list[str], list[str], int]] = deque(
        [(source_id, [source_id], [], 0)]
    )

    while queue:
        node_id, path_nodes, path_rels, depth = queue.popleft()
        if depth >= max_depth:
            continue

        edges = await graph_storage.get_node_edges_with_types(node_id)
        candidates: list[tuple[float, str, str]] = []
        for src, tgt, rel_type in edges:
            neighbor = tgt if src == node_id else src
            if neighbor in path_nodes:
                continue

            edge_props = await graph_storage.get_edge(src, tgt)
            if not edge_props:
                edge_props = await graph_storage.get_edge(tgt, src)
            if not (edge_props and edge_props.get("id")):
                continue

            rel_id = str(edge_props["id"])
            sim = await _score_relationship(
                collection,
                relationship_query_embedding,
                rel_id,
                rel_score_cache,
            )
            combined = _combined_edge_score(sim, edge_props, query_tokens)
            candidates.append((combined, neighbor, rel_id))

        for _, neighbor, rel_id in sorted(candidates, reverse=True)[:beam_width]:
            next_nodes = path_nodes + [neighbor]
            next_rels = path_rels + [rel_id]
            if neighbor == target_id:
                return next_nodes, next_rels
            queue.append((neighbor, next_nodes, next_rels, depth + 1))

    return None


@dataclass
class GraphQueryState:
    discovered_entity_ids: set[str]
    entity_relevance: dict[str, float]
    traversed_rel_ids: list[str]
    rel_score_cache: dict[str, float]
    rel_combined_score_cache: dict[str, float]


@dataclass
class MixInterpretation:
    selected_entities: list[str]
    retrieval_subqueries: list[str]


@dataclass
class DerivedRouteProfile:
    primary_route: str
    route_scores: dict[str, float]
    rel_type_scores: dict[str, float]


@dataclass
class GraphQueryArtifacts:
    context: str
    entities_used: list[str]
    relationships_used: list[str]
    rel_context: str
    route_profile: DerivedRouteProfile


@dataclass
class DocumentRoutingDecision:
    use_all_documents: bool
    document_ids: list[uuid.UUID]
    

@dataclass
class DocumentRoutingCandidate:
    document_id: str
    document_path: str
    best_score: float
    matched_entities: list[str]


def _format_retrieval_query(instruction: str, query: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}"


async def _embed_entity_query(
    embedding_provider: EmbeddingProvider,
    query: str,
) -> list[float]:
    return await embedding_provider.embed_query(
        _format_retrieval_query(_ENTITY_RETRIEVAL_INSTRUCTION, query)
    )


async def _embed_relationship_query(
    embedding_provider: EmbeddingProvider,
    query: str,
    rel_type: str | None = None,
) -> list[float]:
    focus = ""
    if rel_type:
        focus = (
            f" Focus on relationships whose semantic role is {normalize_dim(rel_type)}."
        )
    return await embedding_provider.embed_query(
        _format_retrieval_query(
            _RELATIONSHIP_RETRIEVAL_INSTRUCTION + focus,
            relationship_embedding_text(
                source_name="user-question",
                target_name="graph-answer",
                rel_type=rel_type,
                description=query,
                keywords=[],
            ),
        )
    )


async def _embed_relationship_queries_batch(
    embedding_provider: EmbeddingProvider,
    queries: list[str],
    rel_types: list[str | None],
) -> list[list[float]]:
    """Embed multiple relationship queries in a single API call.

    Each query gets its own rel_type focus prefix, but all are sent to
    the embedding model in one batched request, reducing round-trips.
    """
    texts = []
    for query, rel_type in zip(queries, rel_types):
        focus = ""
        if rel_type:
            focus = (
                f" Focus on relationships whose semantic role is {normalize_dim(rel_type)}."
            )
        texts.append(
            _format_retrieval_query(
                _RELATIONSHIP_RETRIEVAL_INSTRUCTION + focus,
                relationship_embedding_text(
                    source_name="user-question",
                    target_name="graph-answer",
                    rel_type=rel_type,
                    description=query,
                    keywords=[],
                ),
            )
        )
    return await embedding_provider.embed_documents(texts)


async def _resolve_credential(
    session, profile: Profile
) -> tuple[str | None, str | None]:
    if profile.credential_id is None:
        return None, None
    credential = await session.get(Credential, profile.credential_id)
    if not credential:
        raise ValueError(f"Credential {profile.credential_id} not found")
    return _crypto.decrypt(credential.encrypted_secret), credential.base_url


async def _resolve_embedding_provider(collection: Collection) -> EmbeddingProvider:
    if collection.embedding_profile_id is None:
        return get_embedding_provider()
    async with AsyncSessionLocal() as session:
        profile = await session.get(Profile, collection.embedding_profile_id)
        if not profile:
            raise ValueError(
                f"Embedding profile {collection.embedding_profile_id} not found"
            )
        api_key, cred_base_url = await _resolve_credential(session, profile)
        base_url = profile.base_url or cred_base_url
        return get_embedding_provider(
            provider_name=profile.provider,
            model=profile.model,
            dimensions=profile.dimensions,
            api_key=api_key,
            base_url=base_url,
            profile_id=str(profile.id),
            max_concurrent_calls=profile.max_concurrent_calls,
        )


async def _resolve_llm_provider(
    namespace_id: uuid.UUID,
    llm_profile_id: uuid.UUID | None = None,
) -> LLMProvider:
    if llm_profile_id is None:
        return get_llm_provider()
    async with AsyncSessionLocal() as session:
        profile = await session.get(Profile, llm_profile_id)
        if not profile or profile.namespace_id != namespace_id:
            raise ValueError("LLM profile not found in namespace")
        if profile.kind != "llm":
            raise ValueError("Profile kind must be llm")
        api_key, cred_base_url = await _resolve_credential(session, profile)
        base_url = profile.base_url or cred_base_url
        return get_llm_provider(
            provider_name=profile.provider,
            model=profile.model,
            api_key=api_key,
            base_url=base_url,
            profile_id=str(profile.id),
            max_concurrent_calls=profile.max_concurrent_calls,
        )


def get_graph_storage(collection: Collection):
    from graph_core.storage.graph_storage import FalkorDBGraphStorage

    graph_name = collection_graph_name(
        namespace_id=collection.namespace_id,
        collection_id=collection.id,
        collection_name=collection.name,
    )
    return FalkorDBGraphStorage(
        graph_name,
        namespace_id=collection.namespace_id,
    )


def _extract_query_keywords(question: str) -> list[str]:
    stop_words = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to",
        "for", "of", "with", "is", "what", "how", "why", "who",
        "i", "me", "my", "can", "be", "when",
    }
    tokens = [w.strip(string.punctuation).lower() for w in question.split()]
    keywords = [w for w in tokens if w and w not in stop_words and len(w) > 2]
    return list(dict.fromkeys([question] + keywords))


def _query_token_set(question: str) -> set[str]:
    return {w.lower() for w in _extract_query_keywords(question)}


def _parse_source_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            import json

            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [stripped]
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        return [stripped]
    return []


def _maybe_uuid_string(value: str) -> str | None:
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError):
        return None


async def _load_document_candidates_from_query_entities(
    question: str,
    collection: Collection,
    embedding_provider: EmbeddingProvider,
    *,
    limit: int = 20,
) -> list[DocumentRoutingCandidate]:
    query_embedding = await _embed_entity_query(embedding_provider, question)
    entity_hits = await _graph_rag_vectors.search_entity_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=50,
    )

    candidate_map: dict[str, DocumentRoutingCandidate] = {}
    for hit in entity_hits:
        document_id = str(hit.metadata.get("document_id") or "").strip()
        if not document_id:
            continue
        document_path = str(hit.metadata.get("document_path") or "").strip()
        entity_name = str(hit.metadata.get("name") or "").strip()
        score = 1.0 - float(hit.distance)
        candidate = candidate_map.get(document_id)
        if candidate is None:
            candidate_map[document_id] = DocumentRoutingCandidate(
                document_id=document_id,
                document_path=document_path,
                best_score=score,
                matched_entities=[entity_name] if entity_name else [],
            )
            continue
        if score > candidate.best_score:
            candidate.best_score = score
        if not candidate.document_path and document_path:
            candidate.document_path = document_path
        if entity_name and entity_name not in candidate.matched_entities:
            candidate.matched_entities.append(entity_name)

    candidates = sorted(
        candidate_map.values(),
        key=lambda candidate: candidate.best_score,
        reverse=True,
    )
    return candidates[:limit]


async def _resolve_document_routing(
    question: str,
    collection: Collection,
    namespace_id: uuid.UUID,
    llm_profile_id: uuid.UUID | None,
) -> DocumentRoutingDecision:
    embedding_provider = await _resolve_embedding_provider(collection)
    candidates = await _load_document_candidates_from_query_entities(
        question,
        collection,
        embedding_provider,
    )
    if not candidates:
        return DocumentRoutingDecision(use_all_documents=True, document_ids=[])

    llm_provider = await _resolve_llm_provider(
        namespace_id=namespace_id,
        llm_profile_id=llm_profile_id,
    )
    if isinstance(llm_provider, LocalEchoLLMProvider):
        return DocumentRoutingDecision(use_all_documents=True, document_ids=[])

    candidate_lines = []
    for index, candidate in enumerate(candidates[:20], start=1):
        label = candidate.document_path or candidate.document_id
        entity_summary = ", ".join(candidate.matched_entities[:5]) or "none"
        candidate_lines.append(
            f"{index}. {label} | document_id={candidate.document_id} | "
            f"entities={entity_summary} | score={candidate.best_score:.3f}"
        )

    schema = {
        "type": "object",
        "properties": {
            "route": {
                "type": "string",
                "enum": ["all", "documents"],
            },
            "document_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["route", "document_ids"],
    }

    prompt = (
        "You are routing a knowledge-graph question to the most relevant source documents.\n"
        "Choose 'documents' only when the question clearly refers to one or more specific files, sources, or document subsets.\n"
        "Otherwise choose 'all'.\n"
        "Return only document_ids from the candidate list.\n"
        "If the question is broad, comparative across the whole collection, or the target document is unclear, choose all.\n\n"
        f"User question:\n{question}\n\n"
        "Candidate documents:\n"
        f"{chr(10).join(candidate_lines)}"
    )

    try:
        result = await llm_provider.structured_extract(prompt=prompt, schema=schema)
    except Exception:
        return DocumentRoutingDecision(use_all_documents=True, document_ids=[])

    route = str(result.get("route") or "all").strip().lower()
    selected_ids: list[uuid.UUID] = []
    if route == "documents":
        allowed_ids = {candidate.document_id for candidate in candidates}
        for value in result.get("document_ids", []):
            value_str = _maybe_uuid_string(str(value))
            if value_str and value_str in allowed_ids:
                try:
                    selected_ids.append(uuid.UUID(value_str))
                except ValueError:
                    continue
        selected_ids = list(dict.fromkeys(selected_ids))
    if not selected_ids:
        return DocumentRoutingDecision(use_all_documents=True, document_ids=[])
    return DocumentRoutingDecision(use_all_documents=False, document_ids=selected_ids)


async def _search_entity_seeds(
    question: str,
    collection: Collection,
    query_embedding: list[float],
    document_ids: list[uuid.UUID] | None = None,
) -> tuple[list[str], dict[str, float]]:
    top_k = 20
    seed_entity_ids: list[str] = []
    entity_relevance: dict[str, float] = {}

    entity_hits = await _graph_rag_vectors.search_entity_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=top_k,
        document_ids=document_ids,
    )
    for hit in entity_hits:
        entity_id_str = hit.metadata.get("entity_id", "")
        sim = 1.0 - hit.distance
        if entity_id_str and entity_id_str not in seed_entity_ids:
            seed_entity_ids.append(entity_id_str)
            entity_relevance[entity_id_str] = sim

    keywords = _extract_query_keywords(question)
    async with AsyncSessionLocal() as session:
        for kw in keywords[:5]:
            conditions = [
                or_(
                    EntityAlias.alias_name.ilike(f"% {kw} %"),
                    EntityAlias.alias_name.ilike(f"{kw} %"),
                    EntityAlias.alias_name.ilike(f"% {kw}"),
                    EntityAlias.alias_name.ilike(kw),
                ),
                EntityAlias.collection_id == collection.id,
                GraphEntity.collection_id == collection.id,
            ]
            if document_ids:
                conditions.append(EntityAlias.document_id.in_(document_ids))
            alias_result = await session.execute(
                select(EntityAlias)
                .join(GraphEntity, GraphEntity.id == EntityAlias.entity_id)
                .where(*conditions)
                .limit(5)
            )
            for alias in alias_result.scalars().all():
                eid = str(alias.entity_id)
                if eid not in seed_entity_ids:
                    seed_entity_ids.append(eid)
                    entity_relevance[eid] = 1.0

    return seed_entity_ids, entity_relevance


async def _top_entity_candidates(
    collection: Collection,
    query_embedding: list[float],
    *,
    top_k: int = 50,
    document_ids: list[uuid.UUID] | None = None,
) -> list[tuple[str, str, float]]:
    hits = await _graph_rag_vectors.search_entity_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=top_k,
        document_ids=document_ids,
    )
    candidates: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for hit in hits:
        name = str(hit.metadata.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        candidates.append((name, hit.content.strip(), 1.0 - hit.distance))
    logger.info(
        "graph_rag top_entity_candidates collection=%s count=%d top=%s",
        collection.name,
        len(candidates),
        [(name, round(score, 6)) for name, _, score in candidates[:10]],
    )
    return candidates


async def _search_relationship_seeds(
    collection: Collection,
    query_embedding: list[float],
    *,
    top_k: int = 10,
    document_ids: list[uuid.UUID] | None = None,
) -> list[tuple[str, float]]:
    hits = await _graph_rag_vectors.search_relationship_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=top_k,
        document_ids=document_ids,
    )
    rel_seeds: list[tuple[str, float]] = []
    seen: set[str] = set()
    for hit in hits:
        rel_id = hit.metadata.get("relationship_id") or hit.metadata.get("id")
        if not rel_id or rel_id in seen:
            continue
        seen.add(rel_id)
        rel_seeds.append((str(rel_id), 1.0 - hit.distance))
    return rel_seeds


async def _score_relationship(
    collection: Collection,
    relationship_query_embedding: list[float],
    rel_id: str,
    cache: dict[str, float],
    *,
    top_k: int = 4,
    document_ids: list[uuid.UUID] | None = None,
) -> float:
    cached = cache.get(rel_id)
    if cached is not None:
        return cached
    rel_hits = await _graph_rag_vectors.search_relationship_embeddings(
        collection_id=collection.id,
        query_embedding=relationship_query_embedding,
        top_k=top_k,
        relationship_id=uuid.UUID(rel_id),
        document_ids=document_ids,
    )
    score = max((1.0 - hit.distance for hit in rel_hits), default=0.0)
    cache[rel_id] = score
    return score


def _combined_edge_score(
    cos: float,
    edge_props: dict[str, Any] | None,
    query_tokens: set[str],
) -> float:
    weight_ratio = settings.graph_rag_edge_weight_score_ratio
    keyword_ratio = settings.graph_rag_keyword_score_ratio
    if not edge_props:
        return cos
    raw_weight = edge_props.get("weight")
    try:
        max_weight = max(1, int(settings.graph_rag_max_relationship_weight))
        weight_norm = min(int(raw_weight or 0), max_weight) / float(max_weight)
    except (TypeError, ValueError):
        weight_norm = 0.0
    kws = edge_props.get("keywords") or []
    if not isinstance(kws, list):
        kws = []
    if kws and query_tokens:
        norm_kws = [str(k).lower().strip() for k in kws if str(k).strip()]
        hits = sum(1 for k in norm_kws if k in query_tokens)
        kw_norm = hits / len(norm_kws)
    else:
        kw_norm = 0.0
    return (
        (1.0 - weight_ratio - keyword_ratio) * cos
        + weight_ratio * weight_norm
        + keyword_ratio * kw_norm
    )


async def _entity_first_state(
    question: str,
    collection: Collection,
    entity_query_embedding: list[float],
    relationship_query_embedding: list[float],
    *,
    rel_types: list[str] | None = None,
    dimension_weight: float = 1.0,
    document_ids: list[uuid.UUID] | None = None,
) -> GraphQueryState:
    top_k = 10
    min_edge_sim = settings.graph_rag_min_edge_similarity
    energy_budget = 7.0
    max_depth = 8
    query_tokens = _query_token_set(question)

    seed_entity_ids, entity_relevance = await _search_entity_seeds(
        question,
        collection,
        entity_query_embedding,
        document_ids=document_ids,
    )

    async with AsyncSessionLocal() as session:
        seed_entity_rows = await session.execute(
            select(GraphEntity).where(GraphEntity.collection_id == collection.id)
        )
        name_to_eid = {
            entity.canonical_name.lower(): str(entity.id)
            for entity in seed_entity_rows.scalars().all()
        }

    rel_hits = await _graph_rag_vectors.search_relationship_embeddings(
        collection_id=collection.id,
        query_embedding=relationship_query_embedding,
        top_k=max(top_k * 5, 50),
    )

    seed_rel_scores: dict[str, float] = {eid: 0.0 for eid in seed_entity_ids}
    best_seed_sim = max(entity_relevance.values()) if entity_relevance else 0.0
    if best_seed_sim < 0.25:
        effective_min_edge_sim = max(min_edge_sim, 0.5)
        effective_energy_budget = 2.5
    elif best_seed_sim < 0.4:
        effective_min_edge_sim = max(min_edge_sim, 0.4)
        effective_energy_budget = 4.0
    else:
        effective_min_edge_sim = min_edge_sim
        effective_energy_budget = energy_budget

    for hit in rel_hits:
        sim = 1.0 - hit.distance
        if sim < effective_min_edge_sim:
            continue
        for name_field in ("source_name", "target_name"):
            name = hit.metadata.get(name_field, "").lower()
            eid = name_to_eid.get(name)
            if eid and sim > seed_rel_scores.get(eid, 0.0):
                seed_rel_scores[eid] = sim

    graph_storage = get_graph_storage(collection)
    visited = set(seed_entity_ids)
    traversed_rel_ids: list[str] = []
    discovered_entity_ids = set(seed_entity_ids)
    rel_score_cache: dict[str, float] = {}
    rel_combined_score_cache: dict[str, float] = {}
    energy = effective_energy_budget

    sorted_seeds = sorted(seed_entity_ids, key=lambda e: seed_rel_scores.get(e, 0.0))
    stack = [(node_id, 0) for node_id in sorted_seeds]

    while stack and energy > 0:
        node_id, depth = stack.pop()
        if depth >= max_depth:
            continue

        edges = await graph_storage.get_node_edges(
            node_id,
            rel_types=rel_types,
            document_ids=[str(doc_id) for doc_id in document_ids]
            if document_ids
            else None,
        )
        scored_edges: list[tuple[float, str, str]] = []
        for src, tgt in edges:
            neighbor = tgt if src == node_id else src
            if neighbor in visited:
                continue

            edge_props = await graph_storage.get_edge(
                src,
                tgt,
                rel_types=rel_types,
                document_ids=[str(doc_id) for doc_id in document_ids]
                if document_ids
                else None,
            )
            if not edge_props:
                edge_props = await graph_storage.get_edge(
                    tgt,
                    src,
                    rel_types=rel_types,
                    document_ids=[str(doc_id) for doc_id in document_ids]
                    if document_ids
                    else None,
                )
            if not (edge_props and edge_props.get("id")):
                continue

            rel_id_str = str(edge_props["id"])
            sim = await _score_relationship(
                collection,
                relationship_query_embedding,
                rel_id_str,
                rel_score_cache,
            )
            combined = (
                _combined_edge_score(sim, edge_props, query_tokens)
                * dimension_weight
            )
            if combined >= effective_min_edge_sim:
                scored_edges.append((combined, neighbor, rel_id_str))

        for combined, neighbor, rel_id_str in sorted(
            scored_edges,
            key=lambda x: x[0],
            reverse=True,
        ):
            cost = max(0.05, 1.0 - combined)
            if energy - cost <= 0:
                continue
            energy -= cost
            visited.add(neighbor)
            stack.append((neighbor, depth + 1))
            discovered_entity_ids.add(neighbor)
            if rel_id_str not in traversed_rel_ids:
                traversed_rel_ids.append(rel_id_str)
            rel_combined_score_cache[rel_id_str] = combined
            if (
                neighbor not in entity_relevance
                or combined > entity_relevance[neighbor]
            ):
                entity_relevance[neighbor] = combined

    return GraphQueryState(
        discovered_entity_ids=discovered_entity_ids,
        entity_relevance=entity_relevance,
        traversed_rel_ids=traversed_rel_ids,
        rel_score_cache=rel_score_cache,
        rel_combined_score_cache=rel_combined_score_cache,
    )


async def _find_relevant_path(
    graph_storage,
    collection: Collection,
    relationship_query_embedding: list[float],
    source_id: str,
    target_id: str,
    rel_score_cache: dict[str, float],
    rel_combined_score_cache: dict[str, float],
    *,
    max_depth: int = 3,
    beam_width: int = 4,
    query_tokens: set[str] | None = None,
    rel_types: list[str] | None = None,
    dimension_weight: float = 1.0,
    document_ids: list[uuid.UUID] | None = None,
) -> tuple[list[str], list[str]] | None:
    queue: deque[tuple[str, list[str], list[str], int]] = deque(
        [(source_id, [source_id], [], 0)]
    )
    if query_tokens is None:
        query_tokens = set()

    while queue:
        node_id, path_nodes, path_rels, depth = queue.popleft()
        if depth >= max_depth:
            continue

        edges = await graph_storage.get_node_edges(
            node_id,
            rel_types=rel_types,
            document_ids=[str(doc_id) for doc_id in document_ids]
            if document_ids
            else None,
        )
        candidates: list[tuple[float, str, str]] = []
        for src, tgt in edges:
            neighbor = tgt if src == node_id else src
            if neighbor in path_nodes:
                continue

            edge_props = await graph_storage.get_edge(
                src,
                tgt,
                rel_types=rel_types,
                document_ids=[str(doc_id) for doc_id in document_ids]
                if document_ids
                else None,
            )
            if not edge_props:
                edge_props = await graph_storage.get_edge(
                    tgt,
                    src,
                    rel_types=rel_types,
                    document_ids=[str(doc_id) for doc_id in document_ids]
                    if document_ids
                    else None,
                )
            if not (edge_props and edge_props.get("id")):
                continue

            rel_id = str(edge_props["id"])
            sim = await _score_relationship(
                collection,
                relationship_query_embedding,
                rel_id,
                rel_score_cache,
            )
            combined = (
                _combined_edge_score(sim, edge_props, query_tokens)
                * dimension_weight
            )
            previous = rel_combined_score_cache.get(rel_id)
            if previous is None or combined > previous:
                rel_combined_score_cache[rel_id] = combined
            candidates.append((combined, neighbor, rel_id))

        for _, neighbor, rel_id in sorted(candidates, reverse=True)[:beam_width]:
            next_nodes = path_nodes + [neighbor]
            next_rels = path_rels + [rel_id]
            if neighbor == target_id:
                return next_nodes, next_rels
            queue.append((neighbor, next_nodes, next_rels, depth + 1))

    return None


async def _relationship_seed_state(
    collection: Collection,
    relationship_query_embedding: list[float],
    *,
    debug_label: str | None = None,
    top_k: int = 10,
    max_endpoints: int = 20,
    max_pairs: int = 30,
    query_tokens: set[str] | None = None,
    rel_types: list[str] | None = None,
    dimension_weight: float = 1.0,
    document_ids: list[uuid.UUID] | None = None,
) -> GraphQueryState:
    rel_seeds = await _search_relationship_seeds(
        collection,
        relationship_query_embedding,
        top_k=top_k,
        document_ids=document_ids,
    )
    if query_tokens is None:
        query_tokens = set()

    traversed_rel_ids: list[str] = []
    rel_score_cache: dict[str, float] = {}
    rel_combined_score_cache: dict[str, float] = {}
    discovered_entity_ids: set[str] = set()
    entity_relevance: dict[str, float] = {}
    graph_storage = get_graph_storage(collection)
    entity_name_by_id: dict[str, str] = {}
    rel_debug_rows: list[dict[str, Any]] = []
    raw_rel_seed_rows: list[dict[str, Any]] = []

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
        for rel_id_str, sim in rel_seeds[:top_k]:
            rel = await session.get(GraphRelationship, uuid.UUID(rel_id_str))
            if not rel:
                continue
            raw_rel_seed_rows.append(
                {
                    "rel_id": rel_id_str,
                    "sim": round(sim, 6),
                    "source_id": str(rel.source_entity_id),
                    "source_name": entity_name_by_id.get(str(rel.source_entity_id)),
                    "target_id": str(rel.target_entity_id),
                    "target_name": entity_name_by_id.get(str(rel.target_entity_id)),
                    "stored_rel_type": str(rel.rel_type or ""),
                }
            )
            edge_props = await graph_storage.get_edge(
                str(rel.source_entity_id),
                str(rel.target_entity_id),
                rel_types=rel_types,
                document_ids=[str(doc_id) for doc_id in document_ids]
                if document_ids
                else None,
            )
            if edge_props is None:
                edge_props = await graph_storage.get_edge(
                    str(rel.target_entity_id),
                    str(rel.source_entity_id),
                    rel_types=rel_types,
                    document_ids=[str(doc_id) for doc_id in document_ids]
                    if document_ids
                    else None,
                )
            if not edge_props:
                continue
            combined = (
                _combined_edge_score(sim, edge_props, query_tokens)
                * dimension_weight
            )
            rel_debug_rows.append(
                {
                    "rel_id": rel_id_str,
                    "sim": round(sim, 6),
                    "combined": round(combined, 6),
                    "source_id": str(rel.source_entity_id),
                    "source_name": entity_name_by_id.get(str(rel.source_entity_id)),
                    "target_id": str(rel.target_entity_id),
                    "target_name": entity_name_by_id.get(str(rel.target_entity_id)),
                    "rel_type": str(edge_props.get("rel_type") or rel.rel_type or ""),
                    "weight": edge_props.get("weight"),
                    "keywords": edge_props.get("keywords") or [],
                }
            )
            traversed_rel_ids.append(rel_id_str)
            rel_score_cache[rel_id_str] = sim
            rel_combined_score_cache[rel_id_str] = combined
            for eid in (str(rel.source_entity_id), str(rel.target_entity_id)):
                discovered_entity_ids.add(eid)
                if eid not in entity_relevance or combined > entity_relevance[eid]:
                    entity_relevance[eid] = combined

    endpoint_ids = sorted(
        discovered_entity_ids,
        key=lambda eid: entity_relevance.get(eid, 0.0),
        reverse=True,
    )[:max_endpoints]

    pair_count = 0
    for source_id, target_id in combinations(endpoint_ids, 2):
        if pair_count >= max_pairs:
            break
        pair_count += 1
        path = await _find_relevant_path(
            graph_storage,
            collection,
            relationship_query_embedding,
            source_id,
            target_id,
            rel_score_cache,
            rel_combined_score_cache,
            query_tokens=query_tokens,
            rel_types=rel_types,
            dimension_weight=dimension_weight,
            document_ids=document_ids,
        )
        if not path:
            continue
        path_nodes, path_rels = path
        discovered_entity_ids.update(path_nodes)
        for rel_id in path_rels:
            if rel_id not in traversed_rel_ids:
                traversed_rel_ids.append(rel_id)
        path_score = sum(
            rel_combined_score_cache.get(rel_id, 0.0) for rel_id in path_rels
        )
        if path_rels:
            path_score /= len(path_rels)
        for eid in path_nodes:
            if eid not in entity_relevance or path_score > entity_relevance[eid]:
                entity_relevance[eid] = path_score

    logger.info(
        "graph_rag relationship_seed_state label=%r collection=%s rel_types=%s raw_rel_seeds=%s discovered=%d traversed=%d top_entities=%s top_rels=%s rel_debug=%s",
        debug_label,
        collection.name,
        rel_types,
        raw_rel_seed_rows[:10],
        len(discovered_entity_ids),
        len(traversed_rel_ids),
        sorted(
            (
                (
                    eid,
                    entity_name_by_id.get(eid, eid),
                    round(entity_relevance.get(eid, 0.0), 6),
                )
                for eid in discovered_entity_ids
            ),
            key=lambda item: item[2],
            reverse=True,
        )[:10],
        [
            (rel_id, round(rel_combined_score_cache.get(rel_id, 0.0), 6))
            for rel_id in traversed_rel_ids[:10]
        ],
        rel_debug_rows[:10],
    )
    return GraphQueryState(
        discovered_entity_ids=discovered_entity_ids,
        entity_relevance=entity_relevance,
        traversed_rel_ids=traversed_rel_ids,
        rel_score_cache=rel_score_cache,
        rel_combined_score_cache=rel_combined_score_cache,
    )


async def _relationship_first_state(
    question: str,
    collection: Collection,
    entity_query_embedding: list[float],
    relationship_query_embedding: list[float],
    *,
    rel_types: list[str] | None = None,
    dimension_weight: float = 1.0,
    document_ids: list[uuid.UUID] | None = None,
) -> GraphQueryState:
    query_tokens = _query_token_set(question)
    state = await _relationship_seed_state(
        collection,
        relationship_query_embedding,
        debug_label=f"relationship_first:{question}",
        top_k=10,
        max_endpoints=30,
        max_pairs=40,
        query_tokens=query_tokens,
        rel_types=rel_types,
        dimension_weight=dimension_weight,
        document_ids=document_ids,
    )
    return await _filter_relationship_state_by_entity_score(
        collection,
        state,
        entity_query_embedding,
        min_entity_score=_REL_ENDPOINT_ENTITY_SCORE_MIN,
        document_ids=document_ids,
    )


async def _filter_relationship_state_by_entity_score(
    collection: Collection,
    state: GraphQueryState,
    entity_query_embedding: list[float],
    *,
    min_entity_score: float,
    top_k: int = 50,
    document_ids: list[uuid.UUID] | None = None,
) -> GraphQueryState:
    if not state.discovered_entity_ids:
        return state

    candidates = await _top_entity_candidates(
        collection,
        entity_query_embedding,
        top_k=top_k,
        document_ids=document_ids,
    )
    entity_score_by_name = {
        name.strip().lower(): score for name, _, score in candidates
    }

    async with AsyncSessionLocal() as session:
        entity_rows = await session.execute(
            select(GraphEntity.id, GraphEntity.canonical_name).where(
                GraphEntity.collection_id == collection.id,
                GraphEntity.id.in_(
                    [uuid.UUID(entity_id) for entity_id in state.discovered_entity_ids]
                ),
            )
        )
        entity_name_by_id = {
            str(entity_id): canonical_name
            for entity_id, canonical_name in entity_rows.all()
        }

        kept_entity_ids = {
            entity_id
            for entity_id, entity_name in entity_name_by_id.items()
            if entity_score_by_name.get(entity_name.strip().lower(), 0.0)
            >= min_entity_score
        }
        if not kept_entity_ids:
            logger.info(
                "graph_rag relationship_state_filter collection=%s threshold=%.3f kept=0 discovered=%d",
                collection.name,
                min_entity_score,
                len(state.discovered_entity_ids),
            )
            return GraphQueryState(
                discovered_entity_ids=set(),
                entity_relevance={},
                traversed_rel_ids=[],
                rel_score_cache={},
                rel_combined_score_cache={},
            )

        filtered_rel_ids: list[str] = []
        filtered_rel_score_cache: dict[str, float] = {}
        filtered_rel_combined_score_cache: dict[str, float] = {}

        traversed_rel_uuids = [
            uuid.UUID(rel_id) for rel_id in state.traversed_rel_ids if rel_id
        ]
        if traversed_rel_uuids:
            rel_rows = await session.execute(
                select(
                    GraphRelationship.id,
                    GraphRelationship.source_entity_id,
                    GraphRelationship.target_entity_id,
                ).where(GraphRelationship.id.in_(traversed_rel_uuids))
            )
            for rel_id, source_entity_id, target_entity_id in rel_rows.all():
                rel_id_str = str(rel_id)
                if (
                    str(source_entity_id) in kept_entity_ids
                    and str(target_entity_id) in kept_entity_ids
                ):
                    if document_ids:
                        rel_desc_result = await session.execute(
                            select(RelationshipDescription.document_id).where(
                                RelationshipDescription.relationship_id == rel_id,
                                RelationshipDescription.document_id.in_(document_ids),
                            )
                        )
                        if rel_desc_result.scalar_one_or_none() is None:
                            continue
                    filtered_rel_ids.append(rel_id_str)
                    if rel_id_str in state.rel_score_cache:
                        filtered_rel_score_cache[rel_id_str] = state.rel_score_cache[
                            rel_id_str
                        ]
                    if rel_id_str in state.rel_combined_score_cache:
                        filtered_rel_combined_score_cache[rel_id_str] = (
                            state.rel_combined_score_cache[rel_id_str]
                        )

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
            if document_ids:
                rel_desc_result = await session.execute(
                    select(RelationshipDescription.document_id).where(
                        RelationshipDescription.relationship_id == rel.id,
                        RelationshipDescription.document_id.in_(document_ids),
                    )
                )
                if rel_desc_result.scalar_one_or_none() is None:
                    continue
            rel_id_str = str(rel.id)
            if rel_id_str in filtered_rel_ids:
                continue
            filtered_rel_ids.append(rel_id_str)
            filtered_rel_score_cache[rel_id_str] = state.rel_score_cache.get(
                rel_id_str, 0.0
            )
            filtered_rel_combined_score_cache[rel_id_str] = (
                state.rel_combined_score_cache.get(rel_id_str, 0.0)
            )

    filtered_entity_relevance = {
        entity_id: max(
            state.entity_relevance.get(entity_id, 0.0),
            entity_score_by_name.get(
                entity_name_by_id.get(entity_id, "").strip().lower(),
                0.0,
            ),
        )
        for entity_id in kept_entity_ids
    }
    filtered_rel_ids.sort(
        key=lambda rel_id: filtered_rel_combined_score_cache.get(rel_id, 0.0),
        reverse=True,
    )
    logger.info(
        "graph_rag relationship_state_filter collection=%s threshold=%.3f kept=%d filtered_rels=%d kept_entities=%s",
        collection.name,
        min_entity_score,
        len(kept_entity_ids),
        len(filtered_rel_ids),
        sorted(
            (
                (
                    entity_id,
                    entity_name_by_id.get(entity_id, entity_id),
                    round(filtered_entity_relevance.get(entity_id, 0.0), 6),
                )
                for entity_id in kept_entity_ids
            ),
            key=lambda item: item[2],
            reverse=True,
        )[:10],
    )
    return GraphQueryState(
        discovered_entity_ids=kept_entity_ids,
        entity_relevance=filtered_entity_relevance,
        traversed_rel_ids=filtered_rel_ids,
        rel_score_cache=filtered_rel_score_cache,
        rel_combined_score_cache=filtered_rel_combined_score_cache,
    )


def _merge_states(*states: GraphQueryState) -> GraphQueryState:
    discovered_entity_ids: set[str] = set()
    entity_relevance: dict[str, float] = {}
    traversed_rel_ids: list[str] = []
    rel_score_cache: dict[str, float] = {}
    rel_combined_score_cache: dict[str, float] = {}

    for state in states:
        discovered_entity_ids.update(state.discovered_entity_ids)
        for eid, score in state.entity_relevance.items():
            if eid not in entity_relevance or score > entity_relevance[eid]:
                entity_relevance[eid] = score
        for rel_id in state.traversed_rel_ids:
            if rel_id not in traversed_rel_ids:
                traversed_rel_ids.append(rel_id)
        rel_score_cache.update(state.rel_score_cache)
        rel_combined_score_cache.update(state.rel_combined_score_cache)

    traversed_rel_ids.sort(
        key=lambda rel_id: rel_combined_score_cache.get(rel_id, 0.0),
        reverse=True,
    )

    return GraphQueryState(
        discovered_entity_ids=discovered_entity_ids,
        entity_relevance=entity_relevance,
        traversed_rel_ids=traversed_rel_ids,
        rel_score_cache=rel_score_cache,
        rel_combined_score_cache=rel_combined_score_cache,
    )


def _fallback_mix_interpretation(
    candidates: list[tuple[str, str, float]],
) -> MixInterpretation:
    names = [name for name, _, _ in candidates[:8]]
    if not names:
        return MixInterpretation(selected_entities=[], retrieval_subqueries=[])
    subqueries: list[str] = []
    if "Rajas" in names and "The Mind" in names:
        subqueries.append(
            "Why does sustained focused activity of Rajas in The Mind "
            "still lead to exhaustion?"
        )
    if "Prana" in names or "Ojas" in names:
        subqueries.append(
            "How do Prana and Ojas explain mental drain, reduced steadiness, "
            "and loss of endurance after overwork?"
        )
    if "Samkalpa" in names:
        subqueries.append(
            "What is the relationship between Samkalpa, repeated mental effort, "
            "and exhaustion from solving one problem after another?"
        )
    if "Pratyahara" in names:
        subqueries.append(
            "Does lack of Pratyahara or stopping allow The Mind to keep spending "
            "energy without recovery?"
        )
    if not subqueries:
        subqueries = [
            (
                "How do "
                + ", ".join(names[:4])
                + " explain continuous mental effort and exhaustion?"
            ),
            (
                "How do "
                + ", ".join(names[4:8])
                + " relate to steadiness, recovery, and endurance?"
            ),
        ]
    return MixInterpretation(
        selected_entities=names,
        retrieval_subqueries=[query for query in subqueries if query.strip()][:4],
    )


async def _interpret_mix_queries(
    question: str,
    candidates: list[tuple[str, str, float]],
    llm_provider: LLMProvider,
) -> MixInterpretation:
    logger.info(
        "graph_rag mix_interpretation_start question=%r candidates=%s",
        question,
        [(name, round(score, 6)) for name, _, score in candidates[:12]],
    )
    if isinstance(llm_provider, LocalEchoLLMProvider):
        return _fallback_mix_interpretation(candidates)

    entity_lines = []
    for idx, (name, description, score) in enumerate(candidates[:20], start=1):
        snippet = description.replace("\n", " ").strip()
        entity_lines.append(f"{idx}. {name} (score={score:.3f}): {snippet[:120]}")

    schema = {
        "type": "object",
        "properties": {
            "selected_entities": {
                "type": "array",
                "items": {"type": "string"},
            },
            "retrieval_subqueries": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["selected_entities", "retrieval_subqueries"],
    }

    prompt = (
        "You are preparing retrieval queries for a knowledge graph.\n"
        "Use only the entity names from the candidate list.\n"
        "Select up to 8 relevant entities for the user's question.\n"
        "Then produce 2 to 4 longer retrieval subqueries.\n"
        "Each subquery should be one sentence, around 12 to 30 words, and target "
        "a separate explanatory dimension of the question.\n"
        "Use selected entity names heavily, but keep the subqueries semantically "
        "specific instead of turning them into short keyword bags.\n"
        "Favor dimensions like mechanism, energetic depletion, counterbalance, "
        "and the user's stated contrast or objection.\n"
        "Preserve the user's distinctions and negations.\n\n"
        f"User question:\n{question}\n\n"
        "Candidate entities:\n"
        f"{chr(10).join(entity_lines)}"
    )

    try:
        result = await llm_provider.structured_extract(prompt=prompt, schema=schema)
    except Exception:
        return _fallback_mix_interpretation(candidates)

    selected_entities = [
        str(value).strip()
        for value in result.get("selected_entities", [])
        if str(value).strip()
    ]
    retrieval_subqueries = [
        str(value).strip()
        for value in result.get("retrieval_subqueries", [])
        if str(value).strip()
    ]

    if not retrieval_subqueries:
        return _fallback_mix_interpretation(candidates)

    logger.info(
        "graph_rag mix_interpretation_done selected_entities=%s subqueries=%s",
        selected_entities[:8],
        retrieval_subqueries[:4],
    )
    return MixInterpretation(
        selected_entities=selected_entities[:8],
        retrieval_subqueries=retrieval_subqueries[:4],
    )


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


async def _log_exact_name_hits(collection: Collection, question: str) -> None:
    entity_text = _diagnostic_entity_text(question)
    async with AsyncSessionLocal() as session:
        entity_rows = (
            await session.execute(
                select(
                    GraphEntity.id,
                    GraphEntity.canonical_name,
                    GraphEntity.primary_type,
                ).where(
                    GraphEntity.collection_id == collection.id,
                    GraphEntity.canonical_name.ilike(entity_text),
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
                    EntityAlias.collection_id == collection.id,
                    EntityAlias.alias_name.ilike(entity_text),
                )
            )
        ).all()
    logger.info(
        "graph_rag exact_name_hits collection=%s question=%r entity_text=%r entities=%s aliases=%s",
        collection.name,
        question,
        entity_text,
        [
            (str(entity_id), canonical_name, primary_type)
            for entity_id, canonical_name, primary_type in entity_rows
        ],
        [
            (str(alias_id), str(entity_id), alias_name)
            for alias_id, entity_id, alias_name in alias_rows
        ],
    )


async def _mix_state(
    question: str,
    collection: Collection,
    namespace_id: uuid.UUID,
    llm_profile_id: uuid.UUID | None,
    embedding_provider: EmbeddingProvider,
    entity_query_embedding: list[float],
    relationship_query_embedding: list[float],
    *,
    rel_types: list[str] | None = None,
    dimension_weight: float = 1.0,
    document_ids: list[uuid.UUID] | None = None,
) -> GraphQueryState:
    query_tokens = _query_token_set(question)
    await _log_exact_name_hits(collection, question)
    candidates = await _top_entity_candidates(
        collection,
        entity_query_embedding,
        top_k=50,
        document_ids=document_ids,
    )
    top_entity_score = candidates[0][2] if candidates else 0.0
    logger.info(
        "graph_rag mix_state_start collection=%s question=%r top_entity_score=%.6f threshold=%.6f rel_types=%s",
        collection.name,
            question,
            top_entity_score,
            _MIX_REWRITE_MIN_SCORE,
            rel_types,
        )

    rel_base_state = await _relationship_seed_state(
        collection,
        relationship_query_embedding,
        debug_label=f"mix_base:{question}",
        top_k=10,
        max_endpoints=30,
        max_pairs=40,
        query_tokens=query_tokens,
        rel_types=rel_types,
        dimension_weight=dimension_weight,
        document_ids=document_ids,
    )
    rel_base_state = await _filter_relationship_state_by_entity_score(
        collection,
        rel_base_state,
        entity_query_embedding,
        min_entity_score=_REL_ENDPOINT_ENTITY_SCORE_MIN,
        document_ids=document_ids,
    )
    if top_entity_score < _MIX_REWRITE_MIN_SCORE:
        logger.info(
            "graph_rag mix_state_fallback collection=%s reason=top_entity_score_below_threshold",
            collection.name,
        )
        return rel_base_state

    llm_provider = await _resolve_llm_provider(
        namespace_id=namespace_id,
        llm_profile_id=llm_profile_id,
    )
    interpretation = await _interpret_mix_queries(question, candidates, llm_provider)

    # Batch-embed all subquery relationship queries in one API call.
    subqueries = interpretation.retrieval_subqueries
    if subqueries:
        subquery_embeddings = await _embed_relationship_queries_batch(
            embedding_provider,
            subqueries,
            [None] * len(subqueries),
        )
    else:
        subquery_embeddings = []

    async def _build_subquery_state(
        subquery: str, embedding: list[float]
    ) -> GraphQueryState:
        subquery_tokens = query_tokens | _query_token_set(subquery)
        subquery_state = await _relationship_seed_state(
            collection,
            embedding,
            debug_label=f"mix_subquery:{subquery}",
            top_k=10,
            max_endpoints=20,
            max_pairs=30,
            query_tokens=subquery_tokens,
            rel_types=rel_types,
            dimension_weight=dimension_weight,
            document_ids=document_ids,
        )
        subquery_entity_embedding = await _embed_entity_query(
            embedding_provider,
            subquery,
        )
        return await _filter_relationship_state_by_entity_score(
            collection,
            subquery_state,
            subquery_entity_embedding,
            min_entity_score=_REL_ENDPOINT_ENTITY_SCORE_MIN,
            document_ids=document_ids,
        )

    concurrency = settings.graph_rag_query_embedding_concurrency
    sem = asyncio.Semaphore(concurrency)

    async def _gated(sq: str, emb: list[float]):
        async with sem:
            return await _build_subquery_state(sq, emb)

    states = await asyncio.gather(
        *(_gated(sq, emb) for sq, emb in zip(subqueries, subquery_embeddings))
    )
    logger.info(
        "graph_rag mix_state_subqueries collection=%s selected_entities=%s subqueries=%s discovered_counts=%s traversed_counts=%s discovered_entities=%s traversed_rel_ids=%s",
        collection.name,
        interpretation.selected_entities,
        subqueries,
        [len(state.discovered_entity_ids) for state in states],
        [len(state.traversed_rel_ids) for state in states],
        [sorted(state.discovered_entity_ids)[:20] for state in states],
        [state.traversed_rel_ids[:20] for state in states],
    )

    if not states:
        return rel_base_state

    return _merge_states(rel_base_state, *states)


def _normalize_scalar_map(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_value = max(scores.values()) or 0.0
    if max_value <= 0.0:
        return {key: 0.0 for key in scores}
    return {key: float(value) / float(max_value) for key, value in scores.items()}


async def _derive_route_profile(
    collection: Collection,
    state: GraphQueryState,
) -> DerivedRouteProfile:
    _ = collection
    route_scores = {
        "hub": 0.0,
        "authority": 0.0,
        "bridge": 0.0,
        "central": 1.0,
        "importance": 0.8,
    }
    rel_type_scores: dict[str, float] = defaultdict(float)
    if not state.traversed_rel_ids:
        normalized_route_scores = _normalize_scalar_map(route_scores)
        return DerivedRouteProfile(
            primary_route=max(
                normalized_route_scores.items(),
                key=lambda item: item[1],
            )[0],
            route_scores=normalized_route_scores,
            rel_type_scores={},
        )

    async with AsyncSessionLocal() as session:
        rel_rows = (
            await session.execute(
                select(
                    GraphRelationship.id,
                    GraphRelationshipType.canonical_type,
                )
                .select_from(GraphRelationship)
                .join(
                    GraphRelationshipType,
                    GraphRelationshipType.id == GraphRelationship.relationship_type_id,
                )
                .where(
                    GraphRelationship.id.in_(
                        [
                            uuid.UUID(rel_id)
                            for rel_id in state.traversed_rel_ids
                            if rel_id
                        ]
                    )
                )
            )
        ).all()

    for rel_id, rel_type in rel_rows:
        rel_id_str = str(rel_id)
        rel_type_norm = normalize_dim(rel_type)
        score = float(state.rel_combined_score_cache.get(rel_id_str, 0.0) or 0.0)
        rel_type_scores[rel_type_norm] += score
        if rel_type_norm in {"CALLS", "USES", "EMITS", "SENDS", "SPAWNS"}:
            route_scores["hub"] += score
        if rel_type_norm in {
            "DEFINES",
            "EXTENDS",
            "IMPLEMENTS",
            "CONTAINS",
            "AUTHORED_BY",
        }:
            route_scores["authority"] += score
        if rel_type_norm in {
            "DEPENDS_ON",
            "CONNECTS_TO",
            "GUARDS",
            "AUTHORIZES",
            "AUTHENTICATES",
        }:
            route_scores["bridge"] += score
        if rel_type_norm in {"IMPORTS", "REFERENCES", "RELATES_TO", "ABOUT"}:
            route_scores["central"] += score
        if rel_type_norm in {"DOCUMENTS", "DESCRIBES", "SUMMARIZES", "PROVES"}:
            route_scores["importance"] += score / 2.0

    normalized_route_scores = _normalize_scalar_map(route_scores)
    primary_route = max(
        normalized_route_scores.items(),
        key=lambda item: item[1],
    )[0]
    top_rel_types = sorted(
        rel_type_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    logger.info(
        "graph_rag route profile collection=%s primary_route=%s route_scores=%s rel_types=%s matched_entities=%d traversed_rels=%d",
        collection.name,
        primary_route,
        normalized_route_scores,
        top_rel_types,
        len(state.discovered_entity_ids),
        len(state.traversed_rel_ids),
    )
    return DerivedRouteProfile(
        primary_route=primary_route,
        route_scores=normalized_route_scores,
        rel_type_scores=_normalize_scalar_map(
            {key: float(value) for key, value in rel_type_scores.items()}
        ),
    )


async def _build_context(
    state: GraphQueryState,
    collection: Collection,
    *,
    derived_context: str = "",
    document_ids: list[uuid.UUID] | None = None,
) -> tuple[str, list[str], list[str], str]:
    max_entities = 10
    max_entity_descs = 4
    max_rel_descs = 4

    async with AsyncSessionLocal() as session:
        ranked_entity_ids = sorted(
            state.discovered_entity_ids,
            key=lambda eid: state.entity_relevance.get(eid, 0.0),
            reverse=True,
        )

        entity_context_parts: list[str] = []
        entities_used: list[str] = []
        for eid_str in ranked_entity_ids[:max_entities]:
            try:
                eid = uuid.UUID(eid_str)
            except ValueError:
                continue
            entity = await session.get(GraphEntity, eid)
            if not entity:
                continue
            entity_conditions = [EntityDescription.entity_id == eid]
            if document_ids:
                entity_conditions.append(EntityDescription.document_id.in_(document_ids))
            descs_result = await session.execute(
                select(EntityDescription)
                .where(*entity_conditions)
                .order_by(EntityDescription.weight.desc())
                .limit(max_entity_descs)
            )
            descs = descs_result.scalars().all()
            if descs:
                desc_texts = " | ".join(
                    description.description for description in descs
                )
                entity_context_parts.append(
                    f"{entity.canonical_name} ({entity.primary_type or 'unknown'}): "
                    f"{desc_texts}"
                )
                entities_used.append(entity.canonical_name)

        rel_context_parts_by_type: dict[str, list[tuple[float, str]]] = {}
        relationships_used: list[str] = []
        for rel_id_str in state.traversed_rel_ids[:50]:
            try:
                rel_uuid = uuid.UUID(rel_id_str)
            except ValueError:
                continue
            rel = await session.get(GraphRelationship, rel_uuid)
            if not rel:
                continue
            src_entity = await session.get(GraphEntity, rel.source_entity_id)
            tgt_entity = await session.get(GraphEntity, rel.target_entity_id)
            src_name = src_entity.canonical_name if src_entity else "?"
            tgt_name = tgt_entity.canonical_name if tgt_entity else "?"
            rel_conditions = [RelationshipDescription.relationship_id == rel_uuid]
            if document_ids:
                rel_conditions.append(
                    RelationshipDescription.document_id.in_(document_ids)
                )
            descs_result = await session.execute(
                select(RelationshipDescription)
                .where(*rel_conditions)
                .order_by(RelationshipDescription.weight.desc())
                .limit(max_rel_descs)
            )
            descs = descs_result.scalars().all()
            sim = state.rel_combined_score_cache.get(rel_id_str, 0.0)
            rel_type = rel.rel_type or "RELATES_TO"
            for description in descs:
                rel_text = (
                    f"{src_name} -[{rel_type}]-> {tgt_name}: {description.description}"
                )
                rel_context_parts_by_type.setdefault(rel_type, []).append(
                    (sim, rel_text)
                )
                relationships_used.append(f"{src_name} -[{rel_type}]-> {tgt_name}")

    entity_context = "\n".join(entity_context_parts)
    rel_sections: list[tuple[float, str]] = []
    for rel_type, items in rel_context_parts_by_type.items():
        items.sort(key=lambda item: item[0], reverse=True)
        section_text = "\n".join(text for _, text in items)
        section_score = items[0][0] if items else 0.0
        rel_sections.append((section_score, f"{rel_type}:\n{section_text}"))
    rel_sections.sort(key=lambda item: item[0], reverse=True)
    rel_context = "\n\n".join(text for _, text in rel_sections)
    context = "Context:\n"
    if derived_context:
        context += f"Derived Understanding:\n{derived_context}\n"
    context += (
        "Entities:\n"
        f"{entity_context or '(none)'}\n"
        "Relationships By Type:\n"
        f"{rel_context or '(none)'}"
    )
    return context, entities_used, list(dict.fromkeys(relationships_used)), rel_context


async def _answer_from_context(
    question: str,
    namespace_id: uuid.UUID,
    llm_profile_id: uuid.UUID | None,
    context: str,
    fallback_text: str,
) -> str:
    llm_provider = await _resolve_llm_provider(
        namespace_id=namespace_id,
        llm_profile_id=llm_profile_id,
    )
    if isinstance(llm_provider, LocalEchoLLMProvider):
        return fallback_text or "No relevant context found."
    return await llm_provider.chat([
        {
            "role": "system",
            "content": (
                "Use the context below to answer the question. "
                "Draw on the entities and relationships to reason through "
                "your answer - "
                "explain, connect, and illuminate rather than just report. "
                "Write in natural prose. If the context is insufficient "
                "for part of the "
                "question, acknowledge it briefly without making it the focus."
                "\n\nTreat the context as a graph-backed record of stored entities, descriptions, aliases, "
                "and relationships. Use that evidence to ground your answer."
                "\n\nIf an Internal Higher-Level Context section is present, it contains progressively broader, "
                "more synthesized concepts built from lower levels. Use it only for internal navigation: it may "
                "help you identify which lower-level or base-level entities matter. Do not quote, name, "
                "or center the answer on higher-level meta concepts when you can answer using Primary Evidence."
                "\n\nIn the final answer, prefer entities and descriptions from Primary Evidence over entities that "
                "appear only in higher-level meta layers. If a higher-level layer helps you find the right answer, "
                "translate back down before answering."
                "\n\nDo not explain your reasoning using graph terminology. Avoid phrases like graph, node, edge, "
                "meta graph, base graph, EVIDENCED_BY, CONNECTS_TO, chain, layer, or abstraction ladder in the final answer "
                "unless the user explicitly asks about the graph representation itself. Instead, restate the underlying "
                "mechanism, workflow, component behavior, or implementation concern in normal prose."
                "\n\nThe Relationships By Type section groups edges by"
                " semantic role. Each edge is listed in the form"
                " 'SRC -[REL_TYPE]-> TGT: description'. REL_TYPE is the"
                " semantic role of the edge (e.g. EXPLAINS, CAUSES, IS_AN_EXAMPLE_OF);"
                " the same SRC and TGT may appear with several different"
                " REL_TYPEs and each one carries a separate meaning."
            ),
        },
        {
            "role": "user",
            "content": f"{context}\n\nQuestion: {question}",
        },
    ])


async def _load_meta_collections(collection: Collection) -> list[Collection]:
    root_name = base_collection_name(collection.name)
    current_level = meta_collection_level(collection.name)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Collection).where(Collection.namespace_id == collection.namespace_id)
        )
        collections = list(result.scalars().all())
    descendants: dict[int, Collection] = {}
    for candidate in collections:
        parsed = parse_meta_collection_name(candidate.name)
        if parsed is None:
            continue
        candidate_root, candidate_level = parsed
        if candidate_root != root_name or candidate_level <= current_level:
            continue
        existing = descendants.get(candidate_level)
        if existing is None or candidate.name == meta_collection_name(
            root_name, candidate_level
        ):
            descendants[candidate_level] = candidate
    return [descendants[level] for level in sorted(descendants)]


def _strip_context_label(context: str) -> str:
    prefix = "Context:\n"
    if context.startswith(prefix):
        return context[len(prefix):]
    return context


async def _build_graph_query_artifacts(
    question: str,
    collection: Collection,
    namespace_id: uuid.UUID,
    mode: str | None = None,
    llm_profile_id: uuid.UUID | None = None,
    document_ids: list[uuid.UUID] | None = None,
) -> GraphQueryArtifacts:
    embedding_provider = await _resolve_embedding_provider(collection)
    entity_query_embedding = await _embed_entity_query(embedding_provider, question)

    requested_mode = (mode or "mix").lower()
    effective_mode = _MODE_ALIASES.get(requested_mode, "mix")
    dimensions: list[str] = []
    logger.info(
        "graph_rag dimension_gating_disabled collection=%s mode=%s threshold=%.3f",
        collection.name,
        effective_mode,
        _REL_ENDPOINT_ENTITY_SCORE_MIN,
    )

    async def _build_state_for(rel_type: str | None) -> GraphQueryState:
        relationship_query_embedding = await _embed_relationship_query(
            embedding_provider,
            question,
            rel_type=None,
        )
        kwargs = {
            "rel_types": None,
            "dimension_weight": 1.0,
        }
        if effective_mode == "relationship-first":
            return await _relationship_first_state(
                question,
                collection,
                entity_query_embedding,
                relationship_query_embedding,
                **kwargs,
                document_ids=document_ids,
            )
        if effective_mode == "mix":
            return await _mix_state(
                question,
                collection,
                namespace_id,
                llm_profile_id,
                embedding_provider,
                entity_query_embedding,
                relationship_query_embedding,
                **kwargs,
                document_ids=document_ids,
            )
        if effective_mode == "hybrid":
            entity_state = await _entity_first_state(
                question,
                collection,
                entity_query_embedding,
                relationship_query_embedding,
                **kwargs,
                document_ids=document_ids,
            )
            relationship_state = await _relationship_first_state(
                question,
                collection,
                entity_query_embedding,
                relationship_query_embedding,
                **kwargs,
                document_ids=document_ids,
            )
            return _merge_states(entity_state, relationship_state)
        return await _entity_first_state(
            question,
            collection,
            entity_query_embedding,
            relationship_query_embedding,
            **kwargs,
            document_ids=document_ids,
        )

    state = await _fan_out_per_dimension(_build_state_for, dimensions)
    if state is None:
        state = GraphQueryState(
            discovered_entity_ids=set(),
            entity_relevance={},
            traversed_rel_ids=[],
            rel_score_cache={},
            rel_combined_score_cache={},
        )
    if effective_mode not in ("relationship-first", "mix", "hybrid"):
        effective_mode = "entity-first"

    route_profile = await _derive_route_profile(collection, state)
    context, entities_used, relationships_used, rel_context = await _build_context(
        state,
        collection,
        document_ids=document_ids,
    )
    logger.info(
        "graph_rag artifacts collection=%s mode=%s route=%s entities_used=%s relationships_used=%s",
        collection.name,
        effective_mode,
        route_profile.primary_route,
        entities_used[:20],
        relationships_used[:20],
    )
    logger.info(
        "graph_rag context collection=%s route=%s entities=%d relationships=%d",
        collection.name,
        route_profile.primary_route,
        len(entities_used),
        len(relationships_used),
    )
    return GraphQueryArtifacts(
        context=context,
        entities_used=entities_used,
        relationships_used=relationships_used,
        rel_context=rel_context,
        route_profile=route_profile,
    )


async def graph_rag_query(
    question: str,
    collection: Collection,
    namespace_id: uuid.UUID,
    mode: str | None = None,
    llm_profile_id: uuid.UUID | None = None,
) -> QueryResult:
    routing = await _resolve_document_routing(
        question,
        collection,
        namespace_id,
        llm_profile_id,
    )
    document_ids = routing.document_ids if not routing.use_all_documents else None
    logger.info(
        "graph_rag document_routing collection=%s route=%s document_ids=%s",
        collection.name,
        "all" if routing.use_all_documents else "documents",
        [str(document_id) for document_id in document_ids] if document_ids else [],
    )
    base = await _build_graph_query_artifacts(
        question,
        collection,
        namespace_id,
        mode,
        llm_profile_id,
        document_ids=document_ids,
    )
    meta_collections = await _load_meta_collections(collection)
    meta_artifacts: list[tuple[Collection, GraphQueryArtifacts]] = []
    for meta_collection in meta_collections:
        meta_artifacts.append(
            (
                meta_collection,
                await _build_graph_query_artifacts(
                    question,
                    meta_collection,
                    namespace_id,
                    mode,
                    llm_profile_id,
                    document_ids=None,
                ),
            )
        )

    if meta_artifacts:
        meta_sections = "\n\n".join(
            (
                f"Level {meta_collection_level(meta_collection.name)} "
                f"({meta_collection.name}):\n"
                f"{_strip_context_label(artifacts.context)}"
            )
            for meta_collection, artifacts in meta_artifacts
        )
        context = (
            "Internal Higher-Level Context:\n"
            f"{meta_sections}\n\n"
            "Primary Evidence:\n"
            f"{_strip_context_label(base.context)}"
        )
    else:
        context = base.context

    meta_entity_count = sum(
        len(artifacts.entities_used) for _, artifacts in meta_artifacts
    )
    meta_relationship_count = sum(
        len(artifacts.relationships_used) for _, artifacts in meta_artifacts
    )
    logger.info(
        "graph_rag final context collection=%s meta_collections=%s route=%s entities=%d relationships=%d",
        collection.name,
        [meta_collection.name for meta_collection, _ in meta_artifacts],
        base.route_profile.primary_route,
        len(base.entities_used) + meta_entity_count,
        len(base.relationships_used) + meta_relationship_count,
    )
    entity_fallback = "\n".join(base.entities_used)
    meta_fallback = "\n\n".join(
        artifacts.context for _, artifacts in meta_artifacts if artifacts.context
    )
    fallback_text = meta_fallback or base.rel_context or entity_fallback
    all_meta_entities = [
        entity
        for _, artifacts in meta_artifacts
        for entity in artifacts.entities_used
    ]
    all_meta_relationships = [
        relationship
        for _, artifacts in meta_artifacts
        for relationship in artifacts.relationships_used
    ]
    response = await _answer_from_context(
        question,
        namespace_id,
        llm_profile_id,
        context,
        fallback_text,
    )
    return QueryResult(
        response=response,
        entities_used=list(
            dict.fromkeys(
                base.entities_used + all_meta_entities
            )
        ),
        relationships_used=list(
            dict.fromkeys(
                base.relationships_used + all_meta_relationships
            )
        ),
        mode=_MODE_ALIASES.get((mode or "mix").lower(), "mix"),
    )

"""Graph RAG query functions extracted from GraphService."""

from __future__ import annotations

import string
import uuid
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from sqlalchemy import or_, select

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
    RelationshipDescription,
)
from graph_core.models.profile import Profile
from graph_core.models.rel_types import (
    normalize_rel_type as normalize_dim,
)
from graph_core.models.rel_types import (
    relationship_embedding_text,
)
from graph_core.services.crypto import CredentialCrypto
from graph_core.services.graph.analytics import derived_graph_name
from graph_core.services.graph.query.vector import QueryResult
from graph_core.storage.graph_rag_vectors import GraphRAGVectorStore
from graph_core.storage.graph_storage import FalkorDBGraphStorage
from graph_core.storage.vector_store import VectorStore

_graph_rag_vectors = GraphRAGVectorStore()
_vector_store = VectorStore()
_crypto = CredentialCrypto()
_ENTITY_RETRIEVAL_INSTRUCTION = (
    "Retrieve ontology entities whose descriptions best explain the user's "
    "state, process, causal mechanism, or source of exhaustion."
)
_RELATIONSHIP_RETRIEVAL_INSTRUCTION = (
    "Retrieve relationship descriptions that best explain the user's question, "
    "especially causes, mechanisms, tensions, and energy depletion."
)
_DERIVED_RETRIEVAL_INSTRUCTION = (
    "Retrieve high-level derived graph summaries that capture subsystems, "
    "bridge nodes, connector paths, and synthesized understanding relevant "
    "to the user's question."
)
_MIX_REWRITE_MIN_SCORE = 0.55
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


def _active_dimensions() -> list[str]:
    """Active graph dimensions, in priority order.

    Falls back to all known rel_types when the operator has not pinned
    a subset, so collections ingested before this setting was added keep
    working unchanged.
    """
    configured = list(settings.graph_rag_active_dimensions or [])
    if configured:
        return [normalize_dim(d) for d in configured]
    from graph_core.models.rel_types import (
        DEFAULT_REL_TYPE,
        DOMAIN_VOCAB,
    )
    types: list[str] = []
    for vocab in DOMAIN_VOCAB.values():
        for t in vocab:
            if t not in types:
                types.append(t)
    if DEFAULT_REL_TYPE not in types:
        types.insert(0, DEFAULT_REL_TYPE)
    return types


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
    dims = dimensions if dimensions is not None else _active_dimensions()
    if not dims:
        return await build_state(None)
    states: list[GraphQueryState] = []
    for rel_type in dims:
        s = await build_state(rel_type)
        if s is not None:
            states.append(s)
    if not states:
        return None
    return _merge_states(*states)


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
class DerivedContext:
    context: str
    base_entity_ids: dict[str, float]
    derived_nodes_used: list[str]
    derived_edges_used: list[str]


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


async def _embed_derived_query(
    embedding_provider: EmbeddingProvider,
    query: str,
) -> list[float]:
    return await embedding_provider.embed_query(
        _format_retrieval_query(_DERIVED_RETRIEVAL_INSTRUCTION, query)
    )


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


def get_graph_storage(collection_id: uuid.UUID):
    from graph_core.storage.graph_storage import FalkorDBGraphStorage

    graph_name = f"collection_{str(collection_id).replace('-', '')}"
    return FalkorDBGraphStorage(graph_name)


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


async def _search_entity_seeds(
    question: str,
    collection: Collection,
    query_embedding: list[float],
) -> tuple[list[str], dict[str, float]]:
    top_k = 10
    seed_entity_ids: list[str] = []
    entity_relevance: dict[str, float] = {}

    entity_hits = await _graph_rag_vectors.search_entity_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=top_k,
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
            alias_result = await session.execute(
                select(EntityAlias)
                .join(GraphEntity, GraphEntity.id == EntityAlias.entity_id)
                .where(
                    or_(
                        EntityAlias.alias_name.ilike(f"% {kw} %"),
                        EntityAlias.alias_name.ilike(f"{kw} %"),
                        EntityAlias.alias_name.ilike(f"% {kw}"),
                        EntityAlias.alias_name.ilike(kw),
                    ),
                    GraphEntity.collection_id == collection.id,
                )
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
) -> list[tuple[str, str, float]]:
    hits = await _graph_rag_vectors.search_entity_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=top_k,
    )
    candidates: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for hit in hits:
        name = str(hit.metadata.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        candidates.append((name, hit.content.strip(), 1.0 - hit.distance))
    return candidates


async def _search_relationship_seeds(
    collection: Collection,
    query_embedding: list[float],
    *,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    hits = await _graph_rag_vectors.search_relationship_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=top_k,
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
) -> float:
    cached = cache.get(rel_id)
    if cached is not None:
        return cached
    rel_hits = await _graph_rag_vectors.search_relationship_embeddings(
        collection_id=collection.id,
        query_embedding=relationship_query_embedding,
        top_k=top_k,
        relationship_id=uuid.UUID(rel_id),
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

    graph_storage = get_graph_storage(collection.id)
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

        edges = await graph_storage.get_node_edges(node_id, rel_types=rel_types)
        scored_edges: list[tuple[float, str, str]] = []
        for src, tgt in edges:
            neighbor = tgt if src == node_id else src
            if neighbor in visited:
                continue

            edge_props = await graph_storage.get_edge(src, tgt, rel_types=rel_types)
            if not edge_props:
                edge_props = await graph_storage.get_edge(tgt, src, rel_types=rel_types)
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

        for combined, neighbor, rel_id_str in sorted(scored_edges, key=lambda x: x[0]):
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
    *,
    max_depth: int = 3,
    beam_width: int = 4,
    query_tokens: set[str] | None = None,
    rel_types: list[str] | None = None,
    dimension_weight: float = 1.0,
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

        edges = await graph_storage.get_node_edges(node_id, rel_types=rel_types)
        candidates: list[tuple[float, str, str]] = []
        for src, tgt in edges:
            neighbor = tgt if src == node_id else src
            if neighbor in path_nodes:
                continue

            edge_props = await graph_storage.get_edge(src, tgt, rel_types=rel_types)
            if not edge_props:
                edge_props = await graph_storage.get_edge(tgt, src, rel_types=rel_types)
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
    top_k: int = 10,
    max_endpoints: int = 20,
    max_pairs: int = 30,
    query_tokens: set[str] | None = None,
    rel_types: list[str] | None = None,
    dimension_weight: float = 1.0,
) -> GraphQueryState:
    rel_seeds = await _search_relationship_seeds(
        collection,
        relationship_query_embedding,
        top_k=top_k,
    )
    if query_tokens is None:
        query_tokens = set()

    traversed_rel_ids: list[str] = []
    rel_score_cache: dict[str, float] = {}
    rel_combined_score_cache: dict[str, float] = {}
    discovered_entity_ids: set[str] = set()
    entity_relevance: dict[str, float] = {}
    graph_storage = get_graph_storage(collection.id)

    async with AsyncSessionLocal() as session:
        for rel_id_str, sim in rel_seeds[:top_k]:
            rel = await session.get(GraphRelationship, uuid.UUID(rel_id_str))
            if not rel:
                continue
            edge_props = await graph_storage.get_edge(
                str(rel.source_entity_id),
                str(rel.target_entity_id),
                rel_types=rel_types,
            )
            if edge_props is None:
                edge_props = await graph_storage.get_edge(
                    str(rel.target_entity_id),
                    str(rel.source_entity_id),
                    rel_types=rel_types,
                )
            if not edge_props:
                continue
            combined = (
                _combined_edge_score(sim, edge_props, query_tokens)
                * dimension_weight
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
            query_tokens=query_tokens,
            rel_types=rel_types,
            dimension_weight=dimension_weight,
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
) -> GraphQueryState:
    query_tokens = _query_token_set(question)
    state = await _relationship_seed_state(
        collection,
        relationship_query_embedding,
        top_k=10,
        max_endpoints=20,
        max_pairs=30,
        query_tokens=query_tokens,
        rel_types=rel_types,
        dimension_weight=dimension_weight,
    )

    seed_entity_ids, seed_scores = await _search_entity_seeds(
        question,
        collection,
        entity_query_embedding,
    )
    for eid in seed_entity_ids[:5]:
        state.discovered_entity_ids.add(eid)
        if (
            eid not in state.entity_relevance
            or seed_scores[eid] > state.entity_relevance[eid]
        ):
            state.entity_relevance[eid] = seed_scores[eid]

    return state


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

    return MixInterpretation(
        selected_entities=selected_entities[:8],
        retrieval_subqueries=retrieval_subqueries[:4],
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
) -> GraphQueryState:
    query_tokens = _query_token_set(question)
    candidates = await _top_entity_candidates(
        collection,
        entity_query_embedding,
        top_k=50,
    )
    top_entity_score = candidates[0][2] if candidates else 0.0
    if top_entity_score < _MIX_REWRITE_MIN_SCORE:
        return await _relationship_seed_state(
            collection,
            relationship_query_embedding,
            top_k=10,
            max_endpoints=20,
            max_pairs=30,
            query_tokens=query_tokens,
            rel_types=rel_types,
            dimension_weight=dimension_weight,
        )

    llm_provider = await _resolve_llm_provider(
        namespace_id=namespace_id,
        llm_profile_id=llm_profile_id,
    )
    interpretation = await _interpret_mix_queries(question, candidates, llm_provider)

    states: list[GraphQueryState] = []
    for subquery in interpretation.retrieval_subqueries:
        subquery_embedding = await _embed_relationship_query(
            embedding_provider,
            subquery,
            rel_type=rel_types[0] if rel_types else None,
        )
        subquery_tokens = query_tokens | _query_token_set(subquery)
        states.append(
            await _relationship_seed_state(
                collection,
                subquery_embedding,
                top_k=8,
                max_endpoints=16,
                max_pairs=20,
                query_tokens=subquery_tokens,
                rel_types=rel_types,
                dimension_weight=dimension_weight,
            )
        )

    if not states:
        return await _relationship_first_state(
            question,
            collection,
            entity_query_embedding,
            relationship_query_embedding,
            rel_types=rel_types,
            dimension_weight=dimension_weight,
        )

    return _merge_states(*states)


async def _load_derived_context(
    question: str,
    collection: Collection,
    embedding_provider: EmbeddingProvider,
) -> DerivedContext:
    derived_query_embedding = await _embed_derived_query(embedding_provider, question)
    hits = await _vector_store.query_chunks(
        collection_id=collection.id,
        query_embedding=derived_query_embedding,
        top_k=6,
        metadata_filters={"memory_type": "derived_graph"},
    )
    if not hits:
        return DerivedContext(
            context="",
            base_entity_ids={},
            derived_nodes_used=[],
            derived_edges_used=[],
        )

    storage = FalkorDBGraphStorage(derived_graph_name(collection.id))
    summary_lines: list[str] = []
    graph_lines: list[str] = []
    base_entity_ids: dict[str, float] = {}
    derived_nodes_used: list[str] = []
    derived_edges_used: list[str] = []
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()

    for hit in hits:
        metadata = hit.get("metadata") or {}
        derived_id = str(metadata.get("derived_id") or "").strip()
        if not derived_id:
            continue
        score = float(hit.get("score") or 0.0)
        if derived_id not in seen_nodes:
            seen_nodes.add(derived_id)
            derived_nodes_used.append(derived_id)
        summary_lines.append(f"- {hit['content']}")

        node = await storage.get_node(derived_id)
        if not node:
            continue
        for source_id in _parse_source_ids(node.get("source_ids")):
            maybe_uuid = _maybe_uuid_string(source_id)
            if not maybe_uuid:
                continue
            if (
                maybe_uuid not in base_entity_ids
                or score > base_entity_ids[maybe_uuid]
            ):
                base_entity_ids[maybe_uuid] = score

        node_edges = await storage.get_node_edges(derived_id)
        for source_id, target_id in node_edges[:6]:
            edge_props = await storage.get_edge(source_id, target_id)
            if edge_props is None:
                edge_props = await storage.get_edge(target_id, source_id)
            if not edge_props:
                continue
            edge_id = str(edge_props.get("id") or f"{source_id}__{target_id}")
            if edge_id in seen_edges:
                continue
            seen_edges.add(edge_id)
            derived_edges_used.append(edge_id)
            target_node_id = target_id if source_id == derived_id else source_id
            target_node = await storage.get_node(target_node_id)
            target_name = (
                str(target_node.get("name") or target_node_id)
                if target_node
                else target_node_id
            )
            rel_type = str(edge_props.get("rel_type") or "RELATES_TO")
            description = str(edge_props.get("description") or "").strip()
            line = (
                f"- {str(node.get('name') or derived_id)} "
                f"-[{rel_type}]-> {target_name}"
            )
            if description:
                line += f": {description}"
            graph_lines.append(line)
            for source_ref in _parse_source_ids(edge_props.get("source_ids")):
                maybe_uuid = _maybe_uuid_string(source_ref)
                if not maybe_uuid:
                    continue
                if (
                    maybe_uuid not in base_entity_ids
                    or score > base_entity_ids[maybe_uuid]
                ):
                    base_entity_ids[maybe_uuid] = score
            if target_node:
                for source_ref in _parse_source_ids(target_node.get("source_ids")):
                    maybe_uuid = _maybe_uuid_string(source_ref)
                    if not maybe_uuid:
                        continue
                    if (
                        maybe_uuid not in base_entity_ids
                        or score > base_entity_ids[maybe_uuid]
                    ):
                        base_entity_ids[maybe_uuid] = score

    if not summary_lines and not graph_lines:
        return DerivedContext(
            context="",
            base_entity_ids={},
            derived_nodes_used=derived_nodes_used,
            derived_edges_used=derived_edges_used,
        )

    context_parts = []
    if summary_lines:
        context_parts.append("Derived Summaries:\n" + "\n".join(summary_lines))
    if graph_lines:
        context_parts.append("Derived Graph Links:\n" + "\n".join(graph_lines))

    return DerivedContext(
        context="\n\n".join(context_parts),
        base_entity_ids=base_entity_ids,
        derived_nodes_used=derived_nodes_used,
        derived_edges_used=derived_edges_used,
    )


async def _build_context(
    state: GraphQueryState,
    collection: Collection,
    *,
    derived_context: str = "",
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
            descs_result = await session.execute(
                select(EntityDescription)
                .where(EntityDescription.entity_id == eid)
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
            descs_result = await session.execute(
                select(RelationshipDescription)
                .where(RelationshipDescription.relationship_id == rel_uuid)
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
                "\n\nThe Derived Understanding section contains higher-level"
                " summaries synthesized from important communities,"
                " bridge nodes, and connector paths in the graph. Use it as"
                " high-level guidance, but ground specific claims in the"
                " entity and relationship evidence below."
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


async def graph_rag_query(
    question: str,
    collection: Collection,
    namespace_id: uuid.UUID,
    mode: str | None = None,
    llm_profile_id: uuid.UUID | None = None,
) -> QueryResult:
    embedding_provider = await _resolve_embedding_provider(collection)
    entity_query_embedding = await _embed_entity_query(embedding_provider, question)

    requested_mode = (mode or "mix").lower()
    effective_mode = _MODE_ALIASES.get(requested_mode, "mix")
    dimensions = _active_dimensions()

    async def _build_state_for(rel_type: str | None) -> GraphQueryState:
        relationship_query_embedding = await _embed_relationship_query(
            embedding_provider,
            question,
            rel_type=rel_type,
        )
        kwargs = {
            "rel_types": [rel_type] if rel_type else None,
            "dimension_weight": _dimension_weight(rel_type),
        }
        if effective_mode == "relationship-first":
            return await _relationship_first_state(
                question,
                collection,
                entity_query_embedding,
                relationship_query_embedding,
                **kwargs,
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
            )
        if effective_mode == "hybrid":
            entity_state = await _entity_first_state(
                question,
                collection,
                entity_query_embedding,
                relationship_query_embedding,
                **kwargs,
            )
            relationship_state = await _relationship_first_state(
                question,
                collection,
                entity_query_embedding,
                relationship_query_embedding,
                **kwargs,
            )
            return _merge_states(entity_state, relationship_state)
        return await _entity_first_state(
            question,
            collection,
            entity_query_embedding,
            relationship_query_embedding,
            **kwargs,
        )

    state = await _fan_out_per_dimension(_build_state_for, dimensions)
    if state is None:
        # No dimensions yielded any state — synthesize a minimal empty
        # state so the context assembler still returns a structured
        # QueryResult.
        state = GraphQueryState(
            discovered_entity_ids=set(),
            entity_relevance={},
            traversed_rel_ids=[],
            rel_score_cache={},
            rel_combined_score_cache={},
        )
    if effective_mode not in ("relationship-first", "mix", "hybrid"):
        effective_mode = "entity-first"

    derived = await _load_derived_context(
        question,
        collection,
        embedding_provider,
    )
    for eid, score in derived.base_entity_ids.items():
        state.discovered_entity_ids.add(eid)
        if eid not in state.entity_relevance or score > state.entity_relevance[eid]:
            state.entity_relevance[eid] = score

    context, entities_used, relationships_used, rel_context = await _build_context(
        state,
        collection,
        derived_context=derived.context,
    )
    entity_fallback = "\n".join(entities_used)
    fallback_text = derived.context or rel_context or entity_fallback
    response = await _answer_from_context(
        question,
        namespace_id,
        llm_profile_id,
        context,
        fallback_text,
    )
    return QueryResult(
        response=response,
        entities_used=entities_used,
        relationships_used=list(
            dict.fromkeys(relationships_used + derived.derived_edges_used)
        ),
        mode=effective_mode,
    )

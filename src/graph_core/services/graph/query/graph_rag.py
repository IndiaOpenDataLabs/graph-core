"""Graph RAG query functions extracted from GraphService."""

from __future__ import annotations

import asyncio
import inspect
import logging
import string
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

from sqlalchemy import distinct, func, or_, select, text

from graph_core.config import settings
from graph_core.database import AsyncSessionLocal, _uuid_for_sql
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
_MIX_REWRITE_MIN_SCORE = 0.3
_REL_ENDPOINT_ENTITY_SCORE_MIN = 0.0
_META_PROJECTION_ENTITY_SCORE = 0.96
_META_PROJECTION_EDGE_BASE_SCORE = 0.72
_META_PROJECTION_MAX_BASE_REFS = 40
_META_PROJECTION_MAX_BASE_RELS = 80
_CONTEXT_MIX_TOP_K = 40
_CONTEXT_MIX_MAX_CONTEXTS = 8
_COLLECTION_COVERAGE_MAX_DOCUMENTS = 200
_COLLECTION_COVERAGE_CONTEXTS_PER_DOCUMENT = 4
_COLLECTION_COVERAGE_ASSERTIONS_PER_CONTEXT = 16
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
class GraphQueryPlan:
    operation: str
    scope: str
    anchors: list[str]
    requested_fields: list[str]
    output_shape: str


@dataclass
class GraphQueryFramePlan:
    focus_terms: list[str]
    competing_terms: list[str]
    relation_hints: list[str]


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
    state: GraphQueryState


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


@dataclass
class ContextEvidenceCandidate:
    context_id: uuid.UUID
    name: str
    document_path: str
    description: str
    score: float
    reasons: list[str]


@dataclass
class ContextAssertionEvidence:
    context_id: uuid.UUID
    context_name: str
    document_path: str
    assertion_id: uuid.UUID
    assertion: str
    evidence: str


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


# ---------------------------------------------------------------------------
# Entity mention index — ngram-based, cached per collection
# ---------------------------------------------------------------------------

_MENTION_INDEX_TTL_SECONDS = 300  # 5 minutes
_MENTION_INDEX_MAX_NGRAM = 4


@dataclass
class _MentionMatch:
    entity_id: str
    canonical_name: str
    score: float  # 1.0 for canonical match, 0.98 for alias match


class _EntityMentionIndex:
    """Pre-loaded entity/alias name index for fast ngram-based mention detection.

    Instead of iterating all entities and doing substring matching per entity,
    this tokenizes the query into 1-4 word ngrams and looks them up in a set
    of known entity names. O(question_length^2) lookups in a hash set vs
    O(entity_count * question_length) substring scans.
    """

    def __init__(
        self,
        entity_rows: list[tuple[Any, str, str]],
        alias_rows: list[tuple[Any, str]],
    ) -> None:
        # canonical_key -> (entity_id_str, canonical_name)
        self._canonical_map: dict[str, tuple[str, str]] = {}
        # alias_key -> (entity_id_str, canonical_name)
        self._alias_map: dict[str, tuple[str, str]] = {}
        self._entity_name_by_id: dict[str, str] = {}

        for entity_id, canonical_name, _primary_type in entity_rows:
            name = str(canonical_name or "").strip()
            if not name or len(name) < 3:
                continue
            entity_id_str = str(entity_id)
            self._entity_name_by_id[entity_id_str] = name
            key = self._normalize_key(name)
            if key:
                self._canonical_map[key] = (entity_id_str, name)

        for entity_id, alias_name in alias_rows:
            alias = str(alias_name or "").strip()
            if not alias or len(alias) < 3:
                continue
            entity_id_str = str(entity_id)
            key = self._normalize_key(alias)
            if key and key not in self._canonical_map:
                canonical = self._entity_name_by_id.get(entity_id_str, alias)
                self._alias_map.setdefault(key, (entity_id_str, canonical))

    @staticmethod
    def _normalize_key(name: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        lowered = name.casefold()
        cleaned = lowered.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        return " ".join(cleaned.split())

    @staticmethod
    def _question_ngrams(question: str) -> list[str]:
        """Generate 1 to _MENTION_INDEX_MAX_NGRAM word ngrams from the question."""
        cleaned = question.casefold()
        cleaned = cleaned.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        words = cleaned.split()
        ngrams: list[str] = []
        for n in range(1, min(_MENTION_INDEX_MAX_NGRAM + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                ngrams.append(" ".join(words[i : i + n]))
        return ngrams

    def find_mentions(self, question: str, *, limit: int = 20) -> list[_MentionMatch]:
        """Find entities whose names appear as ngrams in the question."""
        ngrams = self._question_ngrams(question)
        matched: dict[str, _MentionMatch] = {}

        for ngram in ngrams:
            if ngram in self._canonical_map:
                entity_id, name = self._canonical_map[ngram]
                if entity_id not in matched or matched[entity_id].score < 1.0:
                    matched[entity_id] = _MentionMatch(
                        entity_id=entity_id,
                        canonical_name=name,
                        score=1.0,
                    )
            elif ngram in self._alias_map:
                entity_id, name = self._alias_map[ngram]
                if entity_id not in matched:
                    matched[entity_id] = _MentionMatch(
                        entity_id=entity_id,
                        canonical_name=name,
                        score=0.98,
                    )

        results = sorted(matched.values(), key=lambda m: (-m.score, -len(m.canonical_name)))
        return results[:limit]


# Module-level cache: collection_id -> (index, timestamp)
_mention_index_cache: dict[uuid.UUID, tuple[_EntityMentionIndex, float]] = {}


async def _get_mention_index(collection: Collection) -> _EntityMentionIndex:
    """Return (possibly cached) mention index for a collection."""
    now = time.monotonic()
    cached = _mention_index_cache.get(collection.id)
    if cached is not None:
        index, created_at = cached
        if now - created_at < _MENTION_INDEX_TTL_SECONDS:
            return index

    async with AsyncSessionLocal() as session:
        entity_rows = (
            await session.execute(
                select(
                    GraphEntity.id,
                    GraphEntity.canonical_name,
                    GraphEntity.primary_type,
                ).where(GraphEntity.collection_id == collection.id)
            )
        ).all()
        alias_rows = (
            await session.execute(
                select(
                    EntityAlias.entity_id,
                    EntityAlias.alias_name,
                ).where(EntityAlias.collection_id == collection.id)
            )
        ).all()

    index = _EntityMentionIndex(entity_rows, alias_rows)
    _mention_index_cache[collection.id] = (index, now)
    logger.info(
        "graph_rag mention_index_built collection=%s entities=%d aliases=%d",
        collection.name,
        len(entity_rows),
        len(alias_rows),
    )
    return index


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
    mention_index: _EntityMentionIndex | None = None,
    limit: int = 20,
) -> list[DocumentRoutingCandidate]:
    query_embedding = await _embed_entity_query(embedding_provider, question)
    if mention_index is None:
        mention_index = await _get_mention_index(collection)
    mentions = mention_index.find_mentions(question, limit=10)
    entity_hits = await _graph_rag_vectors.search_entity_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=50,
    )

    candidate_map: dict[str, DocumentRoutingCandidate] = {}
    async with AsyncSessionLocal() as session:
        for match in mentions:
            desc_rows = (
                await session.execute(
                    select(
                        EntityDescription.document_id,
                    ).where(EntityDescription.entity_id == uuid.UUID(match.entity_id))
                )
            ).all()
            for (document_id,) in desc_rows:
                if document_id is None:
                    continue
                document_id_str = str(document_id)
                candidate = candidate_map.get(document_id_str)
                if candidate is None:
                    candidate_map[document_id_str] = DocumentRoutingCandidate(
                        document_id=document_id_str,
                        document_path="",
                        best_score=match.score,
                        matched_entities=[match.canonical_name],
                    )
                    continue
                if match.score > candidate.best_score:
                    candidate.best_score = match.score
                if match.canonical_name not in candidate.matched_entities:
                    candidate.matched_entities.append(match.canonical_name)

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
    mention_index: _EntityMentionIndex | None = None,
) -> tuple[list[str], dict[str, float]]:
    top_k = 20
    seed_entity_ids: list[str] = []
    entity_relevance: dict[str, float] = {}

    if mention_index is None:
        mention_index = await _get_mention_index(collection)
    for match in mention_index.find_mentions(question, limit=20):
        if match.entity_id not in seed_entity_ids:
            seed_entity_ids.append(match.entity_id)
            entity_relevance[match.entity_id] = match.score

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
    question: str = "",
    top_k: int = 50,
    document_ids: list[uuid.UUID] | None = None,
    mention_index: _EntityMentionIndex | None = None,
) -> list[tuple[str, str, float]]:
    hits = await _graph_rag_vectors.search_entity_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=top_k,
        document_ids=document_ids,
    )
    candidates: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    if question:
        if mention_index is None:
            mention_index = await _get_mention_index(collection)
        for match in mention_index.find_mentions(question, limit=20):
            name = match.canonical_name
            if not name or name in seen:
                continue
            seen.add(name)
            candidates.append((name, name, match.score))
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
    min_similarity: float | None = None,
    document_ids: list[uuid.UUID] | None = None,
) -> list[tuple[str, float]]:
    hits = await _graph_rag_vectors.search_relationship_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=top_k,
        document_ids=document_ids,
    )
    threshold = settings.graph_rag_min_edge_similarity if min_similarity is None else min_similarity
    rel_seeds: list[tuple[str, float]] = []
    seen: set[str] = set()
    for hit in hits:
        rel_id = hit.metadata.get("relationship_id") or hit.metadata.get("id")
        if not rel_id or rel_id in seen:
            continue
        sim = 1.0 - hit.distance
        if sim < threshold:
            continue
        seen.add(rel_id)
        rel_seeds.append((str(rel_id), sim))
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
    mention_index: _EntityMentionIndex | None = None,
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
        mention_index=mention_index,
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
        min_similarity=settings.graph_rag_min_edge_similarity,
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
    mention_index: _EntityMentionIndex | None = None,
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
        question=question,
        min_entity_score=_REL_ENDPOINT_ENTITY_SCORE_MIN,
        document_ids=document_ids,
        mention_index=mention_index,
    )


async def _entity_anchor_state(
    collection: Collection,
    entity_names: list[str],
    *,
    query_tokens: set[str],
    document_ids: list[uuid.UUID] | None = None,
    max_relationships: int = 40,
) -> GraphQueryState:
    wanted_names = [name for name in entity_names if str(name).strip()]
    if not wanted_names:
        return GraphQueryState(
            discovered_entity_ids=set(),
            entity_relevance={},
            traversed_rel_ids=[],
            rel_score_cache={},
            rel_combined_score_cache={},
        )

    wanted_keys = {name.casefold() for name in wanted_names}
    discovered_entity_ids: set[str] = set()
    entity_relevance: dict[str, float] = {}
    traversed_rel_ids: list[str] = []
    rel_score_cache: dict[str, float] = {}
    rel_combined_score_cache: dict[str, float] = {}

    async with AsyncSessionLocal() as session:
        entity_rows = (
            await session.execute(
                select(GraphEntity.id, GraphEntity.canonical_name).where(
                    GraphEntity.collection_id == collection.id,
                    func.lower(GraphEntity.canonical_name).in_(
                        {key.lower() for key in wanted_keys}
                    ),
                )
            )
        ).all()
        alias_rows = (
            await session.execute(
                select(EntityAlias.entity_id, EntityAlias.alias_name)
                .join(GraphEntity, GraphEntity.id == EntityAlias.entity_id)
                .where(
                    EntityAlias.collection_id == collection.id,
                    GraphEntity.collection_id == collection.id,
                    func.lower(EntityAlias.alias_name).in_(
                        {key.lower() for key in wanted_keys}
                    ),
                )
            )
        ).all()
        anchor_ids = {str(entity_id) for entity_id, _ in entity_rows}
        anchor_ids.update(str(entity_id) for entity_id, _ in alias_rows)
        if not anchor_ids:
            return GraphQueryState(
                discovered_entity_ids=set(),
                entity_relevance={},
                traversed_rel_ids=[],
                rel_score_cache={},
                rel_combined_score_cache={},
            )

        for entity_id in anchor_ids:
            discovered_entity_ids.add(entity_id)
            entity_relevance[entity_id] = 1.0

        rel_conditions = [
            GraphRelationship.collection_id == collection.id,
            or_(
                GraphRelationship.source_entity_id.in_(
                    [uuid.UUID(entity_id) for entity_id in anchor_ids]
                ),
                GraphRelationship.target_entity_id.in_(
                    [uuid.UUID(entity_id) for entity_id in anchor_ids]
                ),
            ),
        ]
        if document_ids:
            rel_conditions.append(
                GraphRelationship.id.in_(
                    select(RelationshipDescription.relationship_id).where(
                        RelationshipDescription.document_id.in_(document_ids)
                    )
                )
            )
        rel_rows = (
            await session.execute(
                select(GraphRelationship)
                .where(*rel_conditions)
                .order_by(GraphRelationship.weight.desc())
                .limit(max_relationships)
            )
        ).scalars().all()

    for rel in rel_rows:
        rel_id = str(rel.id)
        edge_props = {
            "weight": rel.weight or 1,
            "keywords": rel.keywords or [],
            "rel_type": rel.rel_type,
        }
        combined = _combined_edge_score(0.95, edge_props, query_tokens)
        traversed_rel_ids.append(rel_id)
        rel_score_cache[rel_id] = 0.95
        rel_combined_score_cache[rel_id] = combined
        for endpoint_id in (str(rel.source_entity_id), str(rel.target_entity_id)):
            discovered_entity_ids.add(endpoint_id)
            entity_relevance[endpoint_id] = max(
                entity_relevance.get(endpoint_id, 0.0),
                combined,
            )

    return GraphQueryState(
        discovered_entity_ids=discovered_entity_ids,
        entity_relevance=entity_relevance,
        traversed_rel_ids=traversed_rel_ids,
        rel_score_cache=rel_score_cache,
        rel_combined_score_cache=rel_combined_score_cache,
    )


async def _filter_relationship_state_by_entity_score(
    collection: Collection,
    state: GraphQueryState,
    entity_query_embedding: list[float],
    *,
    question: str = "",
    min_entity_score: float,
    top_k: int = 50,
    document_ids: list[uuid.UUID] | None = None,
    mention_index: _EntityMentionIndex | None = None,
) -> GraphQueryState:
    if not state.discovered_entity_ids:
        return state

    candidates = await _top_entity_candidates(
        collection,
        entity_query_embedding,
        question=question,
        top_k=top_k,
        document_ids=document_ids,
        mention_index=mention_index,
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


def _vector_hit_score(hit) -> float:
    return max(0.0, 1.0 - float(hit.distance))


def _uuid_from_value(value: Any) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None


def _normalise_plan_list(value: Any, *, max_items: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for raw_item in value:
        item = str(raw_item or "").strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        items.append(item)
        seen.add(key)
        if len(items) >= max_items:
            break
    return items


def _expand_frame_terms(terms: list[str], *, max_items: int = 16) -> list[str]:
    expanded = list(terms)
    token_counts: dict[str, int] = {}
    for term in terms:
        tokens = {
            "".join(ch for ch in raw_token.casefold() if ch.isalnum())
            for raw_token in term.split()
        }
        for token in tokens:
            if len(token) >= 4:
                token_counts[token] = token_counts.get(token, 0) + 1

    seen = {term.casefold() for term in expanded}
    for token, count in sorted(
        token_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        if count < 2 or token in seen:
            continue
        expanded.append(token)
        seen.add(token)
        if len(expanded) >= max_items:
            break
    return expanded[:max_items]


def _fallback_graph_query_plan(question: str) -> GraphQueryPlan:
    lowered = f" {question.casefold()} "
    operation = "describe"
    scope = "top_k"
    output_shape = "table" if " table " in lowered else "prose"

    if any(
        marker in lowered
        for marker in (" compare ", " different ", " difference ", " versus ", " vs ")
    ):
        operation = "compare"
        scope = "anchored"
    if any(
        marker in lowered
        for marker in (
            " all ",
            " every ",
            " each ",
            " list ",
            " enumerate ",
            " inventory ",
            " coverage ",
        )
    ):
        operation = "inventory"
        scope = "collection"

    return GraphQueryPlan(
        operation=operation,
        scope=scope,
        anchors=[],
        requested_fields=[],
        output_shape=output_shape,
    )


def _fallback_graph_query_frame_plan(
    question: str,
    plan: GraphQueryPlan | None = None,
) -> GraphQueryFramePlan:
    terms = list(plan.requested_fields) if plan else []
    if not terms:
        terms = [_diagnostic_entity_text(question)]
    return GraphQueryFramePlan(
        focus_terms=_expand_frame_terms(terms),
        competing_terms=[],
        relation_hints=[],
    )


async def _plan_graph_query(
    question: str,
    namespace_id: uuid.UUID,
    llm_profile_id: uuid.UUID | None,
) -> GraphQueryPlan:
    fallback = _fallback_graph_query_plan(question)
    llm_provider = await _resolve_llm_provider(
        namespace_id=namespace_id,
        llm_profile_id=llm_profile_id,
    )
    if isinstance(llm_provider, LocalEchoLLMProvider):
        return fallback

    schema = {
        "title": "GraphQueryPlan",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["describe", "compare", "inventory", "aggregate"],
            },
            "scope": {
                "type": "string",
                "enum": ["top_k", "anchored", "collection"],
            },
            "anchors": {"type": "array", "items": {"type": "string"}},
            "requested_fields": {"type": "array", "items": {"type": "string"}},
            "output_shape": {
                "type": "string",
                "enum": ["prose", "bullets", "table"],
            },
        },
        "required": [
            "operation",
            "scope",
            "anchors",
            "requested_fields",
            "output_shape",
        ],
    }
    prompt = (
        "Plan graph retrieval for the user's question without assuming any "
        "domain schema.\n\n"
        "Fields:\n"
        "- operation: describe, compare, inventory, or aggregate.\n"
        "- scope: top_k, anchored, or collection.\n"
        "- anchors: exact named literals explicitly mentioned by the user. "
        "Include proper names, acronyms, source names, titles, APIs, symbols, "
        "chapter names, or other concrete identifiers. Do not include generic "
        "requested fields such as role, company, chapter, function, argument, "
        "skill, requirement, theme, document, or concept unless the user uses "
        "that word as a proper name.\n"
        "- requested_fields: generic information fields the user asks to "
        "extract, list, compare, or aggregate, stated in the user's terms. "
        "These are not anchors.\n"
        "- output_shape: prose, bullets, or table.\n\n"
        "Scope rules:\n"
        "- collection: the user asks for all/every/list/inventory/table/coverage "
        "over the collection.\n"
        "- anchored: the user asks about named anchors and does not need full "
        "collection coverage.\n"
        "- top_k: otherwise.\n\n"
        f"Question: {question}"
    )
    try:
        extracted = await llm_provider.structured_extract(prompt, schema)
    except Exception:
        logger.exception("graph_rag query_plan_failed")
        return fallback

    operation = str(extracted.get("operation") or fallback.operation).strip().lower()
    if operation not in {"describe", "compare", "inventory", "aggregate"}:
        operation = fallback.operation
    scope = str(extracted.get("scope") or fallback.scope).strip().lower()
    if scope not in {"top_k", "anchored", "collection"}:
        scope = fallback.scope
    output_shape = str(
        extracted.get("output_shape") or fallback.output_shape
    ).strip().lower()
    if output_shape not in {"prose", "bullets", "table"}:
        output_shape = fallback.output_shape
    anchors = _normalise_plan_list(extracted.get("anchors"))
    requested_fields = _normalise_plan_list(extracted.get("requested_fields"))
    if operation == "compare" and anchors:
        scope = "anchored"

    plan = GraphQueryPlan(
        operation=operation,
        scope=scope,
        anchors=anchors,
        requested_fields=requested_fields,
        output_shape=output_shape,
    )
    logger.info(
        "graph_rag query_plan operation=%s scope=%s anchors=%s fields=%s shape=%s",
        plan.operation,
        plan.scope,
        plan.anchors,
        plan.requested_fields,
        plan.output_shape,
    )
    return plan


async def _plan_graph_query_frame(
    question: str,
    namespace_id: uuid.UUID,
    llm_profile_id: uuid.UUID | None,
    plan: GraphQueryPlan | None = None,
) -> GraphQueryFramePlan:
    fallback = _fallback_graph_query_frame_plan(question, plan)
    llm_provider = await _resolve_llm_provider(
        namespace_id=namespace_id,
        llm_profile_id=llm_profile_id,
    )
    if isinstance(llm_provider, LocalEchoLLMProvider):
        return fallback

    schema = {
        "title": "GraphQueryFramePlan",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "focus_terms": {"type": "array", "items": {"type": "string"}},
            "competing_terms": {"type": "array", "items": {"type": "string"}},
            "relation_hints": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["focus_terms", "competing_terms", "relation_hints"],
    }
    prompt = (
        "Identify the graph retrieval frame for this question without assuming "
        "any domain schema. A frame is the topic, process, source scope, or "
        "conceptual region the answer should stay inside; it is not a named "
        "entity anchor.\n\n"
        "Return:\n"
        "- focus_terms: literal terms and close variants that define the target "
        "context. Use the user's wording where possible.\n"
        "- competing_terms: nearby frames likely to be confused with the focus "
        "and should be demoted when evidence is mostly about them.\n"
        "- relation_hints: generic action/edge words that describe how evidence "
        "inside the frame connects, without inventing a domain schema.\n\n"
        "Do not include output-control words unless they are domain terms. "
        "Do not include named anchors from the question as focus terms.\n\n"
        f"Question: {question}\n"
        f"Query plan requested_fields: {plan.requested_fields if plan else []}\n"
        f"Query plan anchors: {plan.anchors if plan else []}"
    )
    try:
        extracted = await llm_provider.structured_extract(prompt, schema)
    except Exception:
        logger.exception("graph_rag frame_plan_failed")
        return fallback

    focus_terms = _normalise_plan_list(extracted.get("focus_terms"), max_items=10)
    competing_terms = _normalise_plan_list(
        extracted.get("competing_terms"),
        max_items=12,
    )
    relation_hints = _normalise_plan_list(
        extracted.get("relation_hints"),
        max_items=12,
    )
    frame_plan = GraphQueryFramePlan(
        focus_terms=_expand_frame_terms(focus_terms or fallback.focus_terms),
        competing_terms=competing_terms,
        relation_hints=relation_hints,
    )
    logger.info(
        "graph_rag frame_plan focus=%s competing=%s relations=%s",
        frame_plan.focus_terms,
        frame_plan.competing_terms,
        frame_plan.relation_hints,
    )
    return frame_plan


async def _collection_has_context_layer(collection: Collection) -> bool:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(GraphEntity.id)
            .where(
                GraphEntity.collection_id == collection.id,
                GraphEntity.primary_type == "CONTEXT",
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None


async def _contexts_for_document_paths(
    *,
    collection_id: uuid.UUID,
    document_paths: dict[str, float],
    document_ids: list[uuid.UUID] | None = None,
) -> dict[uuid.UUID, ContextEvidenceCandidate]:
    if not document_paths:
        return {}
    params: dict[str, Any] = {"cid": _uuid_for_sql(collection_id)}
    placeholders: list[str] = []
    for index, path in enumerate(document_paths):
        key = f"path_{index}"
        params[key] = path
        placeholders.append(f":{key}")
    doc_filter = ""
    if document_ids:
        doc_placeholders: list[str] = []
        for index, document_id in enumerate(document_ids):
            key = f"document_id_{index}"
            params[key] = _uuid_for_sql(document_id)
            doc_placeholders.append(f":{key}")
        doc_filter = f" AND ed.document_id IN ({', '.join(doc_placeholders)})"

    async with AsyncSessionLocal() as session:
        rows = await session.execute(
            text(
                f"""
                SELECT e.id, e.canonical_name, ed.document_path, ed.description
                FROM graph_entities e
                JOIN entity_descriptions ed ON ed.entity_id = e.id
                WHERE e.collection_id = :cid
                  AND e.primary_type = 'CONTEXT'
                  AND ed.document_path IN ({", ".join(placeholders)})
                  {doc_filter}
                """
            ),
            params,
        )

    candidates: dict[uuid.UUID, ContextEvidenceCandidate] = {}
    for row in rows:
        context_id = uuid.UUID(str(row[0]))
        document_path = str(row[2] or "")
        candidates[context_id] = ContextEvidenceCandidate(
            context_id=context_id,
            name=str(row[1]),
            document_path=document_path,
            description=str(row[3] or ""),
            score=document_paths.get(document_path, 0.0) * 0.85,
            reasons=[f"same document as vector hit ({document_path})"],
        )
    return candidates


async def _contexts_for_graph_hits(
    *,
    collection_id: uuid.UUID,
    entity_scores: dict[uuid.UUID, float],
    relationship_scores: dict[uuid.UUID, float],
    document_ids: list[uuid.UUID] | None = None,
) -> dict[uuid.UUID, ContextEvidenceCandidate]:
    if not entity_scores and not relationship_scores:
        return {}

    params: dict[str, Any] = {"cid": _uuid_for_sql(collection_id)}
    entity_placeholders: list[str] = []
    relationship_placeholders: list[str] = []
    for index, entity_id in enumerate(entity_scores):
        key = f"entity_{index}"
        params[key] = _uuid_for_sql(entity_id)
        entity_placeholders.append(f":{key}")
    for index, relationship_id in enumerate(relationship_scores):
        key = f"relationship_{index}"
        params[key] = _uuid_for_sql(relationship_id)
        relationship_placeholders.append(f":{key}")
    entity_clause = ", ".join(entity_placeholders) or "NULL"
    relationship_clause = ", ".join(relationship_placeholders) or "NULL"

    doc_filter = ""
    if document_ids:
        doc_placeholders: list[str] = []
        for index, document_id in enumerate(document_ids):
            key = f"context_document_id_{index}"
            params[key] = _uuid_for_sql(document_id)
            doc_placeholders.append(f":{key}")
        doc_filter = f" AND ed.document_id IN ({', '.join(doc_placeholders)})"

    async with AsyncSessionLocal() as session:
        rows = await session.execute(
            text(
                f"""
                WITH candidate_contexts AS (
                    SELECT e.id AS context_id, e.id AS matched_entity_id,
                           NULL::uuid AS matched_relationship_id,
                           'direct context entity hit'::text AS reason
                    FROM graph_entities e
                    WHERE e.collection_id = :cid
                      AND e.primary_type = 'CONTEXT'
                      AND e.id IN ({entity_clause})

                    UNION ALL

                    SELECT ctx.id AS context_id,
                           owns.target_entity_id AS matched_entity_id,
                           NULL::uuid AS matched_relationship_id,
                           'matched assertion owned by context'::text AS reason
                    FROM graph_relationships owns
                    JOIN graph_entities ctx ON ctx.id = owns.source_entity_id
                    WHERE owns.collection_id = :cid
                      AND owns.rel_type = 'HAS_ASSERTION'
                      AND ctx.primary_type = 'CONTEXT'
                      AND owns.target_entity_id IN ({entity_clause})

                    UNION ALL

                    SELECT ctx.id AS context_id,
                           edge.target_entity_id AS matched_entity_id,
                           NULL::uuid AS matched_relationship_id,
                           'matched node attached to context'::text AS reason
                    FROM graph_relationships edge
                    JOIN graph_entities ctx ON ctx.id = edge.source_entity_id
                    WHERE edge.collection_id = :cid
                      AND ctx.primary_type = 'CONTEXT'
                      AND edge.target_entity_id IN ({entity_clause})

                    UNION ALL

                    SELECT ctx.id AS context_id,
                           CASE
                             WHEN ar.source_entity_id IN ({entity_clause})
                               THEN ar.source_entity_id
                             ELSE ar.target_entity_id
                           END AS matched_entity_id,
                           NULL::uuid AS matched_relationship_id,
                           'matched node connected to context assertion'::text
                           AS reason
                    FROM graph_relationships owns
                    JOIN graph_entities ctx ON ctx.id = owns.source_entity_id
                    JOIN graph_relationships ar
                      ON ar.source_entity_id = owns.target_entity_id
                      OR ar.target_entity_id = owns.target_entity_id
                    WHERE owns.collection_id = :cid
                      AND owns.rel_type = 'HAS_ASSERTION'
                      AND ctx.primary_type = 'CONTEXT'
                      AND (ar.source_entity_id IN ({entity_clause})
                           OR ar.target_entity_id IN ({entity_clause}))

                    UNION ALL

                    SELECT ctx.id AS context_id, NULL::uuid AS matched_entity_id,
                           edge.id AS matched_relationship_id,
                           'matched relationship attached to context'::text
                           AS reason
                    FROM graph_relationships edge
                    JOIN graph_entities ctx
                      ON ctx.id = edge.source_entity_id
                      OR ctx.id = edge.target_entity_id
                    WHERE edge.collection_id = :cid
                      AND ctx.primary_type = 'CONTEXT'
                      AND edge.id IN ({relationship_clause})

                    UNION ALL

                    SELECT ctx.id AS context_id, NULL::uuid AS matched_entity_id,
                           edge.id AS matched_relationship_id,
                           'matched relationship attached to owned assertion'::text
                           AS reason
                    FROM graph_relationships owns
                    JOIN graph_entities ctx ON ctx.id = owns.source_entity_id
                    JOIN graph_relationships edge
                      ON edge.source_entity_id = owns.target_entity_id
                      OR edge.target_entity_id = owns.target_entity_id
                    WHERE owns.collection_id = :cid
                      AND owns.rel_type = 'HAS_ASSERTION'
                      AND ctx.primary_type = 'CONTEXT'
                      AND edge.id IN ({relationship_clause})
                )
                SELECT c.id, c.canonical_name, ed.document_path, ed.description,
                       cc.matched_entity_id, cc.matched_relationship_id,
                       cc.reason
                FROM candidate_contexts cc
                JOIN graph_entities c ON c.id = cc.context_id
                JOIN entity_descriptions ed ON ed.entity_id = c.id
                WHERE c.collection_id = :cid
                  {doc_filter}
                """
            ),
            params,
        )

    candidates: dict[uuid.UUID, ContextEvidenceCandidate] = {}
    for row in rows:
        context_id = uuid.UUID(str(row[0]))
        entity_id = _uuid_from_value(row[4])
        relationship_id = _uuid_from_value(row[5])
        score = 0.0
        if entity_id is not None:
            score = max(score, entity_scores.get(entity_id, 0.0))
        if relationship_id is not None:
            score = max(score, relationship_scores.get(relationship_id, 0.0))
        if score <= 0:
            continue
        candidate = candidates.get(context_id)
        if candidate is None:
            candidate = ContextEvidenceCandidate(
                context_id=context_id,
                name=str(row[1]),
                document_path=str(row[2] or ""),
                description=str(row[3] or ""),
                score=0.0,
                reasons=[],
            )
            candidates[context_id] = candidate
        candidate.score += score
        reason = str(row[6] or "")
        if reason and reason not in candidate.reasons:
            candidate.reasons.append(reason)
    return candidates


async def _contexts_for_anchor_literals(
    *,
    collection_id: uuid.UUID,
    anchors: list[str],
    document_ids: list[uuid.UUID] | None = None,
) -> list[ContextEvidenceCandidate]:
    if not anchors:
        return []

    params: dict[str, Any] = {"cid": _uuid_for_sql(collection_id)}
    context_clauses: list[str] = []
    graph_clauses: list[str] = []
    for index, anchor in enumerate(anchors):
        key = f"anchor_{index}"
        params[key] = f"%{anchor}%"
        context_clauses.append(
            f"""
            ctx.canonical_name ILIKE :{key}
            OR ctx_ed.description ILIKE :{key}
            OR ctx_ed.document_path ILIKE :{key}
            """
        )
        graph_clauses.append(
            f"""
            e.canonical_name ILIKE :{key}
            OR ed.description ILIKE :{key}
            OR assertion.canonical_name ILIKE :{key}
            OR assertion_ed.description ILIKE :{key}
            """
        )

    doc_filter = ""
    if document_ids:
        doc_placeholders: list[str] = []
        for index, document_id in enumerate(document_ids):
            key = f"anchor_document_id_{index}"
            params[key] = _uuid_for_sql(document_id)
            doc_placeholders.append(f":{key}")
        doc_filter = f" AND ctx_ed.document_id IN ({', '.join(doc_placeholders)})"

    async with AsyncSessionLocal() as session:
        rows = await session.execute(
            text(
                f"""
                WITH matched_contexts AS (
                    SELECT
                        ctx.id,
                        ctx.canonical_name,
                        ctx_ed.document_path,
                        ctx_ed.description,
                        5.0 AS score
                    FROM graph_entities ctx
                    JOIN entity_descriptions ctx_ed ON ctx_ed.entity_id = ctx.id
                    WHERE ctx.collection_id = :cid
                      AND ctx.primary_type = 'CONTEXT'
                      AND (
                        {' OR '.join(f'({clause})' for clause in context_clauses)}
                      )
                      {doc_filter}

                    UNION ALL

                    SELECT
                        ctx.id,
                        ctx.canonical_name,
                        ctx_ed.document_path,
                        ctx_ed.description,
                        CASE
                          WHEN e.primary_type LIKE 'CONCEPT_%' THEN 4.0
                          WHEN e.primary_type LIKE 'MENTION_%' THEN 3.0
                          WHEN assertion.id IS NOT NULL THEN 2.0
                          ELSE 1.0
                        END AS score
                    FROM graph_entities e
                    JOIN entity_descriptions ed ON ed.entity_id = e.id
                    LEFT JOIN graph_relationships edge
                      ON edge.collection_id = :cid
                     AND (edge.source_entity_id = e.id OR edge.target_entity_id = e.id)
                    LEFT JOIN graph_relationships owns
                      ON owns.collection_id = :cid
                     AND owns.rel_type = 'HAS_ASSERTION'
                     AND (
                        owns.target_entity_id = e.id
                        OR owns.target_entity_id = edge.source_entity_id
                        OR owns.target_entity_id = edge.target_entity_id
                     )
                    LEFT JOIN graph_entities assertion
                      ON assertion.id = owns.target_entity_id
                    LEFT JOIN entity_descriptions assertion_ed
                      ON assertion_ed.entity_id = assertion.id
                    JOIN graph_entities ctx ON ctx.id = owns.source_entity_id
                    JOIN entity_descriptions ctx_ed ON ctx_ed.entity_id = ctx.id
                    WHERE e.collection_id = :cid
                      AND ctx.primary_type = 'CONTEXT'
                      AND (
                        {' OR '.join(f'({clause})' for clause in graph_clauses)}
                      )
                      {doc_filter}
                )
                SELECT id, canonical_name, document_path, description, score
                FROM matched_contexts
                ORDER BY document_path, score DESC, canonical_name
                """
            ),
            params,
        )

    candidates: dict[uuid.UUID, ContextEvidenceCandidate] = {}
    for row in rows:
        context_id = uuid.UUID(str(row[0]))
        candidate = candidates.get(context_id)
        if candidate is None:
            candidate = ContextEvidenceCandidate(
                context_id=context_id,
                name=str(row[1]),
                document_path=str(row[2] or ""),
                description=str(row[3] or ""),
                score=0.0,
                reasons=["literal anchor match"],
            )
            candidates[context_id] = candidate
        candidate.score += float(row[4] or 1.0)
    return sorted(candidates.values(), key=lambda item: item.score, reverse=True)


async def _contexts_for_frame_precision(
    *,
    collection_id: uuid.UUID,
    frame_plan: GraphQueryFramePlan,
    document_ids: list[uuid.UUID] | None = None,
) -> list[ContextEvidenceCandidate]:
    if not frame_plan.focus_terms:
        return []

    def _clauses(
        *,
        params: dict[str, Any],
        prefix: str,
        terms: list[str],
        expressions: list[str],
    ) -> str:
        clauses: list[str] = []
        for index, term in enumerate(terms):
            key = f"{prefix}_{index}"
            params[key] = f"%{term}%"
            expression_clauses = [
                f"{expression} ILIKE :{key}" for expression in expressions
            ]
            clauses.append("(" + " OR ".join(expression_clauses) + ")")
        return " OR ".join(clauses) or "FALSE"

    async def _bounded_matches(
        *,
        terms: list[str],
        prefix: str,
        context_score: float,
        assertion_score: float,
        reason: str,
    ) -> dict[uuid.UUID, ContextEvidenceCandidate]:
        if not terms:
            return {}

        params: dict[str, Any] = {
            "cid": _uuid_for_sql(collection_id),
            "context_limit": 100,
            "assertion_limit": 160,
        }
        context_clause = _clauses(
            params=params,
            prefix=f"{prefix}_ctx",
            terms=terms,
            expressions=[
                "ctx.canonical_name",
                "ctx_ed.description",
                "ctx_ed.document_path",
            ],
        )
        assertion_clause = _clauses(
            params=params,
            prefix=f"{prefix}_assertion",
            terms=terms,
            expressions=["assertion.canonical_name", "assertion_ed.description"],
        )
        doc_filter = ""
        if document_ids:
            doc_placeholders: list[str] = []
            for index, document_id in enumerate(document_ids):
                key = f"{prefix}_document_id_{index}"
                params[key] = _uuid_for_sql(document_id)
                doc_placeholders.append(f":{key}")
            doc_filter = f" AND ctx_ed.document_id IN ({', '.join(doc_placeholders)})"

        async with AsyncSessionLocal() as session:
            direct_rows = await session.execute(
                text(
                    f"""
                    SELECT ctx.id, ctx.canonical_name, ctx_ed.document_path,
                           ctx_ed.description
                    FROM graph_entities ctx
                    JOIN entity_descriptions ctx_ed ON ctx_ed.entity_id = ctx.id
                    WHERE ctx.collection_id = :cid
                      AND ctx.primary_type = 'CONTEXT'
                      AND ({context_clause})
                      {doc_filter}
                    ORDER BY ctx_ed.document_path, ctx.canonical_name
                    LIMIT :context_limit
                    """
                ),
                params,
            )
            assertion_rows = await session.execute(
                text(
                    f"""
                    SELECT ctx.id, count(*) AS assertion_matches
                    FROM graph_relationships owns
                    JOIN graph_entities ctx ON ctx.id = owns.source_entity_id
                    JOIN entity_descriptions ctx_ed ON ctx_ed.entity_id = ctx.id
                    JOIN graph_entities assertion
                      ON assertion.id = owns.target_entity_id
                    JOIN entity_descriptions assertion_ed
                      ON assertion_ed.entity_id = assertion.id
                    WHERE owns.collection_id = :cid
                      AND owns.rel_type = 'HAS_ASSERTION'
                      AND ctx.primary_type = 'CONTEXT'
                      AND assertion.primary_type = 'ASSERTION'
                      AND ({assertion_clause})
                      {doc_filter}
                    GROUP BY ctx.id
                    ORDER BY assertion_matches DESC
                    LIMIT :assertion_limit
                    """
                ),
                params,
            )

        candidates: dict[uuid.UUID, ContextEvidenceCandidate] = {}
        for row in direct_rows:
            context_id = uuid.UUID(str(row[0]))
            candidates[context_id] = ContextEvidenceCandidate(
                context_id=context_id,
                name=str(row[1]),
                document_path=str(row[2] or ""),
                description=str(row[3] or ""),
                score=context_score,
                reasons=[f"{reason} context match"],
            )

        for row in assertion_rows:
            context_id = uuid.UUID(str(row[0]))
            candidate = candidates.get(context_id)
            if candidate is None:
                continue
            candidate.score += assertion_score * float(row[1] or 1.0)
            assertion_reason = f"{reason} owned assertion match"
            if assertion_reason not in candidate.reasons:
                candidate.reasons.append(assertion_reason)
        return candidates

    focus_candidates = await _bounded_matches(
        terms=frame_plan.focus_terms,
        prefix="focus",
        context_score=10.0,
        assertion_score=6.0,
        reason="focus frame",
    )
    competing_candidates = await _bounded_matches(
        terms=frame_plan.competing_terms,
        prefix="competing",
        context_score=7.0,
        assertion_score=4.0,
        reason="competing frame",
    )

    for context_id, competing in competing_candidates.items():
        candidate = focus_candidates.get(context_id)
        if candidate is None:
            continue
        candidate.score -= competing.score
        for reason in competing.reasons:
            demotion_reason = f"demoted by {reason}"
            if demotion_reason not in candidate.reasons:
                candidate.reasons.append(demotion_reason)

    return [
        candidate
        for candidate in sorted(
            focus_candidates.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        if candidate.score > 0
    ]


async def _select_context_evidence_candidates(
    *,
    collection: Collection,
    entity_query_embedding: list[float],
    relationship_query_embedding: list[float],
    document_ids: list[uuid.UUID] | None = None,
    plan: GraphQueryPlan | None = None,
    frame_plan: GraphQueryFramePlan | None = None,
    frame_candidates: list[ContextEvidenceCandidate] | None = None,
    top_k: int = 40,
    max_contexts: int = 8,
) -> list[ContextEvidenceCandidate]:
    entity_hits = await _graph_rag_vectors.search_entity_embeddings(
        collection_id=collection.id,
        query_embedding=entity_query_embedding,
        top_k=top_k,
        document_ids=document_ids,
    )
    relationship_hits = await _graph_rag_vectors.search_relationship_embeddings(
        collection_id=collection.id,
        query_embedding=relationship_query_embedding,
        top_k=top_k,
        document_ids=document_ids,
    )

    entity_scores: dict[uuid.UUID, float] = {}
    relationship_scores: dict[uuid.UUID, float] = {}
    document_path_scores: dict[str, float] = {}
    for hit in entity_hits:
        score = _vector_hit_score(hit)
        entity_id = _uuid_from_value(hit.metadata.get("entity_id"))
        if entity_id is not None:
            entity_scores[entity_id] = max(entity_scores.get(entity_id, 0.0), score)
        document_path = str(hit.metadata.get("document_path") or "")
        if document_path:
            document_path_scores[document_path] = max(
                document_path_scores.get(document_path, 0.0),
                score,
            )
    for hit in relationship_hits:
        score = _vector_hit_score(hit)
        relationship_id = _uuid_from_value(hit.metadata.get("relationship_id"))
        if relationship_id is not None:
            relationship_scores[relationship_id] = max(
                relationship_scores.get(relationship_id, 0.0),
                score,
            )
        document_path = str(hit.metadata.get("document_path") or "")
        if document_path:
            document_path_scores[document_path] = max(
                document_path_scores.get(document_path, 0.0),
                score,
            )

    graph_candidates = await _contexts_for_graph_hits(
        collection_id=collection.id,
        entity_scores=entity_scores,
        relationship_scores=relationship_scores,
        document_ids=document_ids,
    )
    document_candidates = await _contexts_for_document_paths(
        collection_id=collection.id,
        document_paths=document_path_scores,
        document_ids=document_ids,
    )
    merged = dict(graph_candidates)
    for context_id, doc_candidate in document_candidates.items():
        candidate = merged.get(context_id)
        if candidate is None:
            merged[context_id] = doc_candidate
            continue
        candidate.score += doc_candidate.score
        for reason in doc_candidate.reasons:
            if reason not in candidate.reasons:
                candidate.reasons.append(reason)

    frame_constrained = False
    if frame_plan is not None and frame_plan.focus_terms:
        if frame_candidates is None:
            frame_candidates = await _contexts_for_frame_precision(
                collection_id=collection.id,
                frame_plan=frame_plan,
                document_ids=document_ids,
            )
        if frame_candidates:
            frame_constrained = True
            precision_merged = {
                candidate.context_id: candidate for candidate in frame_candidates
            }
            for context_id, vector_candidate in merged.items():
                candidate = precision_merged.get(context_id)
                if candidate is None:
                    continue
                candidate.score += vector_candidate.score
                for reason in vector_candidate.reasons:
                    if reason not in candidate.reasons:
                        candidate.reasons.append(reason)
            merged = precision_merged

    if plan is not None and plan.scope == "anchored" and plan.anchors:
        anchor_candidates = await _contexts_for_anchor_literals(
            collection_id=collection.id,
            anchors=plan.anchors,
            document_ids=document_ids,
        )
        for anchor_candidate in anchor_candidates:
            candidate = merged.get(anchor_candidate.context_id)
            if candidate is None:
                if frame_constrained:
                    continue
                merged[anchor_candidate.context_id] = anchor_candidate
                continue
            candidate.score += anchor_candidate.score
            for reason in anchor_candidate.reasons:
                if reason not in candidate.reasons:
                    candidate.reasons.append(reason)
        if not frame_constrained:
            max_contexts = max(max_contexts, len(plan.anchors) * 3)

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[
        :max_contexts
    ]


def _merge_context_candidates(
    primary_contexts: list[ContextEvidenceCandidate],
    secondary_contexts: list[ContextEvidenceCandidate],
    *,
    max_contexts: int,
) -> list[ContextEvidenceCandidate]:
    merged: list[ContextEvidenceCandidate] = []
    seen: set[uuid.UUID] = set()
    for context in [*primary_contexts, *secondary_contexts]:
        if context.context_id in seen:
            continue
        merged.append(context)
        seen.add(context.context_id)
        if len(merged) >= max_contexts:
            break
    return merged


async def _load_context_assertions(
    *,
    collection_id: uuid.UUID,
    context_ids: list[uuid.UUID],
    max_assertions_per_context: int = 40,
) -> list[ContextAssertionEvidence]:
    if not context_ids:
        return []
    params: dict[str, Any] = {
        "cid": _uuid_for_sql(collection_id),
        "limit_per_context": max_assertions_per_context,
    }
    placeholders: list[str] = []
    for index, context_id in enumerate(context_ids):
        key = f"context_{index}"
        params[key] = _uuid_for_sql(context_id)
        placeholders.append(f":{key}")

    async with AsyncSessionLocal() as session:
        rows = await session.execute(
            text(
                f"""
                WITH owned_assertions AS (
                    SELECT
                        ctx.id AS context_id,
                        ctx.canonical_name AS context_name,
                        ctx_ed.document_path AS context_document_path,
                        assertion.id AS assertion_id,
                        assertion.canonical_name AS assertion_name,
                        assertion_ed.description AS evidence,
                        row_number() OVER (
                            PARTITION BY ctx.id
                            ORDER BY assertion.canonical_name
                        ) AS rn
                    FROM graph_relationships owns
                    JOIN graph_entities ctx ON ctx.id = owns.source_entity_id
                    JOIN entity_descriptions ctx_ed ON ctx_ed.entity_id = ctx.id
                    JOIN graph_entities assertion
                      ON assertion.id = owns.target_entity_id
                    JOIN entity_descriptions assertion_ed
                      ON assertion_ed.entity_id = assertion.id
                    WHERE owns.collection_id = :cid
                      AND owns.rel_type = 'HAS_ASSERTION'
                      AND ctx.id IN ({", ".join(placeholders)})
                )
                SELECT context_id, context_name, context_document_path,
                       assertion_id, assertion_name, evidence
                FROM owned_assertions
                WHERE rn <= :limit_per_context
                ORDER BY context_document_path, context_name, assertion_name
                """
            ),
            params,
        )
    return [
        ContextAssertionEvidence(
            context_id=uuid.UUID(str(row[0])),
            context_name=str(row[1]),
            document_path=str(row[2] or ""),
            assertion_id=uuid.UUID(str(row[3])),
            assertion=str(row[4]),
            evidence=str(row[5] or ""),
        )
        for row in rows
    ]


async def _collection_coverage_contexts(
    *,
    collection: Collection,
    document_ids: list[uuid.UUID] | None = None,
    max_documents: int = _COLLECTION_COVERAGE_MAX_DOCUMENTS,
    contexts_per_document: int = _COLLECTION_COVERAGE_CONTEXTS_PER_DOCUMENT,
) -> list[ContextEvidenceCandidate]:
    params: dict[str, Any] = {
        "cid": _uuid_for_sql(collection.id),
        "max_documents": max_documents,
        "contexts_per_document": contexts_per_document,
    }
    doc_filter = ""
    if document_ids:
        doc_placeholders: list[str] = []
        for index, document_id in enumerate(document_ids):
            key = f"coverage_document_id_{index}"
            params[key] = _uuid_for_sql(document_id)
            doc_placeholders.append(f":{key}")
        doc_filter = f" AND ed.document_id IN ({', '.join(doc_placeholders)})"

    async with AsyncSessionLocal() as session:
        rows = await session.execute(
            text(
                f"""
                WITH selected_documents AS (
                    SELECT ed.document_path
                    FROM graph_entities e
                    JOIN entity_descriptions ed ON ed.entity_id = e.id
                    WHERE e.collection_id = :cid
                      AND e.primary_type = 'CONTEXT'
                      AND ed.document_path IS NOT NULL
                      {doc_filter}
                    GROUP BY ed.document_path
                    ORDER BY ed.document_path
                    LIMIT :max_documents
                ),
                ranked_contexts AS (
                    SELECT
                        e.id,
                        e.canonical_name,
                        ed.document_path,
                        ed.description,
                        row_number() OVER (
                            PARTITION BY ed.document_path
                            ORDER BY e.canonical_name
                        ) AS rn
                    FROM selected_documents sd
                    JOIN entity_descriptions ed
                      ON ed.document_path = sd.document_path
                    JOIN graph_entities e ON e.id = ed.entity_id
                    WHERE e.collection_id = :cid
                      AND e.primary_type = 'CONTEXT'
                )
                SELECT id, canonical_name, document_path, description, rn
                FROM ranked_contexts
                WHERE rn <= :contexts_per_document
                ORDER BY document_path, rn, canonical_name
                """
            ),
            params,
        )

    contexts: list[ContextEvidenceCandidate] = []
    for row in rows:
        document_rank = float(row[4] or 1)
        contexts.append(
            ContextEvidenceCandidate(
                context_id=uuid.UUID(str(row[0])),
                name=str(row[1]),
                document_path=str(row[2] or ""),
                description=str(row[3] or ""),
                score=max(0.1, 1.0 / document_rank),
                reasons=["collection coverage"],
            )
        )
    return contexts


def _build_context_evidence_text(
    contexts: list[ContextEvidenceCandidate],
    assertions: list[ContextAssertionEvidence],
    *,
    coverage: bool = False,
    requested_fields: list[str] | None = None,
) -> tuple[str, list[str], list[str], str]:
    assertions_by_context: dict[uuid.UUID, list[ContextAssertionEvidence]] = {}
    for assertion in assertions:
        assertions_by_context.setdefault(assertion.context_id, []).append(assertion)

    if coverage:
        requested = ", ".join(requested_fields or []) or "(unspecified)"
        lines = [
            "Context:",
            "Collection Coverage Evidence:",
            "Each context below is a source-local evidence scope. Use coverage "
            "queries to preserve source boundaries and avoid omitting documents "
            "only because vector ranking would place them lower.",
            f"Requested fields: {requested}",
        ]
    else:
        lines = [
            "Context:",
            "Context-Scoped Evidence:",
            "Question-frame evidence should be treated as the primary evidence "
            "for the question's requested subject, fields, or scope. Selected "
            "retrieved evidence is additional context that may qualify, contrast, "
            "or fill gaps, but should not override more direct source-local "
            "evidence.",
        ]
    entities_used: list[str] = []
    relationships_used: list[str] = []
    rel_lines: list[str] = []

    if coverage:
        context_sections = [("Evidence", contexts)]
    else:
        frame_contexts = [
            context
            for context in contexts
            if any("focus frame" in reason for reason in context.reasons)
        ]
        frame_context_ids = {context.context_id for context in frame_contexts}
        retrieved_contexts = [
            context for context in contexts if context.context_id not in frame_context_ids
        ]
        context_sections = [
            ("Question-frame evidence", frame_contexts),
            ("Selected retrieved evidence", retrieved_contexts),
        ]

    context_index = 1
    for section_title, section_contexts in context_sections:
        if not section_contexts:
            continue
        if not coverage:
            lines.append(section_title + ":")
        for context in section_contexts:
            entities_used.append(context.name)
            lines.extend([
                f"Context {context_index}: {context.name}",
                f"Source: {context.document_path or '(unknown)'}",
                f"Routing score: {context.score:.4f}",
                f"Routing reasons: {', '.join(context.reasons) or '(none)'}",
                f"Context description: {context.description}",
                "Assertions:",
            ])
            owned_assertions = assertions_by_context.get(context.context_id, [])
            if not owned_assertions:
                lines.append("- (none)")
                context_index += 1
                continue
            for assertion in owned_assertions:
                evidence = " ".join(assertion.evidence.split())
                line = f"- {assertion.assertion}\n  Evidence: {evidence}"
                lines.append(line)
                rel_lines.append(line)
                relationships_used.append(assertion.assertion)
            context_index += 1

    rel_context = "\n".join(rel_lines)
    return (
        "\n".join(lines),
        list(dict.fromkeys(entities_used)),
        list(dict.fromkeys(relationships_used)),
        rel_context,
    )


async def _context_mix_artifacts(
    question: str,
    collection: Collection,
    namespace_id: uuid.UUID,
    entity_query_embedding: list[float],
    relationship_query_embedding: list[float],
    *,
    document_ids: list[uuid.UUID] | None = None,
    plan: GraphQueryPlan | None = None,
    llm_profile_id: uuid.UUID | None = None,
) -> GraphQueryArtifacts | None:
    if not await _collection_has_context_layer(collection):
        return None
    if plan is not None and plan.scope == "collection":
        contexts = await _collection_coverage_contexts(
            collection=collection,
            document_ids=document_ids,
        )
        max_assertions_per_context = _COLLECTION_COVERAGE_ASSERTIONS_PER_CONTEXT
        coverage = True
    else:
        frame_plan = await _plan_graph_query_frame(
            question,
            namespace_id,
            llm_profile_id,
            plan,
        )
        frame_contexts = (
            await _contexts_for_frame_precision(
                collection_id=collection.id,
                frame_plan=frame_plan,
                document_ids=document_ids,
            )
            if frame_plan.focus_terms
            else []
        )
        contexts = await _select_context_evidence_candidates(
            collection=collection,
            entity_query_embedding=entity_query_embedding,
            relationship_query_embedding=relationship_query_embedding,
            document_ids=document_ids,
            plan=plan,
            frame_plan=frame_plan,
            frame_candidates=frame_contexts,
            top_k=_CONTEXT_MIX_TOP_K,
            max_contexts=_CONTEXT_MIX_MAX_CONTEXTS,
        )
        contexts = _merge_context_candidates(
            frame_contexts,
            contexts,
            max_contexts=_CONTEXT_MIX_MAX_CONTEXTS,
        )
        max_assertions_per_context = 40
        coverage = False
    if not contexts:
        return None
    assertions = await _load_context_assertions(
        collection_id=collection.id,
        context_ids=[context.context_id for context in contexts],
        max_assertions_per_context=max_assertions_per_context,
    )
    context, entities_used, relationships_used, rel_context = (
        _build_context_evidence_text(
            contexts,
            assertions,
            coverage=coverage,
            requested_fields=plan.requested_fields if plan else None,
        )
    )
    discovered_entity_ids = {str(context.context_id) for context in contexts}
    discovered_entity_ids.update(
        str(assertion.assertion_id) for assertion in assertions
    )
    state = GraphQueryState(
        discovered_entity_ids=discovered_entity_ids,
        entity_relevance={
            str(context.context_id): context.score for context in contexts
        },
        traversed_rel_ids=[],
        rel_score_cache={},
        rel_combined_score_cache={},
    )
    route_profile = DerivedRouteProfile(
        primary_route="context",
        route_scores={"context": 1.0},
        rel_type_scores={},
    )
    logger.info(
        "graph_rag context_mix collection=%s question=%r scope=%s contexts=%s assertions=%d",
        collection.name,
        question,
        plan.scope if plan else "top_k",
        [(context.name, round(context.score, 6)) for context in contexts],
        len(assertions),
    )
    return GraphQueryArtifacts(
        context=context,
        entities_used=entities_used,
        relationships_used=relationships_used,
        rel_context=rel_context,
        route_profile=route_profile,
        state=state,
    )


def _fallback_mix_interpretation(
    candidates: list[tuple[str, str, float]],
) -> MixInterpretation:
    names = [name for name, _, _ in candidates[:8]]
    if not names:
        return MixInterpretation(selected_entities=[], retrieval_subqueries=[])
    subqueries: list[str] = []
    if len(names) >= 2:
        group_a = names[: max(1, len(names) // 2)]
        group_b = names[max(1, len(names) // 2) :]
        subqueries.append(
            "How do " + ", ".join(group_a) + " relate to each other?"
        )
        if group_b:
            subqueries.append(
                "How do " + ", ".join(group_b) + " relate to each other?"
            )
    if not subqueries:
        subqueries = [
            "How does " + names[0] + " relate to the question?",
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
        "You are rewriting a user's question using entity names from a knowledge "
        "graph so that vector retrieval can find relevant relationships.\n"
        "The original question may not mention any graph entities by name.\n"
        "Your job is to reformulate it using the candidate entity names so the "
        "graph's stored relationships are more likely to match.\n\n"
        "Select up to 8 relevant entities from the candidate list.\n"
        "Then produce 2 to 4 retrieval subqueries.\n"
        "Each subquery must use selected entity names and be focused on the "
        "topic of the original question. Use entity names exactly as they "
        "appear in the candidate list.\n"
        "Keep subqueries entity-focused and semantically complete — do not "
        "add philosophical prose or unnecessary explanations.\n\n"
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
    mention_index: _EntityMentionIndex | None = None,
) -> GraphQueryState:
    if mention_index is None:
        mention_index = await _get_mention_index(collection)
    query_tokens = _query_token_set(question)
    await _log_exact_name_hits(collection, question)
    candidates = await _top_entity_candidates(
        collection,
        entity_query_embedding,
        question=question,
        top_k=50,
        document_ids=document_ids,
        mention_index=mention_index,
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
        question=question,
        min_entity_score=_REL_ENDPOINT_ENTITY_SCORE_MIN,
        document_ids=document_ids,
        mention_index=mention_index,
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

    anchor_state = await _entity_anchor_state(
        collection,
        interpretation.selected_entities,
        query_tokens=query_tokens,
        document_ids=document_ids,
    )

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
            question=subquery,
            min_entity_score=_REL_ENDPOINT_ENTITY_SCORE_MIN,
            document_ids=document_ids,
            mention_index=mention_index,
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
        return _merge_states(rel_base_state, anchor_state)

    return _merge_states(rel_base_state, anchor_state, *states)


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


async def _meta_base_ref_names_from_artifacts(
    meta_artifacts: list[tuple[Collection, GraphQueryArtifacts]],
) -> list[str]:
    """Project selected meta concepts back to their base entity references."""
    names: list[str] = []
    seen: set[str] = set()

    async with AsyncSessionLocal() as session:
        for meta_collection, artifacts in meta_artifacts:
            discovered_ids = [
                uuid.UUID(entity_id)
                for entity_id in artifacts.state.discovered_entity_ids
                if entity_id
            ]
            traversed_rel_ids = [
                uuid.UUID(rel_id) for rel_id in artifacts.state.traversed_rel_ids if rel_id
            ]
            if not discovered_ids and not traversed_rel_ids:
                continue

            candidate_entity_ids: set[uuid.UUID] = set(discovered_ids)
            if traversed_rel_ids:
                rel_rows = await session.execute(
                    select(
                        GraphRelationship.source_entity_id,
                        GraphRelationship.target_entity_id,
                    ).where(
                        GraphRelationship.collection_id == meta_collection.id,
                        GraphRelationship.id.in_(traversed_rel_ids),
                    )
                )
                for source_id, target_id in rel_rows.all():
                    candidate_entity_ids.add(source_id)
                    candidate_entity_ids.add(target_id)

            if discovered_ids:
                evidence_rows = await session.execute(
                    select(
                        GraphRelationship.source_entity_id,
                        GraphRelationship.target_entity_id,
                    ).where(
                        GraphRelationship.collection_id == meta_collection.id,
                        GraphRelationship.rel_type == "EVIDENCED_BY",
                        or_(
                            GraphRelationship.source_entity_id.in_(discovered_ids),
                            GraphRelationship.target_entity_id.in_(discovered_ids),
                        ),
                    )
                )
                for source_id, target_id in evidence_rows.all():
                    candidate_entity_ids.add(source_id)
                    candidate_entity_ids.add(target_id)

            if not candidate_entity_ids:
                continue
            entity_rows = await session.execute(
                select(
                    GraphEntity.canonical_name,
                    GraphEntity.primary_type,
                ).where(
                    GraphEntity.collection_id == meta_collection.id,
                    GraphEntity.id.in_(list(candidate_entity_ids)),
                )
            )
            for canonical_name, primary_type in entity_rows.all():
                if primary_type != "base_entity_ref":
                    continue
                name = str(canonical_name or "").strip()
                key = name.casefold()
                if name and key not in seen:
                    seen.add(key)
                    names.append(name)
                if len(names) >= _META_PROJECTION_MAX_BASE_REFS:
                    return names

    return names


async def _base_entity_ids_for_names(
    collection: Collection,
    names: list[str],
) -> dict[str, float]:
    if not names:
        return {}
    lowered = {name.lower() for name in names if name.strip()}
    if not lowered:
        return {}

    entity_scores: dict[str, float] = {}
    async with AsyncSessionLocal() as session:
        entity_rows = await session.execute(
            select(GraphEntity.id, GraphEntity.canonical_name).where(
                GraphEntity.collection_id == collection.id,
                func.lower(GraphEntity.canonical_name).in_(lowered),
            )
        )
        for entity_id, canonical_name in entity_rows.all():
            score = _META_PROJECTION_ENTITY_SCORE
            if str(canonical_name or "").lower() in lowered:
                entity_scores[str(entity_id)] = score

        alias_rows = await session.execute(
            select(EntityAlias.entity_id, EntityAlias.alias_name)
            .join(GraphEntity, GraphEntity.id == EntityAlias.entity_id)
            .where(
                EntityAlias.collection_id == collection.id,
                GraphEntity.collection_id == collection.id,
                func.lower(EntityAlias.alias_name).in_(lowered),
            )
        )
        for entity_id, alias_name in alias_rows.all():
            if str(alias_name or "").lower() in lowered:
                entity_scores[str(entity_id)] = _META_PROJECTION_ENTITY_SCORE

    return entity_scores


async def _meta_projection_state(
    question: str,
    collection: Collection,
    meta_artifacts: list[tuple[Collection, GraphQueryArtifacts]],
    *,
    document_ids: list[uuid.UUID] | None = None,
) -> GraphQueryState:
    base_ref_names = await _meta_base_ref_names_from_artifacts(meta_artifacts)
    projected_entity_scores = await _base_entity_ids_for_names(
        collection,
        base_ref_names,
    )
    if not projected_entity_scores:
        return GraphQueryState(
            discovered_entity_ids=set(),
            entity_relevance={},
            traversed_rel_ids=[],
            rel_score_cache={},
            rel_combined_score_cache={},
        )

    projected_entity_ids = set(projected_entity_scores)
    query_tokens = _query_token_set(question)
    rel_score_cache: dict[str, float] = {}
    rel_combined_score_cache: dict[str, float] = {}
    traversed_rel_ids: list[str] = []

    async with AsyncSessionLocal() as session:
        rel_conditions = [
            GraphRelationship.collection_id == collection.id,
            or_(
                GraphRelationship.source_entity_id.in_(
                    [uuid.UUID(entity_id) for entity_id in projected_entity_ids]
                ),
                GraphRelationship.target_entity_id.in_(
                    [uuid.UUID(entity_id) for entity_id in projected_entity_ids]
                ),
            ),
        ]
        if document_ids:
            rel_conditions.append(
                GraphRelationship.id.in_(
                    select(RelationshipDescription.relationship_id).where(
                        RelationshipDescription.document_id.in_(document_ids)
                    )
                )
            )
        rel_rows = await session.execute(
            select(GraphRelationship)
            .where(*rel_conditions)
            .order_by(GraphRelationship.weight.desc())
            .limit(_META_PROJECTION_MAX_BASE_RELS)
        )
        for rel in rel_rows.scalars().all():
            rel_id = str(rel.id)
            edge_props = {
                "weight": rel.weight or 1,
                "keywords": rel.keywords or [],
                "rel_type": rel.rel_type,
            }
            combined = _combined_edge_score(
                _META_PROJECTION_EDGE_BASE_SCORE,
                edge_props,
                query_tokens,
            )
            rel_score_cache[rel_id] = _META_PROJECTION_EDGE_BASE_SCORE
            rel_combined_score_cache[rel_id] = combined
            traversed_rel_ids.append(rel_id)
            for endpoint_id in (str(rel.source_entity_id), str(rel.target_entity_id)):
                projected_entity_ids.add(endpoint_id)
                previous = projected_entity_scores.get(endpoint_id, 0.0)
                if combined > previous:
                    projected_entity_scores[endpoint_id] = combined

    traversed_rel_ids.sort(
        key=lambda rel_id: rel_combined_score_cache.get(rel_id, 0.0),
        reverse=True,
    )
    logger.info(
        "graph_rag mix_meta_projection collection=%s refs=%s projected_entities=%d projected_rels=%d",
        collection.name,
        base_ref_names[:20],
        len(projected_entity_ids),
        len(traversed_rel_ids),
    )
    return GraphQueryState(
        discovered_entity_ids=projected_entity_ids,
        entity_relevance=projected_entity_scores,
        traversed_rel_ids=traversed_rel_ids,
        rel_score_cache=rel_score_cache,
        rel_combined_score_cache=rel_combined_score_cache,
    )


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
                "\n\nIf a Context-Scoped Evidence section is present, each "
                "context is a source-local evidence scope and each assertion "
                "is true only inside that source context. Compare or group "
                "across contexts only after preserving which source each "
                "assertion came from."
                "\n\nThe context may contain both central evidence and nearby "
                "distractors. Prefer assertions and source descriptions that "
                "directly answer the question's subject, action, comparison, "
                "or requested field. Use routing scores and routing reasons "
                "only as relevance hints, not as facts. Use lower-scoring or "
                "merely related contexts only to qualify, contrast, or explain "
                "absence of support. If evidence appears to conflict, prefer "
                "the more direct source-local assertion over a broader or more "
                "generic related assertion. Do not infer that two items "
                "participate in the same process, argument, event, role, or "
                "mechanism merely because both are retrieved or mentioned in "
                "the question; state the connection only when the evidence "
                "shows it."
                "\n\nIf a Collection Coverage Evidence section is present, the "
                "question is collection-wide. Use every listed source scope "
                "that contains relevant requested fields; do not answer from "
                "only the most semantically similar contexts."
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
    query_plan: GraphQueryPlan | None = None,
) -> GraphQueryArtifacts:
    embedding_provider = await _resolve_embedding_provider(collection)
    entity_query_embedding = await _embed_entity_query(embedding_provider, question)
    mention_index = await _get_mention_index(collection)

    requested_mode = (mode or "mix").lower()
    effective_mode = _MODE_ALIASES.get(requested_mode, "mix")
    if effective_mode == "mix" and query_plan is None:
        query_plan = await _plan_graph_query(question, namespace_id, llm_profile_id)
    dimensions: list[str] = []
    logger.info(
        "graph_rag dimension_gating_disabled collection=%s mode=%s threshold=%.3f",
        collection.name,
        effective_mode,
        _REL_ENDPOINT_ENTITY_SCORE_MIN,
    )
    relationship_query_embedding = await _embed_relationship_query(
        embedding_provider,
        question,
        rel_type=None,
    )
    if effective_mode == "mix":
        context_artifacts = await _context_mix_artifacts(
            question,
            collection,
            namespace_id,
            entity_query_embedding,
            relationship_query_embedding,
            document_ids=document_ids,
            plan=query_plan,
            llm_profile_id=llm_profile_id,
        )
        if context_artifacts is not None:
            return context_artifacts

    async def _build_state_for(rel_type: str | None) -> GraphQueryState:
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
                mention_index=mention_index,
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
                mention_index=mention_index,
            )
        if effective_mode == "hybrid":
            entity_state = await _entity_first_state(
                question,
                collection,
                entity_query_embedding,
                relationship_query_embedding,
                **kwargs,
                document_ids=document_ids,
                mention_index=mention_index,
            )
            relationship_state = await _relationship_first_state(
                question,
                collection,
                entity_query_embedding,
                relationship_query_embedding,
                **kwargs,
                document_ids=document_ids,
                mention_index=mention_index,
            )
            return _merge_states(entity_state, relationship_state)
        return await _entity_first_state(
            question,
            collection,
            entity_query_embedding,
            relationship_query_embedding,
            **kwargs,
            document_ids=document_ids,
            mention_index=mention_index,
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
        state=state,
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
    effective_mode = _MODE_ALIASES.get((mode or "mix").lower(), "mix")
    query_plan = (
        await _plan_graph_query(question, namespace_id, llm_profile_id)
        if effective_mode == "mix"
        else None
    )
    logger.info(
        "graph_rag document_routing collection=%s route=%s document_ids=%s",
        collection.name,
        "all" if routing.use_all_documents else "documents",
        [str(document_id) for document_id in document_ids] if document_ids else [],
    )
    artifact_kwargs: dict[str, Any] = {"document_ids": document_ids}
    if "query_plan" in inspect.signature(_build_graph_query_artifacts).parameters:
        artifact_kwargs["query_plan"] = query_plan
    base = await _build_graph_query_artifacts(
        question,
        collection,
        namespace_id,
        mode,
        llm_profile_id,
        **artifact_kwargs,
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
                    **{
                        **artifact_kwargs,
                        "document_ids": None,
                    },
                ),
            )
        )

    if effective_mode == "mix" and meta_artifacts:
        projection_state = await _meta_projection_state(
            question,
            collection,
            meta_artifacts,
            document_ids=document_ids,
        )
        if projection_state.discovered_entity_ids or projection_state.traversed_rel_ids:
            projected_base_state = _merge_states(base.state, projection_state)
            (
                projected_context,
                projected_entities_used,
                projected_relationships_used,
                projected_rel_context,
            ) = await _build_context(
                projected_base_state,
                collection,
                document_ids=document_ids,
            )
            base = replace(
                base,
                context=projected_context,
                entities_used=projected_entities_used,
                relationships_used=projected_relationships_used,
                rel_context=projected_rel_context,
                state=projected_base_state,
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

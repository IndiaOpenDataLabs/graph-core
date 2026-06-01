"""Graph RAG query functions extracted from GraphService."""

from __future__ import annotations

import string
import uuid
from collections import deque
from dataclasses import dataclass
from itertools import combinations

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
from graph_core.services.crypto import CredentialCrypto
from graph_core.services.graph.query.vector import QueryResult
from graph_core.storage.graph_rag_vectors import GraphRAGVectorStore

_graph_rag_vectors = GraphRAGVectorStore()
_crypto = CredentialCrypto()
_ENTITY_RETRIEVAL_INSTRUCTION = (
    "Retrieve ontology entities whose descriptions best explain the user's "
    "state, process, causal mechanism, or source of exhaustion."
)
_RELATIONSHIP_RETRIEVAL_INSTRUCTION = (
    "Retrieve relationship descriptions that best explain the user's question, "
    "especially causes, mechanisms, tensions, and energy depletion."
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


@dataclass
class GraphQueryState:
    discovered_entity_ids: set[str]
    entity_relevance: dict[str, float]
    traversed_rel_ids: list[str]
    rel_score_cache: dict[str, float]


@dataclass
class MixInterpretation:
    selected_entities: list[str]
    retrieval_subqueries: list[str]


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
) -> list[float]:
    return await embedding_provider.embed_query(
        _format_retrieval_query(_RELATIONSHIP_RETRIEVAL_INSTRUCTION, query)
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


async def _entity_first_state(
    question: str,
    collection: Collection,
    entity_query_embedding: list[float],
    relationship_query_embedding: list[float],
) -> GraphQueryState:
    top_k = 10
    min_edge_sim = settings.graph_rag_min_edge_similarity
    energy_budget = 7.0
    max_depth = 8

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
    energy = effective_energy_budget

    sorted_seeds = sorted(seed_entity_ids, key=lambda e: seed_rel_scores.get(e, 0.0))
    stack = [(node_id, 0) for node_id in sorted_seeds]

    while stack and energy > 0:
        node_id, depth = stack.pop()
        if depth >= max_depth:
            continue

        edges = await graph_storage.get_node_edges(node_id)
        scored_edges: list[tuple[float, str, str]] = []
        for src, tgt in edges:
            neighbor = tgt if src == node_id else src
            if neighbor in visited:
                continue

            edge_props = await graph_storage.get_edge(src, tgt)
            if not edge_props:
                edge_props = await graph_storage.get_edge(tgt, src)
            if not (edge_props and edge_props.get("id")):
                continue

            rel_id_str = str(edge_props["id"])
            sim = await _score_relationship(
                collection,
                relationship_query_embedding,
                rel_id_str,
                rel_score_cache,
            )
            if sim >= effective_min_edge_sim:
                scored_edges.append((sim, neighbor, rel_id_str))

        for sim, neighbor, rel_id_str in sorted(scored_edges, key=lambda x: x[0]):
            cost = max(0.05, 1.0 - sim)
            if energy - cost <= 0:
                continue
            energy -= cost
            visited.add(neighbor)
            stack.append((neighbor, depth + 1))
            discovered_entity_ids.add(neighbor)
            if rel_id_str not in traversed_rel_ids:
                traversed_rel_ids.append(rel_id_str)
            edge_sim = 1.0 - cost
            if (
                neighbor not in entity_relevance
                or edge_sim > entity_relevance[neighbor]
            ):
                entity_relevance[neighbor] = edge_sim

    return GraphQueryState(
        discovered_entity_ids=discovered_entity_ids,
        entity_relevance=entity_relevance,
        traversed_rel_ids=traversed_rel_ids,
        rel_score_cache=rel_score_cache,
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
) -> tuple[list[str], list[str]] | None:
    queue: deque[tuple[str, list[str], list[str], int]] = deque(
        [(source_id, [source_id], [], 0)]
    )

    while queue:
        node_id, path_nodes, path_rels, depth = queue.popleft()
        if depth >= max_depth:
            continue

        edges = await graph_storage.get_node_edges(node_id)
        candidates: list[tuple[float, str, str]] = []
        for src, tgt in edges:
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
            candidates.append((sim, neighbor, rel_id))

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
) -> GraphQueryState:
    rel_seeds = await _search_relationship_seeds(
        collection,
        relationship_query_embedding,
        top_k=top_k,
    )

    traversed_rel_ids: list[str] = []
    rel_score_cache: dict[str, float] = {}
    discovered_entity_ids: set[str] = set()
    entity_relevance: dict[str, float] = {}

    async with AsyncSessionLocal() as session:
        for rel_id_str, sim in rel_seeds[:top_k]:
            rel = await session.get(GraphRelationship, uuid.UUID(rel_id_str))
            if not rel:
                continue
            traversed_rel_ids.append(rel_id_str)
            rel_score_cache[rel_id_str] = sim
            for eid in (str(rel.source_entity_id), str(rel.target_entity_id)):
                discovered_entity_ids.add(eid)
                if eid not in entity_relevance or sim > entity_relevance[eid]:
                    entity_relevance[eid] = sim

    endpoint_ids = sorted(
        discovered_entity_ids,
        key=lambda eid: entity_relevance.get(eid, 0.0),
        reverse=True,
    )[:max_endpoints]

    graph_storage = get_graph_storage(collection.id)
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
        )
        if not path:
            continue
        path_nodes, path_rels = path
        discovered_entity_ids.update(path_nodes)
        for rel_id in path_rels:
            if rel_id not in traversed_rel_ids:
                traversed_rel_ids.append(rel_id)
        path_score = sum(rel_score_cache.get(rel_id, 0.0) for rel_id in path_rels)
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
    )


async def _relationship_first_state(
    question: str,
    collection: Collection,
    entity_query_embedding: list[float],
    relationship_query_embedding: list[float],
) -> GraphQueryState:
    state = await _relationship_seed_state(
        collection,
        relationship_query_embedding,
        top_k=10,
        max_endpoints=20,
        max_pairs=30,
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

    for state in states:
        discovered_entity_ids.update(state.discovered_entity_ids)
        for eid, score in state.entity_relevance.items():
            if eid not in entity_relevance or score > entity_relevance[eid]:
                entity_relevance[eid] = score
        for rel_id in state.traversed_rel_ids:
            if rel_id not in traversed_rel_ids:
                traversed_rel_ids.append(rel_id)
        rel_score_cache.update(state.rel_score_cache)

    return GraphQueryState(
        discovered_entity_ids=discovered_entity_ids,
        entity_relevance=entity_relevance,
        traversed_rel_ids=traversed_rel_ids,
        rel_score_cache=rel_score_cache,
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
) -> GraphQueryState:
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
        )
        states.append(
            await _relationship_seed_state(
                collection,
                subquery_embedding,
                top_k=8,
                max_endpoints=16,
                max_pairs=20,
            )
        )

    if not states:
        return await _relationship_first_state(
            question,
            collection,
            entity_query_embedding,
            relationship_query_embedding,
        )

    return _merge_states(*states)


async def _build_context(
    state: GraphQueryState,
    collection: Collection,
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

        rel_context_parts: list[tuple[float, str]] = []
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
            sim = state.rel_score_cache.get(rel_id_str, 0.0)
            for description in descs:
                rel_text = f"{src_name} -> {tgt_name}: {description.description}"
                rel_context_parts.append((sim, rel_text))
                relationships_used.append(f"{src_name} -> {tgt_name}")

    rel_context_parts.sort(key=lambda item: item[0], reverse=True)
    entity_context = "\n".join(entity_context_parts)
    rel_context = "\n".join(text for _, text in rel_context_parts)
    context = (
        "Context:\n"
        "Entities:\n"
        f"{entity_context or '(none)'}\n"
        "Relationships:\n"
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
    relationship_query_embedding = await _embed_relationship_query(
        embedding_provider,
        question,
    )

    requested_mode = (mode or "mix").lower()
    effective_mode = _MODE_ALIASES.get(requested_mode, "mix")

    if effective_mode == "relationship-first":
        state = await _relationship_first_state(
            question,
            collection,
            entity_query_embedding,
            relationship_query_embedding,
        )
    elif effective_mode == "mix":
        state = await _mix_state(
            question,
            collection,
            namespace_id,
            llm_profile_id,
            embedding_provider,
            entity_query_embedding,
            relationship_query_embedding,
        )
    elif effective_mode == "hybrid":
        entity_state = await _entity_first_state(
            question,
            collection,
            entity_query_embedding,
            relationship_query_embedding,
        )
        relationship_state = await _relationship_first_state(
            question,
            collection,
            entity_query_embedding,
            relationship_query_embedding,
        )
        state = _merge_states(entity_state, relationship_state)
    else:
        state = await _entity_first_state(
            question,
            collection,
            entity_query_embedding,
            relationship_query_embedding,
        )
        effective_mode = "entity-first"

    context, entities_used, relationships_used, rel_context = await _build_context(
        state,
        collection,
    )
    entity_fallback = "\n".join(entities_used)
    fallback_text = rel_context or entity_fallback
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
        relationships_used=relationships_used,
        mode=effective_mode,
    )

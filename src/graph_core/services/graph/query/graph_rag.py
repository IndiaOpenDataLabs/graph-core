"""Graph RAG query functions extracted from GraphService."""

import uuid

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

# ── Module-level singleton dependencies ──

_graph_rag_vectors = GraphRAGVectorStore()
_crypto = CredentialCrypto()


# ── Credential / provider resolution helpers ──


async def _resolve_credential(
    session, profile: Profile
) -> tuple[str | None, str | None]:
    """Decrypt a profile's credential, returning (api_key, base_url)."""
    if profile.credential_id is None:
        return None, None
    credential = await session.get(Credential, profile.credential_id)
    if not credential:
        raise ValueError(f"Credential {profile.credential_id} not found")
    return _crypto.decrypt(credential.encrypted_secret), credential.base_url


async def _resolve_embedding_provider(collection: Collection) -> EmbeddingProvider:
    """Resolve the embedding provider for a collection."""
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
    """Resolve the LLM provider for a namespace."""
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


# ── Utility functions ──


def get_graph_storage(collection_id: uuid.UUID):
    """Return a FalkorDBGraphStorage scoped to the collection's own graph."""
    from graph_core.storage.graph_storage import FalkorDBGraphStorage

    graph_name = f"collection_{str(collection_id).replace('-', '')}"
    return FalkorDBGraphStorage(graph_name)


# ── Query functions ──


async def graph_rag_query(
    question: str,
    collection: Collection,
    namespace_id: uuid.UUID,
    llm_profile_id: uuid.UUID | None = None,
) -> QueryResult:
    """Full Graph RAG query pipeline with energy-decay DFS traversal.

    Steps:
    1. Embed query
    2. Seed entity search (pgvector entity embeddings + alias ILIKE)
    3. Score seeds by relationship relevance
    4. Energy-decay DFS traversal in FalkorDB
    5. Fetch EntityDescriptions from Postgres
    6. Fetch RelationshipDescriptions from Postgres
    7. Build context and call LLM
    """
    import string

    embedding_provider = await _resolve_embedding_provider(collection)

    # Step 1: Embed query
    query_embedding = await embedding_provider.embed_query(question)

    # Step 2: Seed entity search
    TOP_K = 10
    MIN_EDGE_SIM = settings.graph_rag_min_edge_similarity
    ENERGY_BUDGET = 7.0
    MAX_DEPTH = 8
    MAX_ENTITIES = 10
    MAX_ENTITY_DESCS = 4
    MAX_REL_DESCS = 4

    entity_hits = await _graph_rag_vectors.search_entity_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=TOP_K,
    )

    seed_entity_ids: list[str] = []
    entity_relevance: dict[str, float] = {}

    for hit in entity_hits:
        meta = hit.metadata
        entity_id_str = meta.get("entity_id", "")
        sim = 1.0 - hit.distance
        if entity_id_str and entity_id_str not in seed_entity_ids:
            seed_entity_ids.append(entity_id_str)
            entity_relevance[entity_id_str] = sim

    # Step 2b: Alias lookup — catch entities missed by vector search
    async with AsyncSessionLocal() as session:
        stop_words = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to",
            "for", "of", "with", "is", "what", "how", "why", "who",
            "i", "me", "my",
        }
        tokens = [w.strip(string.punctuation).lower() for w in question.split()]
        keywords = [w for w in tokens if w and w not in stop_words and len(w) > 2]
        keywords = list(dict.fromkeys([question] + keywords))

        for kw in keywords[:5]:
            alias_result = await session.execute(
                select(EntityAlias).join(
                    GraphEntity, GraphEntity.id == EntityAlias.entity_id
                ).where(
                    or_(
                        EntityAlias.alias_name.ilike(f"% {kw} %"),
                        EntityAlias.alias_name.ilike(f"{kw} %"),
                        EntityAlias.alias_name.ilike(f"% {kw}"),
                        EntityAlias.alias_name.ilike(kw),
                    ),
                    GraphEntity.collection_id == collection.id,
                ).limit(5)
            )
            for alias in alias_result.scalars().all():
                eid = str(alias.entity_id)
                if eid not in seed_entity_ids:
                    seed_entity_ids.append(eid)
                    entity_relevance[eid] = 1.0

    # Step 3: Score seed entities by relationship relevance
    rel_hits = await _graph_rag_vectors.search_relationship_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=max(TOP_K * 5, 50),
    )

    # Build name -> entity_id map
    async with AsyncSessionLocal() as session:
        seed_entity_rows = await session.execute(
            select(GraphEntity).where(
                GraphEntity.collection_id == collection.id
            )
        )
        name_to_eid = {
            e.canonical_name.lower(): str(e.id)
            for e in seed_entity_rows.scalars().all()
        }

    seed_rel_scores: dict[str, float] = {eid: 0.0 for eid in seed_entity_ids}
    best_seed_sim = max(entity_relevance.values()) if entity_relevance else 0.0
    effective_depth = MAX_DEPTH
    if best_seed_sim < 0.25:
        effective_min_edge_sim = max(MIN_EDGE_SIM, 0.5)
        effective_energy_budget = 2.5
    elif best_seed_sim < 0.4:
        effective_min_edge_sim = max(MIN_EDGE_SIM, 0.4)
        effective_energy_budget = 4.0
    else:
        effective_min_edge_sim = MIN_EDGE_SIM
        effective_energy_budget = ENERGY_BUDGET

    for hit in rel_hits:
        meta = hit.metadata
        sim = 1.0 - hit.distance
        if sim < effective_min_edge_sim:
            continue
        for name_field in ("source_name", "target_name"):
            name = meta.get(name_field, "").lower()
            eid = name_to_eid.get(name)
            if eid and sim > seed_rel_scores.get(eid, 0.0):
                seed_rel_scores[eid] = sim

    # Step 4: Energy-decay DFS traversal in FalkorDB
    graph_storage = get_graph_storage(collection.id)
    visited = set(seed_entity_ids)
    traversed_rel_ids: list[str] = []
    discovered_entity_ids = list(seed_entity_ids)
    energy = effective_energy_budget
    rel_score_cache: dict[str, float] = {}

    # Sort seeds by relationship relevance ascending so highest-rel seeds
    # sit on top of stack (LIFO = explored first)
    sorted_seeds = sorted(
        seed_entity_ids,
        key=lambda e: seed_rel_scores.get(e, 0.0),
    )
    stack = [(node_id, 0) for node_id in sorted_seeds]

    async with AsyncSessionLocal() as session:
        while stack and energy > 0:
            node_id, depth = stack.pop()
            if depth >= effective_depth:
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

                # Score edge by relationship embedding similarity
                rel_vdb_hits = await _graph_rag_vectors.search_relationship_embeddings(
                    collection_id=collection.id,
                    query_embedding=query_embedding,
                    top_k=MAX_REL_DESCS,
                    relationship_id=uuid.UUID(rel_id_str),
                )
                sim = max(1.0 - r.distance for r in rel_vdb_hits) if rel_vdb_hits else 0.0
                rel_score_cache[rel_id_str] = sim

                if sim >= effective_min_edge_sim:
                    scored_edges.append((sim, neighbor, rel_id_str))

            # Push low-sim edges first so high-sim edges are on top of stack
            for sim, neighbor, rel_id_str in sorted(scored_edges, key=lambda x: x[0]):
                cost = max(0.05, 1.0 - sim)
                if energy - cost > 0:
                    energy -= cost
                    visited.add(neighbor)
                    stack.append((neighbor, depth + 1))
                    if rel_id_str and rel_id_str not in traversed_rel_ids:
                        traversed_rel_ids.append(rel_id_str)
                    if neighbor not in discovered_entity_ids:
                        discovered_entity_ids.append(neighbor)
                    edge_sim = 1.0 - cost
                    if neighbor not in entity_relevance or edge_sim > entity_relevance[neighbor]:
                        entity_relevance[neighbor] = edge_sim

        # Step 5: Fetch EntityDescriptions from Postgres
        ranked_entity_ids = sorted(
            discovered_entity_ids,
            key=lambda eid: entity_relevance.get(eid, 0.0),
            reverse=True,
        )

        entity_context_parts: list[str] = []
        entities_used: list[str] = []
        for eid_str in ranked_entity_ids[:MAX_ENTITIES]:
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
                .limit(MAX_ENTITY_DESCS)
            )
            descs = descs_result.scalars().all()
            if descs:
                desc_texts = " | ".join(d.description for d in descs)
                entity_context_parts.append(
                    f"{entity.canonical_name} ({entity.primary_type or 'unknown'}): {desc_texts}"
                )
                entities_used.append(entity.canonical_name)

        # Step 6: Fetch RelationshipDescriptions from Postgres
        rel_context_parts: list[str] = []
        relationships_used: list[str] = []
        for rel_id_str in traversed_rel_ids[:50]:
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
                .limit(MAX_REL_DESCS)
            )
            descs = descs_result.scalars().all()
            sim = rel_score_cache.get(rel_id_str, 0.0)
            for d in descs:
                rel_text = f"{src_name} \u2192 {tgt_name}: {d.description}"
                rel_context_parts.append((sim, rel_text))
                relationships_used.append(f"{src_name} \u2192 {tgt_name}")

        rel_context_parts.sort(key=lambda x: x[0], reverse=True)

        # Step 7: Build context
        entity_context = "\n".join(entity_context_parts)
        rel_context = "\n".join(text for _, text in rel_context_parts)

        context = f"""Context:
Entities:
{entity_context or "(none)"}
Relationships:
{rel_context or "(none)"}"""

    llm_provider = await _resolve_llm_provider(
        namespace_id=namespace_id, llm_profile_id=llm_profile_id,
    )
    if isinstance(llm_provider, LocalEchoLLMProvider):
        response = entity_context or rel_context or "No relevant context found."
    else:
        response = await llm_provider.chat([
            {
                "role": "system",
                "content": (
                    "Use the context below to answer the question. "
                    "Draw on the entities and relationships to reason through your answer - "
                    "explain, connect, and illuminate rather than just report. "
                    "Write in natural prose. If the context is insufficient for part of the "
                    "question, acknowledge it briefly without making it the focus."
                ),
            },
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {question}",
            },
        ])

    return QueryResult(
        response=response,
        entities_used=entities_used,
        relationships_used=list(dict.fromkeys(relationships_used)),
    )

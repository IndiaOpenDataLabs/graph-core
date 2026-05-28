"""LightRAG query functions extracted from GraphService."""

import uuid
from typing import Any

from sqlalchemy import text

from graph_core.config import settings
from graph_core.database import AsyncSessionLocal
from graph_core.embedding import get_embedding_provider
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.llm import LocalEchoLLMProvider, get_llm_provider
from graph_core.llm.interface import LLMProvider
from graph_core.models.collection import Collection
from graph_core.models.credential import Credential
from graph_core.models.profile import Profile
from graph_core.services.crypto import CredentialCrypto
from graph_core.services.graph.query.vector import QueryResult
from graph_core.storage.graph_rag_vectors import GraphRAGVectorStore
from graph_core.storage.vector_tables import table_name

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
        )


# ── Utility functions ──


def get_graph_storage(collection_id: uuid.UUID):
    """Return a FalkorDBGraphStorage scoped to the collection's own graph."""
    from graph_core.storage.graph_storage import FalkorDBGraphStorage

    graph_name = f"collection_{str(collection_id).replace('-', '')}"
    return FalkorDBGraphStorage(graph_name)


# ── Query functions ──


async def lightrag_query(
    question: str,
    collection: Collection,
    namespace_id: uuid.UUID,
    mode: str,
    llm_profile_id: uuid.UUID | None = None,
) -> QueryResult:
    """LightRAG query with keyword-driven retrieval.

    Supports modes: local, global, hybrid, naive.
    1. Extract high/low level keywords from query
    2. Mode-specific retrieval (entity/relationship/chunk vector search)
    3. Graph traversal for connected entities/relationships
    4. Token budget management
    5. LLM answer generation
    """
    embedding_provider = await _resolve_embedding_provider(collection)
    llm_provider = await _resolve_llm_provider(
        namespace_id=namespace_id, llm_profile_id=llm_profile_id,
    )

    keywords = await extract_keywords(question, llm_provider)

    if mode == "naive":
        return await _lightrag_query_naive(
            question, collection, embedding_provider, llm_provider,
        )
    elif mode == "local":
        return await _lightrag_query_local(
            question, collection, keywords, embedding_provider, llm_provider,
        )
    elif mode == "global":
        return await _lightrag_query_global(
            question, collection, keywords, embedding_provider, llm_provider,
        )
    elif mode == "hybrid":
        return await _lightrag_query_hybrid(
            question, collection, keywords, embedding_provider, llm_provider,
        )
    elif mode == "mix":
        return await _lightrag_query_mix(
            question, collection, keywords, embedding_provider, llm_provider,
        )
    else:
        return await _lightrag_query_local(
            question, collection, keywords, embedding_provider, llm_provider,
        )


async def extract_keywords(
    query: str, llm_provider: LLMProvider
) -> tuple[list[str], list[str]]:
    """Extract high-level and low-level keywords from query.

    Returns (high_level, low_level) keyword lists.
    Falls back to word-level extraction on failure.
    """
    if not query or not query.strip():
        return [], []

    _KW_SCHEMA = {
        "type": "object",
        "properties": {
            "high_level_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Conceptual, abstract keywords for broad topic search",
            },
            "low_level_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific, concrete keywords for precise entity search",
            },
        },
        "required": ["high_level_keywords", "low_level_keywords"],
    }

    try:
        result = await llm_provider.structured_extract(
            prompt=(
                "Extract keywords from this query for knowledge graph search.\n\n"
                "Return JSON with two arrays:\n"
                "- high_level_keywords: conceptual terms describing the topic/theme\n"
                "- low_level_keywords: specific entity names, places, or concrete terms\n\n"
                f"Query: {query}"
            ),
            schema=_KW_SCHEMA,
        )
        hl = result.get("high_level_keywords", [])
        ll = result.get("low_level_keywords", [])
        if isinstance(hl, list) and isinstance(ll, list):
            hl = [str(k) for k in hl if k]
            ll = [str(k) for k in ll if k]
            if hl and ll:
                return hl, ll
    except Exception:
        pass

    return fallback_keywords(query), fallback_keywords(query)


def fallback_keywords(query: str) -> list[str]:
    """Fallback keyword extraction using simple tokenization."""
    import string

    stop_words = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to",
        "for", "of", "with", "is", "what", "how", "why", "who",
        "i", "me", "my", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "this", "that", "these", "those", "there", "here",
    }
    words = query.lower().split()
    words = [w.strip(string.punctuation) for w in words]
    words = [w for w in words if w and w not in stop_words and len(w) > 2]
    return words if words else [w for w in words if w]


async def _lightrag_query_naive(
    question: str,
    collection: Collection,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
) -> QueryResult:
    """Naive mode: pure vector search on chunk embeddings."""
    query_embedding = await embedding_provider.embed_query(question)
    hits = await _graph_rag_vectors.search_chunk_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=settings.vector_query_top_k,
    )
    chunks = [h.content for h in hits]
    response = await _generate_vector_answer(
        question=question,
        chunks=chunks,
        namespace_id=collection.namespace_id,
        llm_profile_id=None,
    )
    if isinstance(llm_provider, LocalEchoLLMProvider):
        response = chunks[0] if chunks else ""
    return QueryResult(
        response=response, entities_used=[], relationships_used=[], mode="naive",
    )


async def _lightrag_query_local(
    question: str,
    collection: Collection,
    keywords: tuple[list[str], list[str]],
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
) -> QueryResult:
    """Local mode: entity-focused retrieval + graph traversal.

    1. Search entity embeddings with low-level keywords
    2. For each entity, get connected edges from graph
    3. Collect source chunks from entities and relationships
    4. Build context with token budgets
    5. Call LLM
    """
    high_level, low_level = keywords
    search_terms = low_level if low_level else [question]
    search_text = " ".join(search_terms)
    query_embedding = await embedding_provider.embed_query(search_text)

    entity_hits = await _graph_rag_vectors.search_entity_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=20,
    )

    entities: list[dict[str, Any]] = []
    entity_names: list[str] = []
    graph_storage = get_graph_storage(collection.id)
    collection_id_str = str(collection.id)

    for hit in entity_hits:
        name = hit.metadata.get("name", "")
        if name and name not in entity_names:
            entity_names.append(name)
            node = await graph_storage.get_lightrag_node(name, collection_id_str)
            if node:
                entities.append(node)

    relationships: list[dict[str, Any]] = []
    rel_ids_seen: set[str] = set()
    for entity in entities:
        name = entity.get("name", "")
        if not name:
            continue
        try:
            edges = await graph_storage.get_lightrag_node_edges(name, collection_id_str)
            for src, tgt in edges:
                edge_data = await graph_storage.get_lightrag_edge(src, tgt, collection_id_str)
                if edge_data:
                    edge_id = edge_data.get("id", f"{src}__{tgt}")
                    if edge_id not in rel_ids_seen:
                        rel_ids_seen.add(edge_id)
                        relationships.append(edge_data)
        except Exception:
            continue

    chunk_ids = set()
    for entity in entities:
        chunk_ids.update(entity.get("source_ids") or [])
    for rel in relationships:
        chunk_ids.update(rel.get("source_ids") or [])

    chunks = await _get_chunks_by_hashes(collection.id, list(chunk_ids))

    entity_context, entities_used = _build_budgeted_context(
        [f"{e.get('name', '?')} ({e.get('type', '?')}): {e.get('description', '')}" for e in entities],
        [e.get("name", "") for e in entities],
        6000,
    )

    rel_context, rels_used = _build_budgeted_context(
        [f"{r.get('id', '?')}: {r.get('description', '')}" for r in relationships],
        [r.get("id", "") for r in relationships],
        8000,
    )

    chunk_context, _ = _build_budgeted_context(
        [c["content"] for c in chunks],
        [c["id"] for c in chunks],
        30000,
    )

    context = f"""Context:
Entities:
{entity_context or "(none)"}
Relationships:
{rel_context or "(none)"}
Source Text:
{chunk_context or "(none)"}"""

    response = await _generate_lightrag_response(context, question, llm_provider)

    return QueryResult(
        response=response,
        entities_used=entities_used,
        relationships_used=rels_used,
        mode="local",
    )


async def _lightrag_query_global(
    question: str,
    collection: Collection,
    keywords: tuple[list[str], list[str]],
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
) -> QueryResult:
    """Global mode: relationship-focused retrieval.

    1. Search relationship embeddings with high-level keywords
    2. For each relationship, get connected entities
    3. Collect source chunks
    4. Build context with token budgets
    5. Call LLM
    """
    high_level, low_level = keywords
    search_terms = high_level if high_level else [question]
    search_text = " ".join(search_terms)
    query_embedding = await embedding_provider.embed_query(search_text)

    rel_hits = await _graph_rag_vectors.search_relationship_embeddings(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=30,
    )

    relationships: list[dict[str, Any]] = []
    rel_ids: list[str] = []
    graph_storage = get_graph_storage(collection.id)
    collection_id_str = str(collection.id)

    for hit in rel_hits:
        meta = hit.metadata
        src_name = meta.get("source_name", "")
        tgt_name = meta.get("target_name", "")
        if src_name and tgt_name:
            rel_id = f"{src_name}__{tgt_name}"
            if rel_id not in rel_ids:
                rel_ids.append(rel_id)
                edge = await graph_storage.get_lightrag_edge(src_name, tgt_name, collection_id_str)
                if edge:
                    relationships.append(edge)

    entity_ids_set: set[str] = set()
    entities: list[dict[str, Any]] = []
    for rel in relationships:
        for entity_name in (rel.get("source_name"), rel.get("target_name")):
            if entity_name and entity_name not in entity_ids_set:
                entity_ids_set.add(entity_name)
                node = await graph_storage.get_lightrag_node(entity_name, collection_id_str)
                if node:
                    entities.append(node)

    chunk_ids = set()
    for rel in relationships:
        chunk_ids.update(rel.get("source_ids") or [])
    for entity in entities:
        chunk_ids.update(entity.get("source_ids") or [])

    chunks = await _get_chunks_by_hashes(collection.id, list(chunk_ids))

    rel_context, rels_used = _build_budgeted_context(
        [f"{r.get('id', '?')}: {r.get('description', '')}" for r in relationships],
        [r.get("id", "") for r in relationships],
        8000,
    )

    entity_context, entities_used = _build_budgeted_context(
        [f"{e.get('name', '?')} ({e.get('type', '?')}): {e.get('description', '')}" for e in entities],
        [e.get("name", "") for e in entities],
        6000,
    )

    chunk_context, _ = _build_budgeted_context(
        [c["content"] for c in chunks],
        [c["id"] for c in chunks],
        30000,
    )

    context = f"""Context:
Relationships:
{rel_context or "(none)"}
Entities:
{entity_context or "(none)"}
Source Text:
{chunk_context or "(none)"}"""

    response = await _generate_lightrag_response(context, question, llm_provider)

    return QueryResult(
        response=response,
        entities_used=entities_used,
        relationships_used=rels_used,
        mode="global",
    )


async def _lightrag_query_hybrid(
    question: str,
    collection: Collection,
    keywords: tuple[list[str], list[str]],
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
) -> QueryResult:
    """Hybrid mode: merge local + global retrieval."""
    local_result = await _lightrag_query_local(
        question, collection, keywords, embedding_provider, llm_provider,
    )
    global_result = await _lightrag_query_global(
        question, collection, keywords, embedding_provider, llm_provider,
    )

    merged_entities = _merge_unique(local_result.entities_used, global_result.entities_used)
    merged_rels = _merge_unique(local_result.relationships_used, global_result.relationships_used)
    merged_response = local_result.response

    if global_result.response and global_result.response != local_result.response:
        if isinstance(llm_provider, LocalEchoLLMProvider):
            merged_response = local_result.response
        else:
            merged_response = local_result.response

    return QueryResult(
        response=merged_response,
        entities_used=merged_entities,
        relationships_used=merged_rels,
        mode="hybrid",
    )


async def _lightrag_query_mix(
    question: str,
    collection: Collection,
    keywords: tuple[list[str], list[str]],
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
) -> QueryResult:
    """Mix mode: combine local + global + naive retrieval.

    Runs all three strategies, deduplicates entities/relationships,
    and merges chunks with token budgets.
    """
    local_result = await _lightrag_query_local(
        question, collection, keywords, embedding_provider, llm_provider,
    )
    global_result = await _lightrag_query_global(
        question, collection, keywords, embedding_provider, llm_provider,
    )
    naive_result = await _lightrag_query_naive(
        question, collection, embedding_provider, llm_provider,
    )

    merged_entities = _merge_unique(
        local_result.entities_used, global_result.entities_used,
    )
    merged_rels = _merge_unique(
        local_result.relationships_used, global_result.relationships_used,
    )

    response = local_result.response
    if isinstance(llm_provider, LocalEchoLLMProvider):
        response = local_result.response

    return QueryResult(
        response=response,
        entities_used=merged_entities,
        relationships_used=merged_rels,
        mode="mix",
    )


def _merge_unique(first: list[str], second: list[str]) -> list[str]:
    return list(dict.fromkeys(first + second))


def _build_budgeted_context(
    texts: list[str],
    ids: list[str],
    max_tokens: int,
) -> tuple[str, list[str]]:
    """Build context string respecting approximate token budget."""
    if not texts:
        return "", []

    used_ids = []
    used_parts = []
    total_tokens = 0

    for t, item_id in zip(texts, ids):
        tokens = max(1, len(t.split()))
        if total_tokens + tokens <= max_tokens:
            used_parts.append(t)
            used_ids.append(item_id)
            total_tokens += tokens
        else:
            remaining = max_tokens - total_tokens
            if remaining > 10:
                max_chars = remaining * 4
                truncated = t[:max_chars] + "..."
                used_parts.append(truncated)
            break

    return "\n\n".join(used_parts), used_ids


async def _get_chunks_by_hashes(
    collection_id: uuid.UUID, chunk_hashes: list[str]
) -> list[dict]:
    """Retrieve chunk contents from per-collection chunk_embeddings by hashes."""
    if not chunk_hashes:
        return []

    tbl = table_name(collection_id, "chunk_embeddings")
    async with AsyncSessionLocal() as session:
        placeholders = ",".join(f"'{h}'" for h in chunk_hashes)
        result = await session.execute(
            text(
                f"SELECT id::text, content FROM {tbl} "
                f"WHERE chunk_hash IN ({placeholders}) "
                f"ORDER BY chunk_index"
            )
        )
        return [{"id": row[0], "content": row[1]} for row in result]


async def _generate_lightrag_response(
    context: str, question: str, llm_provider: LLMProvider
) -> str:
    if isinstance(llm_provider, LocalEchoLLMProvider):
        return context
    return await llm_provider.chat([
        {
            "role": "system",
            "content": (
                "Use the provided context to answer the question. "
                "Draw on the entities, relationships, and source text to reason through your answer. "
                "Write in natural prose. If the context is insufficient, acknowledge it briefly."
            ),
        },
        {
            "role": "user",
            "content": f"{context}\n\nQuestion: {question}",
        },
    ])


async def _generate_vector_answer(
    *,
    question: str,
    chunks: list[str],
    namespace_id: uuid.UUID,
    llm_profile_id: uuid.UUID | None,
) -> str:
    """Generate answer from vector search chunks (used by naive mode)."""
    if not chunks:
        return ""
    llm_provider = await _resolve_llm_provider(
        namespace_id=namespace_id, llm_profile_id=llm_profile_id,
    )
    if isinstance(llm_provider, LocalEchoLLMProvider):
        return chunks[0]
    context = "\n\n".join(f"Chunk {i + 1}:\n{c}" for i, c in enumerate(chunks))
    return await llm_provider.chat([
        {"role": "system", "content": "Answer using only the provided context."},
        {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"},
    ])

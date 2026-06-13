"""Vector query functions extracted from GraphService."""

import uuid
from dataclasses import dataclass

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
from graph_core.storage.vector_store import VectorStore

# ── QueryResult dataclass ──


@dataclass
class QueryResult:
    response: str
    entities_used: list[str]
    relationships_used: list[str]
    mode: str | None = None
    chat_id: str | None = None


# ── Module-level singleton dependencies ──

_vector_store = VectorStore()
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


# ── Query functions ──


async def vector_query(
    question: str,
    collection: Collection,
    namespace_id: uuid.UUID,
    mode: str,
    llm_profile_id: uuid.UUID | None = None,
) -> QueryResult:
    embedding_provider = await _resolve_embedding_provider(collection)
    query_embedding = await embedding_provider.embed_query(question)
    results = await _vector_store.query_chunks(
        collection_id=collection.id,
        query_embedding=query_embedding,
        top_k=settings.vector_query_top_k,
    )
    chunks = [r["content"] for r in results]
    response = await generate_vector_answer(
        question=question,
        chunks=chunks,
        namespace_id=namespace_id,
        llm_profile_id=llm_profile_id,
    )
    return QueryResult(
        response=response, entities_used=[], relationships_used=[], mode=mode,
    )


async def generate_vector_answer(
    *,
    question: str,
    chunks: list[str],
    namespace_id: uuid.UUID,
    llm_profile_id: uuid.UUID | None,
) -> str:
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

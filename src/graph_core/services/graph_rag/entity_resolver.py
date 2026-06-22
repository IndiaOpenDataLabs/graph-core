"""Incremental entity resolver — zero-LLM entity resolution.

Three-tier pipeline:
1. Exact alias lookup in EntityAlias table
2. Embedding similarity against entity centroids
3. Fuzzy name matching
"""

from __future__ import annotations

import hashlib
import difflib
import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from graph_core.config import settings
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.models.domain_config import get_domain_config
from graph_core.models.graph_rag import (
    EntityAlias,
    EntityDescription,
    EntityType,
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
    RelationshipDescription,
    RelationshipTypeAlias,
)
from graph_core.models.rel_types import normalize_rel_type, relationship_embedding_text
from graph_core.storage.graph_names import (
    collection_graph_name,
    legacy_collection_graph_name,
)
from graph_core.storage.graph_rag_vectors import GraphRAGVectorStore
from graph_core.storage.graph_storage import FalkorDBGraphStorage

logger = logging.getLogger(__name__)


@dataclass
class EntityResolutionResult:
    is_new: bool
    entity_id: uuid.UUID
    canonical_name: str


@dataclass
class RelationshipResolutionResult:
    is_new: bool
    relationship_id: uuid.UUID


@dataclass
class RelationshipTypeResolutionResult:
    relationship_type_id: uuid.UUID
    canonical_type: str


class IncrementalEntityResolver:
    """Resolves extracted entities/relationships against existing DB records."""

    HIGH_CONFIDENCE_SIMILARITY = 0.8
    MEDIUM_CONFIDENCE_SIMILARITY = 0.65
    FUZZY_NAME_THRESHOLD = 0.8
    DESCRIPTION_SIMILARITY_THRESHOLD = 0.90

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        collection_id: uuid.UUID,
        domain: str | None = None,
        namespace_id: uuid.UUID | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._embedding = embedding_provider
        self._collection_id = collection_id
        self._domain = (domain or "").strip().lower() or None
        self._domain_cfg = get_domain_config(self._domain)
        self._vstore = GraphRAGVectorStore()
        graph_name = (
            collection_graph_name(
                namespace_id=namespace_id,
                collection_id=collection_id,
                collection_name=collection_name,
            )
            if collection_name is not None
            else legacy_collection_graph_name(collection_id)
        )
        self._graph_storage = FalkorDBGraphStorage(
            graph_name,
            namespace_id=namespace_id,
        )

    async def _resolve_rel_type(
        self,
        session: AsyncSession,
        rel_type: str,
    ) -> RelationshipTypeResolutionResult:
        """Resolve a rel_type to its canonical form using cluster-based matching.

        Three-tier resolution:
        1. Exact alias lookup in relationship_type_aliases (backfilled from clustering)
        2. Prefix embedding similarity against all known rel_types — map to the
           canonical of the matched cluster
        3. Accept the rel_type as-is (truly novel)
        """
        # Tier 1: Static alias table (backfilled from clustering)
        normalized_rel_type = normalize_rel_type(rel_type)

        relationship_type = await self._find_relationship_type_by_label(
            session,
            normalized_rel_type,
        )
        if relationship_type:
            relationship_type = await self._record_relationship_type_observation(
                session,
                relationship_type,
                normalized_rel_type,
            )
            logger.debug(
                "rel_type alias: %s -> %s",
                normalized_rel_type,
                relationship_type.canonical_type,
            )
            return RelationshipTypeResolutionResult(
                relationship_type_id=relationship_type.id,
                canonical_type=relationship_type.canonical_type,
            )

        # Tier 2: Prefix embedding similarity — find cluster member, map to canonical
        await self._vstore.ensure_prefix_embeddings_table(self._collection_id)

        stored = await self._vstore.load_all_prefix_embeddings(self._collection_id)
        if stored:
            query_emb = await self._embedding.embed_query(normalized_rel_type)

            best_match: str | None = None
            best_dist: float = 1e9

            for rt, emb in stored.items():
                if rt == normalized_rel_type:
                    continue
                dot = sum(a * b for a, b in zip(query_emb, emb))
                na = sum(a * a for a in query_emb) ** 0.5
                nb = sum(b * b for b in emb) ** 0.5
                if na > 0 and nb > 0:
                    dist = 1.0 - dot / (na * nb)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = rt

            if best_match and (1.0 - best_dist) >= self.HIGH_CONFIDENCE_SIMILARITY:
                # best_match is a cluster member — look up its canonical
                relationship_type = await self._find_relationship_type_by_label(
                    session,
                    best_match,
                )
                if relationship_type is None:
                    relationship_type = await self._resolve_or_create_relationship_type(
                        session,
                        best_match,
                    )

                # Store prefix embedding for future matching
                await self._vstore.upsert_prefix_embedding(
                    collection_id=self._collection_id,
                    rel_type=normalized_rel_type,
                    embedding=query_emb,
                )

                # Insert alias mapping
                await self._add_relationship_type_alias(
                    session,
                    relationship_type.id,
                    relationship_type.canonical_type,
                    normalized_rel_type,
                )
                relationship_type = await self._record_relationship_type_observation(
                    session,
                    relationship_type,
                    normalized_rel_type,
                )
                logger.debug(
                    "rel_type cluster match: %s -> %s (sim=%.4f, via %s)",
                    normalized_rel_type,
                    relationship_type.canonical_type,
                    1.0 - best_dist,
                    best_match,
                )
                return RelationshipTypeResolutionResult(
                    relationship_type_id=relationship_type.id,
                    canonical_type=relationship_type.canonical_type,
                )

        # Novel rel_type — store prefix embedding for future matching
        await self._vstore.ensure_prefix_embeddings_table(self._collection_id)
        emb = await self._embedding.embed_query(normalized_rel_type)
        await self._vstore.upsert_prefix_embedding(
            collection_id=self._collection_id,
            rel_type=normalized_rel_type,
            embedding=emb,
        )
        relationship_type = await self._resolve_or_create_relationship_type(
            session,
            normalized_rel_type,
        )
        await self._add_relationship_type_alias(
            session,
            relationship_type.id,
            relationship_type.canonical_type,
            normalized_rel_type,
        )
        relationship_type = await self._record_relationship_type_observation(
            session,
            relationship_type,
            normalized_rel_type,
        )
        return RelationshipTypeResolutionResult(
            relationship_type_id=relationship_type.id,
            canonical_type=relationship_type.canonical_type,
        )

    async def resolve_entity(
        self,
        session: AsyncSession,
        name: str,
        entity_type: str,
        description: str,
        source_chunk_hash: str,
        document_id: uuid.UUID | None = None,
        document_path: str | None = None,
    ) -> EntityResolutionResult:
        normalized_name = self._normalize_entity_name(name)
        exact_resolution = entity_type == "base_entity_ref" or self._requires_exact_name_resolution(
            normalized_name,
            entity_type,
        )

        if exact_resolution:
            existing_result = await session.execute(
                select(GraphEntity).where(
                    GraphEntity.canonical_name == normalized_name,
                    GraphEntity.collection_id == self._collection_id,
                )
            )
            existing = existing_result.scalar_one_or_none()
            if existing:
                logger.debug("Exact canonical match: %s", normalized_name)
                search_text = f"{normalized_name}: {description}"
                context_embedding = await self._embedding.embed_query(search_text)
                await self._add_description_and_update_centroid(
                    session,
                    existing.id,
                    existing,
                    description,
                    source_chunk_hash,
                    context_embedding,
                    document_id=document_id,
                    document_path=document_path,
                )
                await self._add_or_increment_type(session, existing.id, entity_type)
                return EntityResolutionResult(
                    is_new=False, entity_id=existing.id, canonical_name=existing.canonical_name
                )

            new_id = uuid.uuid4()
            stmt = (
                pg_insert(GraphEntity)
                .values(
                    id=new_id,
                    canonical_name=normalized_name,
                    primary_type=entity_type,
                    description_count=0,
                    collection_id=self._collection_id,
                )
                .on_conflict_do_nothing(
                    constraint="uq_graph_entities_canonical_name_collection_id"
                )
                .returning(GraphEntity.id)
            )
            result = await session.execute(stmt)
            row = result.fetchone()
            if row:
                entity_id = row[0]
                entity = await session.get(GraphEntity, entity_id)
                search_text = f"{normalized_name}: {description}"
                context_embedding = await self._embedding.embed_query(search_text)
                await self._add_alias(
                    session,
                    entity_id,
                    normalized_name,
                    source_chunk_hash,
                    document_id=document_id,
                    document_path=document_path,
                )
                await self._add_description_and_update_centroid(
                    session,
                    entity_id,
                    entity,
                    description,
                    source_chunk_hash,
                    context_embedding,
                    document_id=document_id,
                    document_path=document_path,
                )
                await self._add_or_increment_type(session, entity_id, entity_type)
                return EntityResolutionResult(
                    is_new=True, entity_id=entity_id, canonical_name=normalized_name
                )

            existing_result = await session.execute(
                select(GraphEntity).where(
                    GraphEntity.canonical_name == normalized_name,
                    GraphEntity.collection_id == self._collection_id,
                )
            )
            existing = existing_result.scalar_one_or_none()
            if existing:
                search_text = f"{normalized_name}: {description}"
                context_embedding = await self._embedding.embed_query(search_text)
                await self._add_description_and_update_centroid(
                    session,
                    existing.id,
                    existing,
                    description,
                    source_chunk_hash,
                    context_embedding,
                    document_id=document_id,
                    document_path=document_path,
                )
                await self._add_or_increment_type(session, existing.id, entity_type)
                return EntityResolutionResult(
                    is_new=False, entity_id=existing.id, canonical_name=existing.canonical_name
                )

        # Step 1: Exact alias lookup
        alias_result = await session.execute(
            select(EntityAlias)
            .where(
                EntityAlias.alias_name == normalized_name,
                EntityAlias.collection_id == self._collection_id,
            )
        )
        alias = alias_result.scalar_one_or_none()
        if alias:
            entity_result = await session.get(GraphEntity, alias.entity_id)
            if entity_result:
                logger.debug("Alias match: %s -> %s", normalized_name, entity_result.canonical_name)
                search_text = f"{normalized_name}: {description}"
                context_embedding = await self._embedding.embed_query(search_text)
                await self._add_description_and_update_centroid(
                    session, entity_result.id, entity_result, description,
                    source_chunk_hash, context_embedding,
                    document_id=document_id,
                    document_path=document_path,
                )
                await self._add_or_increment_type(session, entity_result.id, entity_type)
                return EntityResolutionResult(
                    is_new=False, entity_id=entity_result.id, canonical_name=entity_result.canonical_name
                )

        # Check canonical name directly
        existing_result = await session.execute(
            select(GraphEntity).where(
                GraphEntity.canonical_name == normalized_name,
                GraphEntity.collection_id == self._collection_id,
            )
        )
        existing = existing_result.scalar_one_or_none()
        if existing:
            logger.debug("Canonical match: %s", normalized_name)
            search_text = f"{normalized_name}: {description}"
            context_embedding = await self._embedding.embed_query(search_text)
            await self._add_description_and_update_centroid(
                session, existing.id, existing, description, source_chunk_hash, context_embedding,
                document_id=document_id,
                document_path=document_path,
            )
            await self._add_or_increment_type(session, existing.id, entity_type)
            return EntityResolutionResult(
                is_new=False, entity_id=existing.id, canonical_name=existing.canonical_name
            )

        # Step 2: Embedding similarity (if embedding is real, not hash-based)
        search_text = f"{normalized_name}: {description}"
        query_embedding = await self._embedding.embed_query(search_text)

        # For now, skip centroid search if using hash embeddings (dimensions < 100)
        if (
            self._embedding.dimensions >= 100
            and not self._requires_exact_name_resolution(
                normalized_name,
                entity_type,
            )
        ):
            entity_match = await self._find_similar_entity(
                session, query_embedding, normalized_name, entity_type
            )
            if entity_match:
                logger.debug("Embedding match: %s -> %s", normalized_name, entity_match.canonical_name)
                await self._add_alias(
                    session,
                    entity_match.id,
                    normalized_name,
                    source_chunk_hash,
                    document_id=document_id,
                    document_path=document_path,
                )
                await self._add_description_and_update_centroid(
                    session, entity_match.id, entity_match, description, source_chunk_hash, query_embedding,
                    document_id=document_id,
                    document_path=document_path,
                )
                await self._add_or_increment_type(session, entity_match.id, entity_type)
                return EntityResolutionResult(
                    is_new=False, entity_id=entity_match.id, canonical_name=entity_match.canonical_name
                )

        # Step 3: Create new entity (atomic upsert)
        for _attempt in range(3):
            new_id = uuid.uuid4()
            stmt = (
                pg_insert(GraphEntity)
                .values(
                    id=new_id,
                    canonical_name=normalized_name,
                    primary_type=entity_type,
                    description_count=0,
                    collection_id=self._collection_id,
                )
                .on_conflict_do_nothing(
                    constraint="uq_graph_entities_canonical_name_collection_id"
                )
                .returning(GraphEntity.id)
            )
            result = await session.execute(stmt)
            row = result.fetchone()
            if row:
                entity_id = row[0]
                entity = await session.get(GraphEntity, entity_id)
                await self._add_alias(
                    session, entity_id, normalized_name, source_chunk_hash,
                    document_id=document_id,
                    document_path=document_path,
                )
                await self._add_description_and_update_centroid(
                    session,
                    entity_id,
                    entity,
                    description,
                    source_chunk_hash,
                    query_embedding,
                    document_id=document_id,
                    document_path=document_path,
                )
                await self._add_or_increment_type(session, entity_id, entity_type)
                logger.info("New entity created: %s (type=%s)", normalized_name, entity_type)
                return EntityResolutionResult(
                    is_new=True, entity_id=entity_id, canonical_name=normalized_name
                )

            # Conflict — another worker inserted it
            existing_result = await session.execute(
                select(GraphEntity).where(
                    GraphEntity.canonical_name == normalized_name,
                    GraphEntity.collection_id == self._collection_id,
                )
            )
            existing = existing_result.scalar_one_or_none()
            if existing:
                logger.debug("Concurrent entity creation, using existing: %s", normalized_name)
                await self._add_description_and_update_centroid(
                    session, existing.id, existing, description, source_chunk_hash, query_embedding,
                    document_id=document_id,
                    document_path=document_path,
                )
                await self._add_or_increment_type(session, existing.id, entity_type)
                return EntityResolutionResult(
                    is_new=False, entity_id=existing.id, canonical_name=existing.canonical_name
                )

        raise RuntimeError(f"Failed to resolve entity after retries: {normalized_name}")

    async def resolve_relationship(
        self,
        session: AsyncSession,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        description: str,
        keywords: list[str],
        weight: float,
        source_chunk_hash: str,
        rel_type: str = "RELATES_TO",
        document_id: uuid.UUID | None = None,
        document_path: str | None = None,
    ) -> RelationshipResolutionResult:
        # Normalize rel_type using alias table
        rel_type_resolution = await self._resolve_rel_type(session, rel_type)
        rel_type = rel_type_resolution.canonical_type

        # Check for existing relationship (bidirectional, scoped to rel_type).
        # Two rels between the same pair with different rel_types are
        # distinct edges (multi-dimensional graph) and must not merge.
        existing_result = await session.execute(
            select(GraphRelationship).where(
                GraphRelationship.relationship_type_id == rel_type_resolution.relationship_type_id,
            ).where(
                (
                    (GraphRelationship.source_entity_id == source_entity_id)
                    & (GraphRelationship.target_entity_id == target_entity_id)
                )
                | (
                    (GraphRelationship.source_entity_id == target_entity_id)
                    & (GraphRelationship.target_entity_id == source_entity_id)
                )
            )
        )
        existing = existing_result.scalar_one_or_none()

        if existing:
            # Fetch source/target names
            src_entity = await session.get(GraphEntity, existing.source_entity_id)
            tgt_entity = await session.get(GraphEntity, existing.target_entity_id)
            src_name = src_entity.canonical_name if src_entity else ""
            tgt_name = tgt_entity.canonical_name if tgt_entity else ""
            await self._add_relationship_description(
                session, existing.id, description, keywords, source_chunk_hash,
                source_name=src_name, target_name=tgt_name,
                rel_type=rel_type,
                document_id=document_id,
                document_path=document_path,
            )
            new_kw = list(set(existing.keywords or []) | set(keywords))
            max_weight = settings.graph_rag_max_relationship_weight
            await session.execute(
                text(
                    "UPDATE graph_relationships SET"
                    " keywords = CAST(:kw AS json),"
                    " weight = LEAST("
                    "   (SELECT COALESCE(SUM(weight), 0) FROM relationship_descriptions"
                    "    WHERE relationship_id = :rel_id),"
                    "   :max_weight"
                    " ) WHERE id = :rel_id"
                ),
                {"kw": str(new_kw).replace("'", '"'), "rel_id": existing.id, "max_weight": max_weight},
            )
            return RelationshipResolutionResult(is_new=False, relationship_id=existing.id)

        new_rel_id = uuid.uuid4()
        rel = GraphRelationship(
            id=new_rel_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            weight=int(weight * 10),
            keywords=keywords,
            relationship_type_id=rel_type_resolution.relationship_type_id,
            rel_type=rel_type,
            collection_id=self._collection_id,
        )
        session.add(rel)

        # Fetch source/target names for embedding
        src_entity = await session.get(GraphEntity, source_entity_id)
        tgt_entity = await session.get(GraphEntity, target_entity_id)
        src_name = src_entity.canonical_name if src_entity else ""
        tgt_name = tgt_entity.canonical_name if tgt_entity else ""

        desc = RelationshipDescription(
            id=uuid.uuid4(),
            relationship_id=new_rel_id,
            description=description,
            keywords=keywords,
            weight=1,
            source_chunk_hashes=[source_chunk_hash],
            document_id=document_id,
            document_path=document_path,
        )
        session.add(desc)
        await session.commit()

        # Write relationship embedding
        embed_text = relationship_embedding_text(
            src_name,
            tgt_name,
            rel_type,
            description,
            keywords,
        )
        embedding = await self._embedding.embed_query(embed_text)
        await self._vstore.upsert_relationship_embedding(
            relationship_id=new_rel_id,
            collection_id=self._collection_id,
            source_name=src_name,
            target_name=tgt_name,
            description=description,
            embedding=embedding,
            document_id=document_id,
            document_path=document_path,
        )

        return RelationshipResolutionResult(is_new=True, relationship_id=new_rel_id)

    async def _find_similar_entity(
        self,
        session: AsyncSession,
        query_embedding: list[float],
        name: str,
        entity_type: str,
    ) -> GraphEntity | None:
        """Search for similar entities using centroid proximity via pgvector."""
        centroid_hits = await self._vstore.search_entity_centroids(
            collection_id=self._collection_id,
            query_embedding=query_embedding,
            top_k=5,
        )

        for hit in centroid_hits:
            entity_id_str = hit.metadata.get("entity_id")
            if not entity_id_str:
                continue
            try:
                entity_id = uuid.UUID(entity_id_str)
            except ValueError:
                continue

            entity = await session.get(GraphEntity, entity_id)
            if not entity:
                continue

            similarity = 1.0 - hit.distance
            if similarity >= self.HIGH_CONFIDENCE_SIMILARITY:
                if not self._types_compatible(entity_type, entity.primary_type or ""):
                    continue
                return entity
            elif similarity >= self.MEDIUM_CONFIDENCE_SIMILARITY:
                if not self._types_compatible(entity_type, entity.primary_type or ""):
                    continue
                if self._fuzzy_match(name, hit.metadata.get("canonical_name", "")):
                    return entity
            else:
                break

        # Fallback: fuzzy name matching against all entities in collection
        entities_result = await session.execute(
            select(GraphEntity).where(GraphEntity.collection_id == self._collection_id)
        )
        candidates = entities_result.scalars().all()
        for candidate in candidates:
            if self._fuzzy_match(name, candidate.canonical_name):
                if not self._types_compatible(entity_type, candidate.primary_type or ""):
                    continue
                return candidate
        return None

    async def _add_description_and_update_centroid(
        self,
        session: AsyncSession,
        entity_id: uuid.UUID,
        entity: GraphEntity | None,
        description: str,
        source_chunk_hash: str,
        context_embedding: list[float],
        document_id: uuid.UUID | None = None,
        document_path: str | None = None,
    ) -> None:
        if not description:
            return

        if entity is None:
            entity = await session.get(GraphEntity, entity_id)
        if entity is None:
            return

        await self._acquire_entity_lock(session, entity_id)

        embed_text = f"{entity.canonical_name}: {description}"
        embedding = await self._embedding.embed_query(embed_text)

        existing_descs = await self._vstore.search_entity_embeddings(
            collection_id=self._collection_id,
            query_embedding=embedding,
            top_k=1,
            entity_id=entity_id,
        )
        if existing_descs:
            best_match = existing_descs[0]
            cosine_sim = 1.0 - best_match.distance
            if cosine_sim >= self.DESCRIPTION_SIMILARITY_THRESHOLD:
                desc_id = best_match.metadata.get("description_id")
                if desc_id:
                    try:
                        existing_desc_id = uuid.UUID(desc_id)
                    except ValueError:
                        existing_desc_id = None
                    if existing_desc_id:
                        existing_desc = await session.get(
                            EntityDescription, existing_desc_id
                        )
                        hashes = list(existing_desc.source_chunk_hashes or []) if existing_desc else []
                        if source_chunk_hash not in hashes:
                            hashes.append(source_chunk_hash)
                        await session.execute(
                            update(EntityDescription)
                            .where(EntityDescription.id == existing_desc_id)
                            .values(
                                weight=EntityDescription.weight + 1,
                                source_chunk_hashes=hashes,
                            )
                        )
                        return

        desc = EntityDescription(
            id=uuid.uuid4(),
            entity_id=entity_id,
            description=description,
            weight=1,
            source_chunk_hashes=[source_chunk_hash],
            document_id=document_id,
            document_path=document_path,
        )
        desc_id = desc.id
        session.add(desc)

        n = entity.description_count or 0

        # Hebbian learning: new_centroid = (old_centroid * n + new) / (n + 1)
        old_centroid = await self._vstore.get_entity_centroid(entity_id, self._collection_id)
        if old_centroid:
            new_centroid = [
                (old_c * n + new_c) / (n + 1)
                for old_c, new_c in zip(old_centroid, context_embedding)
            ]
        else:
            new_centroid = list(context_embedding)

        canonical_name = entity.canonical_name
        primary_type = entity.primary_type

        await self._vstore.upsert_entity_embedding(
            entity_id=entity_id,
            collection_id=self._collection_id,
            name=canonical_name,
            description=description,
            description_id=desc_id,
            embedding=embedding,
            document_id=document_id,
            document_path=document_path,
            session=session,
        )

        await self._vstore.upsert_entity_centroid(
            entity_id=entity_id,
            collection_id=self._collection_id,
            canonical_name=canonical_name,
            primary_type=primary_type,
            description_count=n + 1,
            embedding=new_centroid,
            session=session,
        )

        await session.execute(
            update(GraphEntity)
            .where(GraphEntity.id == entity_id)
            .values(description_count=GraphEntity.description_count + 1)
        )

    async def _acquire_entity_lock(
        self,
        session: AsyncSession,
        entity_id: uuid.UUID,
    ) -> None:
        """Serialize concurrent centroid updates for the same entity."""
        bind = getattr(session, "bind", None)
        dialect_name = getattr(getattr(bind, "dialect", None), "name", "")
        if dialect_name != "postgresql":
            return
        lock_bytes = hashlib.sha256(
            f"{self._collection_id}:{entity_id}".encode("utf-8")
        ).digest()[:8]
        lock_key = int.from_bytes(lock_bytes, "big") & ((1 << 63) - 1)
        await session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": lock_key},
        )

    async def _add_relationship_description(
        self,
        session: AsyncSession,
        relationship_id: uuid.UUID,
        description: str,
        keywords: list[str],
        source_chunk_hash: str,
        source_name: str = "",
        target_name: str = "",
        rel_type: str = "RELATES_TO",
        document_id: uuid.UUID | None = None,
        document_path: str | None = None,
    ) -> None:
        # Check for exact (relationship_id, document_id) match
        if document_id:
            existing_exact = await session.execute(
                select(RelationshipDescription).where(
                    RelationshipDescription.relationship_id == relationship_id,
                    RelationshipDescription.document_id == document_id,
                )
            )
            existing_desc = existing_exact.scalar_one_or_none()
            if existing_desc:
                # Update existing description, don't create duplicate
                hashes = list(existing_desc.source_chunk_hashes or [])
                if source_chunk_hash not in hashes:
                    hashes.append(source_chunk_hash)
                await session.execute(
                    update(RelationshipDescription)
                    .where(RelationshipDescription.id == existing_desc.id)
                    .values(
                        weight=RelationshipDescription.weight + 1,
                        source_chunk_hashes=hashes,
                    )
                )
                return

        embed_text = relationship_embedding_text(
            source_name,
            target_name,
            rel_type,
            description,
            keywords,
        )
        embedding = await self._embedding.embed_query(embed_text)

        desc = RelationshipDescription(
            id=uuid.uuid4(),
            relationship_id=relationship_id,
            description=description,
            keywords=keywords,
            weight=1,
            source_chunk_hashes=[source_chunk_hash],
            document_id=document_id,
            document_path=document_path,
        )
        session.add(desc)

        await self._vstore.upsert_relationship_embedding(
            relationship_id=relationship_id,
            collection_id=self._collection_id,
            source_name=source_name,
            target_name=target_name,
            description=description,
            embedding=embedding,
            document_id=document_id,
            document_path=document_path,
        )

    async def _resolve_or_create_relationship_type(
        self,
        session: AsyncSession,
        canonical_type: str,
    ) -> GraphRelationshipType:
        normalized_type = normalize_rel_type(canonical_type)
        existing_result = await session.execute(
            select(GraphRelationshipType).where(
                GraphRelationshipType.collection_id == self._collection_id,
                GraphRelationshipType.canonical_type == normalized_type,
            )
        )
        existing = existing_result.scalar_one_or_none()
        if existing:
            return existing

        stmt = (
            pg_insert(GraphRelationshipType)
            .values(
                id=uuid.uuid4(),
                collection_id=self._collection_id,
                canonical_type=normalized_type,
            )
            .on_conflict_do_nothing(
                constraint="uq_graph_relationship_types_collection_canonical_type"
            )
            .returning(GraphRelationshipType.id)
        )
        result = await session.execute(stmt)
        row = result.fetchone()
        if row:
            created = await session.get(GraphRelationshipType, row[0])
            if created:
                return created

        existing_result = await session.execute(
            select(GraphRelationshipType).where(
                GraphRelationshipType.collection_id == self._collection_id,
                GraphRelationshipType.canonical_type == normalized_type,
            )
        )
        existing = existing_result.scalar_one_or_none()
        if existing:
            return existing
        raise RuntimeError(
            f"Failed to resolve relationship type after retries: {normalized_type}"
        )

    async def _find_relationship_type_by_label(
        self,
        session: AsyncSession,
        label: str,
    ) -> GraphRelationshipType | None:
        normalized_label = normalize_rel_type(label)
        alias_result = await session.execute(
            select(RelationshipTypeAlias).where(
                RelationshipTypeAlias.collection_id == self._collection_id,
                RelationshipTypeAlias.alias_type == normalized_label,
            )
        )
        alias_row = alias_result.scalar_one_or_none()
        if alias_row:
            relationship_type = await session.get(
                GraphRelationshipType,
                alias_row.relationship_type_id,
            )
            if relationship_type:
                return relationship_type

        canonical_result = await session.execute(
            select(GraphRelationshipType).where(
                GraphRelationshipType.collection_id == self._collection_id,
                GraphRelationshipType.canonical_type == normalized_label,
            )
        )
        return canonical_result.scalar_one_or_none()

    async def _add_relationship_type_alias(
        self,
        session: AsyncSession,
        relationship_type_id: uuid.UUID,
        canonical_type: str,
        alias_type: str,
    ) -> None:
        normalized_alias = normalize_rel_type(alias_type)
        stmt = (
            pg_insert(RelationshipTypeAlias)
            .values(
                id=uuid.uuid4(),
                collection_id=self._collection_id,
                relationship_type_id=relationship_type_id,
                canonical_type=normalize_rel_type(canonical_type),
                alias_type=normalized_alias,
                frequency=1,
            )
            .on_conflict_do_update(
                constraint="uq_relationship_type_aliases_collection_alias_type",
                set_={
                    "relationship_type_id": relationship_type_id,
                    "canonical_type": normalize_rel_type(canonical_type),
                    "frequency": RelationshipTypeAlias.frequency + 1,
                },
            )
        )
        await session.execute(stmt)

    async def _record_relationship_type_observation(
        self,
        session: AsyncSession,
        relationship_type: GraphRelationshipType,
        observed_label: str,
    ) -> GraphRelationshipType:
        await self._add_relationship_type_alias(
            session,
            relationship_type.id,
            relationship_type.canonical_type,
            observed_label,
        )
        updated = await session.get(GraphRelationshipType, relationship_type.id)
        if updated is None:
            updated = relationship_type
        return await self._reelect_relationship_type_canonical(session, updated)

    async def _reelect_relationship_type_canonical(
        self,
        session: AsyncSession,
        relationship_type: GraphRelationshipType,
    ) -> GraphRelationshipType:
        alias_result = await session.execute(
            select(RelationshipTypeAlias)
            .where(RelationshipTypeAlias.relationship_type_id == relationship_type.id)
            .order_by(
                RelationshipTypeAlias.frequency.desc(),
                RelationshipTypeAlias.alias_type.asc(),
            )
        )
        aliases = alias_result.scalars().all()
        if not aliases:
            return relationship_type
        best_alias = aliases[0]
        best_canonical = normalize_rel_type(best_alias.alias_type)
        old_canonical = normalize_rel_type(relationship_type.canonical_type)
        if best_canonical == old_canonical:
            return relationship_type

        await session.execute(
            update(GraphRelationshipType)
            .where(GraphRelationshipType.id == relationship_type.id)
            .values(canonical_type=best_canonical)
        )
        await session.execute(
            update(RelationshipTypeAlias)
            .where(RelationshipTypeAlias.relationship_type_id == relationship_type.id)
            .values(canonical_type=best_canonical)
        )

        rel_rows = (
            await session.execute(
                select(GraphRelationship).where(
                    GraphRelationship.relationship_type_id == relationship_type.id
                )
            )
        ).scalars().all()

        await session.execute(
            update(GraphRelationship)
            .where(GraphRelationship.relationship_type_id == relationship_type.id)
            .values(rel_type=best_canonical)
        )

        if rel_rows:
            await self._graph_storage.relabel_edges(
                old_rel_type=old_canonical,
                new_rel_type=best_canonical,
                edges=[
                    {
                        "source_id": str(rel.source_entity_id),
                        "target_id": str(rel.target_entity_id),
                        "id": str(rel.id),
                        "weight": int(rel.weight or 1),
                        "keywords": rel.keywords or [],
                        "collection_id": str(self._collection_id),
                        "rel_type": best_canonical,
                    }
                    for rel in rel_rows
                ],
            )

        relationship_type.canonical_type = best_canonical
        return relationship_type

    async def _add_alias(
        self,
        session: AsyncSession,
        entity_id: uuid.UUID,
        alias_name: str,
        source_chunk_hash: str,
        document_id: uuid.UUID | None = None,
        document_path: str | None = None,
    ) -> None:
        stmt = (
            pg_insert(EntityAlias)
            .values(
                id=uuid.uuid4(),
                collection_id=self._collection_id,
                alias_name=alias_name,
                entity_id=entity_id,
                source_chunk_hash=source_chunk_hash,
                document_id=document_id,
                document_path=document_path,
            )
            .on_conflict_do_nothing()
        )
        await session.execute(stmt)

    async def _add_or_increment_type(
        self, session: AsyncSession, entity_id: uuid.UUID, type_name: str
    ) -> None:
        if not type_name:
            return
        stmt = (
            pg_insert(EntityType)
            .values(id=uuid.uuid4(), entity_id=entity_id, type_name=type_name, frequency=1)
            .on_conflict_do_update(
                constraint="uq_entity_types_entity_type",
                set_={"frequency": EntityType.frequency + 1},
            )
        )
        await session.execute(stmt)

    def _types_compatible(self, type_a: str, type_b: str) -> bool:
        if not type_a or not type_b:
            return True
        a, b = type_a.strip().lower(), type_b.strip().lower()
        if a == b:
            return True
        incompatible = {
            frozenset({"person", "place"}),
            frozenset({"person", "object"}),
            frozenset({"place", "concept"}),
        }
        return frozenset({a, b}) not in incompatible

    @staticmethod
    def _strip_diacritics(text: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )

    @staticmethod
    def _normalize_for_comparison(name: str) -> str:
        text = IncrementalEntityResolver._strip_diacritics(name).lower().strip()
        text = re.sub(r"^(the|a|an)\s+", "", text)
        text = re.sub(r"[\s\-]", "", text)
        return text

    @staticmethod
    def _normalize_entity_name(name: str) -> str:
        if not name:
            return ""
        normalized = " ".join(name.strip().split())
        if len(normalized) > 256:
            normalized = normalized[:256]
        return normalized

    def _fuzzy_match(self, name1: str, name2: str) -> bool:
        if not name1 or not name2:
            return False
        if self._requires_exact_name_resolution(name1) or self._requires_exact_name_resolution(name2):
            return self._normalize_for_comparison(name1) == self._normalize_for_comparison(name2)
        n1 = self._normalize_for_comparison(name1)
        n2 = self._normalize_for_comparison(name2)
        if n1 == n2:
            return True
        ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
        return ratio >= self.FUZZY_NAME_THRESHOLD

    def _requires_exact_name_resolution(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> bool:
        if not self._domain_cfg.requires_exact_resolution:
            return False
        normalized_type = (entity_type or "").strip().upper()
        if normalized_type in {
            "FUNCTION",
            "METHOD",
            "CLASS",
            "MODULE",
            "PACKAGE",
            "VARIABLE",
            "INTERFACE",
            "EXCEPTION",
            "CONFIG",
        }:
            return True
        return self._looks_like_code_symbol(name)

    @staticmethod
    def _looks_like_code_symbol(name: str) -> bool:
        if not name:
            return False
        if any(ch in name for ch in "._[](){}'\"`:/\\"):
            return True
        if "_" in name:
            return True
        if re.search(r"\bself\.", name):
            return True
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))

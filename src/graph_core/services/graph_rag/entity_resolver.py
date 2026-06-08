"""Incremental entity resolver — zero-LLM entity resolution.

Three-tier pipeline:
1. Exact alias lookup in EntityAlias table
2. Embedding similarity against entity centroids
3. Fuzzy name matching
"""

from __future__ import annotations

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
from graph_core.models.graph_rag import (
    EntityAlias,
    EntityDescription,
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipDescription,
)
from graph_core.models.rel_types import (
    normalize_rel_type,
    relationship_embedding_text,
)
from graph_core.storage.graph_rag_vectors import GraphRAGVectorStore

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
    ) -> None:
        self._embedding = embedding_provider
        self._collection_id = collection_id
        self._vstore = GraphRAGVectorStore()

    async def resolve_entity(
        self,
        session: AsyncSession,
        name: str,
        entity_type: str,
        description: str,
        source_chunk_hash: str,
    ) -> EntityResolutionResult:
        normalized_name = name.strip().title()

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
                )
                self._add_or_increment_type(session, entity_result.id, entity_type)
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
            )
            self._add_or_increment_type(session, existing.id, entity_type)
            return EntityResolutionResult(
                is_new=False, entity_id=existing.id, canonical_name=existing.canonical_name
            )

        # Step 2: Embedding similarity (if embedding is real, not hash-based)
        search_text = f"{normalized_name}: {description}"
        query_embedding = await self._embedding.embed_query(search_text)

        # For now, skip centroid search if using hash embeddings (dimensions < 100)
        if self._embedding.dimensions >= 100:
            entity_match = await self._find_similar_entity(
                session, query_embedding, normalized_name, entity_type
            )
            if entity_match:
                logger.debug("Embedding match: %s -> %s", normalized_name, entity_match.canonical_name)
                await self._add_alias(session, entity_match.id, normalized_name, source_chunk_hash)
                await self._add_description_and_update_centroid(
                    session, entity_match.id, entity_match, description, source_chunk_hash, query_embedding,
                )
                self._add_or_increment_type(session, entity_match.id, entity_type)
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
                    session, entity_id, normalized_name, source_chunk_hash
                )
                await self._add_description_and_update_centroid(
                    session,
                    entity_id,
                    entity,
                    description,
                    source_chunk_hash,
                    query_embedding,
                )
                self._add_or_increment_type(session, entity_id, entity_type)
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
                )
                self._add_or_increment_type(session, existing.id, entity_type)
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
    ) -> RelationshipResolutionResult:
        # Check for existing relationship (bidirectional, scoped to rel_type).
        # Two rels between the same pair with different rel_types are
        # distinct edges (multi-dimensional graph) and must not merge.
        existing_result = await session.execute(
            select(GraphRelationship).where(
                GraphRelationship.rel_type == rel_type,
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
        )

        # Store prefix embedding for this rel_type so dimension ranking at
        # query time is a pure-CPU cosine similarity instead of N API calls.
        normalized = normalize_rel_type(rel_type)
        prefix_embedding = await self._embedding.embed_query(normalized)
        await self._vstore.ensure_prefix_embeddings_table(self._collection_id)
        await self._vstore.upsert_prefix_embedding(
            self._collection_id, normalized, prefix_embedding
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
    ) -> None:
        if not description:
            return

        if entity is None:
            entity = await session.get(GraphEntity, entity_id)
        if entity is None:
            return

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
    ) -> None:
        embed_text = relationship_embedding_text(
            source_name,
            target_name,
            rel_type,
            description,
            keywords,
        )
        embedding = await self._embedding.embed_query(embed_text)

        existing_descs = await self._vstore.search_relationship_embeddings(
            collection_id=self._collection_id,
            query_embedding=embedding,
            top_k=1,
            relationship_id=relationship_id,
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
                            RelationshipDescription, existing_desc_id
                        )
                        hashes = list(existing_desc.source_chunk_hashes or []) if existing_desc else []
                        if source_chunk_hash not in hashes:
                            hashes.append(source_chunk_hash)
                        await session.execute(
                            update(RelationshipDescription)
                            .where(RelationshipDescription.id == existing_desc_id)
                            .values(
                                weight=RelationshipDescription.weight + 1,
                                source_chunk_hashes=hashes,
                            )
                        )
                        return

        desc = RelationshipDescription(
            id=uuid.uuid4(),
            relationship_id=relationship_id,
            description=description,
            keywords=keywords,
            weight=1,
            source_chunk_hashes=[source_chunk_hash],
        )
        session.add(desc)

        await self._vstore.upsert_relationship_embedding(
            relationship_id=relationship_id,
            collection_id=self._collection_id,
            source_name=source_name,
            target_name=target_name,
            description=description,
            embedding=embedding,
        )

    async def _add_alias(
        self,
        session: AsyncSession,
        entity_id: uuid.UUID,
        alias_name: str,
        source_chunk_hash: str,
    ) -> None:
        stmt = (
            pg_insert(EntityAlias)
            .values(
                id=uuid.uuid4(),
                collection_id=self._collection_id,
                alias_name=alias_name,
                entity_id=entity_id,
                source_chunk_hash=source_chunk_hash,
            )
            .on_conflict_do_nothing()
        )
        await session.execute(stmt)

    def _add_or_increment_type(
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
        session.execute(stmt)

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

    def _fuzzy_match(self, name1: str, name2: str) -> bool:
        if not name1 or not name2:
            return False
        n1 = self._normalize_for_comparison(name1)
        n2 = self._normalize_for_comparison(name2)
        if n1 == n2:
            return True
        ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
        return ratio >= self.FUZZY_NAME_THRESHOLD

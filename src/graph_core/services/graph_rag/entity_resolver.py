"""Canonical resolution: embeddings propose candidates; the decision model decides identity.

Text generation lives in the extractor. Similarity is never proof of identity.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import re
import unicodedata
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graph_core.services.graph_rag.ingestion_batch import (
        EntityInput,
        RelationshipInput,
    )

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from graph_core.decisions import GraphDecisions
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.models.domain_config import get_domain_config
from graph_core.models.graph_rag import (
    EntityAlias,
    EntityType,
    GraphRelationship,
    GraphRelationshipType,
    RelationshipTypeAlias,
)
from graph_core.models.rel_types import normalize_rel_type
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
        source_text: str = "",
        decisions: GraphDecisions | None = None,
    ) -> None:
        self._decisions = decisions or GraphDecisions()
        self._source_text = source_text
        self._identity_traces: dict[str, dict] = {}
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
        from graph_core.services.graph_rag.ingestion_batch import IngestionDecisionBatch

        self._ingestion_batch = IngestionDecisionBatch(self)

    async def resolve_entities(
        self,
        session: AsyncSession,
        entities: Sequence[EntityInput],
        source_chunk_hash: str,
        document_id: uuid.UUID | None = None,
        document_path: str | None = None,
        *,
        defer_descriptions: bool = False,
    ) -> list[EntityResolutionResult]:
        results = await self._ingestion_batch.resolve_entities(
            session, entities, source_chunk_hash, document_id, document_path
        )
        if not defer_descriptions:
            await self.flush_descriptions(session)
        return results

    async def resolve_relationships(
        self,
        session: AsyncSession,
        relationships: Sequence[RelationshipInput],
        source_chunk_hash: str,
        document_id: uuid.UUID | None = None,
        document_path: str | None = None,
        *,
        defer_descriptions: bool = False,
    ) -> list[RelationshipResolutionResult]:
        results = await self._ingestion_batch.resolve_relationships(
            session, relationships, source_chunk_hash, document_id, document_path
        )
        if not defer_descriptions:
            await self.flush_descriptions(session)
        return results

    async def flush_descriptions(self, session: AsyncSession) -> None:
        await self._ingestion_batch.flush(session)

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
        """Single-item compatibility API; chunk ingestion uses resolve_entities."""
        from types import SimpleNamespace

        results = await self.resolve_entities(
            session,
            [
                SimpleNamespace(
                    name=name, entity_type=entity_type, description=description
                )
            ],
            source_chunk_hash,
            document_id,
            document_path,
        )
        return results[0]

    async def resolve_relationship(
        self,
        session: AsyncSession,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        description: str,
        keywords: list[str],
        source_chunk_hash: str,
        rel_type: str = "RELATES_TO",
        document_id: uuid.UUID | None = None,
        document_path: str | None = None,
        original_source_name: str | None = None,
        original_target_name: str | None = None,
    ) -> RelationshipResolutionResult:
        """Single-item compatibility API; chunk ingestion batches relationships."""
        results = await self.resolve_relationships(
            session,
            [
                {
                    "source_entity_id": source_entity_id,
                    "target_entity_id": target_entity_id,
                    "description": description,
                    "keywords": keywords,
                    "rel_type": rel_type,
                    "original_source_name": original_source_name,
                    "original_target_name": original_target_name,
                }
            ],
            source_chunk_hash,
            document_id,
            document_path,
        )
        return results[0]

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
            (
                await session.execute(
                    select(GraphRelationship).where(
                        GraphRelationship.relationship_type_id == relationship_type.id
                    )
                )
            )
            .scalars()
            .all()
        )

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
                identity_decision=self._identity_traces.get(alias_name),
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
            .values(
                id=uuid.uuid4(), entity_id=entity_id, type_name=type_name, frequency=1
            )
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
            c
            for c in unicodedata.normalize("NFKD", text)
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
        if self._requires_exact_name_resolution(
            name1
        ) or self._requires_exact_name_resolution(name2):
            return self._normalize_for_comparison(
                name1
            ) == self._normalize_for_comparison(name2)
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

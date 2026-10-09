"""Plan decisions on snapshots, infer in batches, then persist under short locks.

The resolver owns one instance per chunk. Pending description work can span the
entity and relationship stages, so both share the same support forward passes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, TypeVar

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from graph_core.database import AsyncSessionLocal
from graph_core.decisions.graph import IDENTITY_MIN_PROBABILITY
from graph_core.models.graph_rag import (
    EntityAlias,
    EntityDescription,
    EntityResolutionDecision,
    GraphEntity,
    GraphRelationship,
    RelationshipDescription,
)

if TYPE_CHECKING:
    from graph_core.services.graph_rag.entity_resolver import (
        EntityResolutionResult,
        IncrementalEntityResolver,
        RelationshipResolutionResult,
    )


class EntityInput(Protocol):
    name: str
    entity_type: str
    description: str


class RelationshipInput(TypedDict, total=False):
    source_entity_id: uuid.UUID
    target_entity_id: uuid.UUID
    description: str
    keywords: list[str]
    rel_type: str
    original_source_name: str | None
    original_target_name: str | None


T = TypeVar("T")

logger = logging.getLogger(__name__)


async def persist_transaction(bind, apply: Callable[[AsyncSession], Awaitable[T]]) -> T:
    """Retry only the rolled-back write, never already-committed work or inference."""
    for attempt in range(3):
        try:
            async with AsyncSessionLocal(bind=bind) as writer:
                result = await apply(writer)
                await writer.commit()
                return result
        except DBAPIError as exc:
            sqlstate = getattr(exc.orig, "sqlstate", None) or getattr(
                exc.orig, "pgcode", None
            )
            if sqlstate != "40P01" or attempt == 2:
                raise
            logger.warning(
                "Ingestion write deadlock; retrying attempt=%d/3", attempt + 1
            )
            await asyncio.sleep(random.uniform(0.1, 0.2) * (2**attempt))
    raise RuntimeError("Ingestion transaction retry attempts exhausted")


def snapshot(rows: list[Any]) -> str:
    return json.dumps(
        sorted(
            (str(row.id), row.description, row.source_evidence or []) for row in rows
        ),
        sort_keys=True,
    )


def evidence_key(item: dict) -> tuple:
    return item.get("document_id"), item["chunk_hash"]


class IngestionDecisionBatch:
    def __init__(self, resolver: IncrementalEntityResolver) -> None:
        self.resolver = resolver
        self.entities: list[dict] = []
        self.relationships: list[dict] = []

    async def resolve_entities(
        self,
        session: AsyncSession,
        entities: Sequence[EntityInput],
        chunk_hash: str,
        document_id: uuid.UUID | None,
        document_path: str | None,
    ) -> list[EntityResolutionResult]:
        """Batch candidate pairs, including earlier entities in this same chunk."""
        r = self.resolver
        await session.commit()
        if not entities:
            return []
        inputs = []
        for item in entities:
            # Accept extractor records without coupling the resolver to the extractor.
            inputs.append(
                {
                    "name": r._normalize_entity_name(item.name),
                    "type": item.entity_type,
                    "description": item.description,
                    "provisional_id": uuid.uuid4(),
                }
            )
        texts = [f"{item['name']}: {item['description']}" for item in inputs]
        if texts and hasattr(r._embedding, "embed_documents"):
            embeddings = await r._embedding.embed_documents(texts)
        else:
            embeddings = [await r._embedding.embed_query(value) for value in texts]
        if len(embeddings) != len(inputs):
            raise RuntimeError(
                "Embedding provider returned the wrong number of vectors"
            )

        pairs = []
        candidates_by_input: list[list[dict]] = []
        exact_matches: list[dict | None] = []
        async with AsyncSessionLocal(bind=session.bind) as reader:
            stored = (
                (
                    await reader.execute(
                        select(GraphEntity).where(
                            GraphEntity.collection_id == r._collection_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            by_id = {row.id: row for row in stored}
            by_name = {row.canonical_name: row for row in stored}
            aliases = (
                (
                    await reader.execute(
                        select(EntityAlias).where(
                            EntityAlias.collection_id == r._collection_id,
                            EntityAlias.alias_name.in_(
                                [item["name"] for item in inputs]
                            ),
                        )
                    )
                )
                .scalars()
                .all()
                if inputs
                else []
            )
            alias_ids = {row.alias_name: row.entity_id for row in aliases}
            description_cache: dict[uuid.UUID, list[str]] = {}

            async def candidate(row):
                if row.id not in description_cache:
                    description_cache[row.id] = list(
                        (
                            await reader.execute(
                                select(EntityDescription.description)
                                .where(EntityDescription.entity_id == row.id)
                                .order_by(
                                    EntityDescription.created_at, EntityDescription.id
                                )
                                .limit(5)
                            )
                        )
                        .scalars()
                        .all()
                    )
                return {
                    "id": row.id,
                    "name": row.canonical_name,
                    "type": row.primary_type,
                    "descriptions": description_cache[row.id],
                }

            for index, incoming in enumerate(inputs):
                exact = by_name.get(incoming["name"])
                exact_matches.append(await candidate(exact) if exact else None)
                proposed: dict[uuid.UUID, dict] = {}
                exact_only = incoming[
                    "type"
                ] == "base_entity_ref" or r._requires_exact_name_resolution(
                    incoming["name"], incoming["type"]
                )
                if exact is None and not exact_only:
                    alias = by_id.get(alias_ids.get(incoming["name"]))
                    if alias:
                        proposed[alias.id] = await candidate(alias)
                    if r._embedding.dimensions >= 100:
                        embedding_candidates = {}
                        hits = await r._vstore.search_entity_centroids(
                            collection_id=r._collection_id,
                            query_embedding=embeddings[index],
                            top_k=5,
                        )
                        for hit in hits:
                            if 1 - hit.distance < r.MEDIUM_CONFIDENCE_SIMILARITY:
                                continue
                            try:
                                row = by_id.get(
                                    uuid.UUID(hit.metadata.get("entity_id", ""))
                                )
                            except (ValueError, TypeError, AttributeError):
                                continue
                            if row and r._types_compatible(
                                incoming["type"], row.primary_type or ""
                            ):
                                embedding_candidates[row.id] = (
                                    1 - hit.distance,
                                    await candidate(row),
                                )
                        fuzzy_candidates: dict[uuid.UUID, dict] = {}
                        for row in stored:
                            if r._fuzzy_match(
                                incoming["name"], row.canonical_name
                            ) and r._types_compatible(
                                incoming["type"], row.primary_type or ""
                            ):
                                fuzzy_candidates.setdefault(
                                    row.id, await candidate(row)
                                )
                        for previous_index, previous in enumerate(inputs[:index]):
                            if not r._types_compatible(
                                incoming["type"], previous["type"]
                            ):
                                continue
                            a, b = embeddings[index], embeddings[previous_index]
                            norm = (
                                sum(v * v for v in a) ** 0.5
                                * sum(v * v for v in b) ** 0.5
                            )
                            similarity = (
                                sum(x * y for x, y in zip(a, b)) / norm if norm else 0
                            )
                            fuzzy = r._fuzzy_match(incoming["name"], previous["name"])
                            if similarity >= r.MEDIUM_CONFIDENCE_SIMILARITY or fuzzy:
                                # Earlier inputs only: no cycles or forward references.
                                known = by_name.get(previous["name"])
                                prior = (
                                    await candidate(known)
                                    if known
                                    else {
                                        "id": previous["provisional_id"],
                                        "name": previous["name"],
                                        "type": previous["type"],
                                        "descriptions": [previous["description"]],
                                    }
                                )
                                if fuzzy:
                                    fuzzy_candidates.setdefault(prior["id"], prior)
                                else:
                                    embedding_candidates[prior["id"]] = (
                                        similarity,
                                        prior,
                                    )
                        for _, prior in sorted(
                            embedding_candidates.values(), key=lambda item: -item[0]
                        )[:5]:
                            proposed.setdefault(prior["id"], prior)
                        for candidate_id, prior in fuzzy_candidates.items():
                            proposed.setdefault(candidate_id, prior)
                candidates_by_input.append(list(proposed.values()))
                for candidate_index, proposed_candidate in enumerate(proposed.values()):
                    pair_id = f"identity_{index}_{len(pairs)}"
                    proposed_candidate = dict(proposed_candidate, question_id=pair_id)
                    # Question references are internal, not model evidence.
                    candidates_by_input[index][candidate_index] = proposed_candidate
                    pairs.append(
                        {
                            "id": pair_id,
                            "incoming_id": str(index),
                            "candidate_id": str(proposed_candidate["id"]),
                            "incoming": {
                                key: incoming[key]
                                for key in ("name", "type", "description")
                            },
                            "candidate": {
                                key: proposed_candidate[key]
                                for key in ("name", "type", "descriptions")
                            },
                        }
                    )
        # Close the reader before inference or semaphore waiting.
        decisions = (
            await r._decisions.identity_many(pairs, r._source_text) if pairs else {}
        )
        results = []
        pair_lookup = {pair["id"]: pair for pair in pairs}
        provisional_results: dict[uuid.UUID, Any] = {}
        from graph_core.services.graph_rag.entity_resolver import EntityResolutionResult

        for index, incoming in enumerate(inputs):
            selected = exact_matches[index]
            audits = []
            for proposed in candidates_by_input[index]:
                decision = decisions[proposed["question_id"]]
                accepted = (
                    decision.choice == "same"
                    and decision.probabilities["same"] >= IDENTITY_MIN_PROBABILITY
                )
                trace = {**decision.trace("entity_identity"), "accepted": accepted}
                resolved_candidate = provisional_results.get(proposed["id"])
                candidate_id = (
                    resolved_candidate.entity_id
                    if resolved_candidate
                    else proposed["id"]
                )
                audits.append(
                    dict(
                        collection_id=r._collection_id,
                        candidate_id=candidate_id,
                        incoming_name=incoming["name"],
                        source_chunk_hash=chunk_hash,
                        source_evidence={
                            "incoming": pair_lookup[proposed["question_id"]][
                                "incoming"
                            ],
                            "candidate": {
                                key: proposed[key]
                                for key in ("name", "type", "descriptions")
                            },
                            "source_passage": r._source_text,
                        },
                        decision=trace,
                    )
                )
                if selected is None and accepted:
                    selected = dict(
                        proposed,
                        id=candidate_id,
                        name=resolved_candidate.canonical_name
                        if resolved_candidate
                        else proposed["name"],
                    )
                    r._identity_traces[incoming["name"]] = trace

            async def persist_entity(writer):
                # Fresh ORM objects on every attempt: detached, flushed audit
                # objects from a rolled-back transaction must not be reused.
                writer.add_all([EntityResolutionDecision(**audit) for audit in audits])
                is_new = False
                if selected:
                    entity = await writer.get(GraphEntity, selected["id"])
                    if entity is None:
                        raise RuntimeError(
                            "Identity candidate disappeared during resolution; "
                            "retry the chunk"
                        )
                else:
                    result = await writer.execute(
                        pg_insert(GraphEntity)
                        .values(
                            id=incoming["provisional_id"],
                            canonical_name=incoming["name"],
                            primary_type=incoming["type"],
                            description_count=0,
                            collection_id=r._collection_id,
                        )
                        .on_conflict_do_nothing(
                            constraint="uq_graph_entities_canonical_name_collection_id"
                        )
                        .returning(GraphEntity.id)
                    )
                    is_new = result.scalar_one_or_none() is not None
                    entity = (
                        await writer.execute(
                            select(GraphEntity).where(
                                GraphEntity.collection_id == r._collection_id,
                                GraphEntity.canonical_name == incoming["name"],
                            )
                        )
                    ).scalar_one()
                await r._add_alias(
                    writer,
                    entity.id,
                    incoming["name"],
                    chunk_hash,
                    document_id=document_id,
                    document_path=document_path,
                )
                await r._add_or_increment_type(writer, entity.id, incoming["type"])
                return EntityResolutionResult(is_new, entity.id, entity.canonical_name)

            resolved = await persist_transaction(session.bind, persist_entity)
            results.append(resolved)
            provisional_results[incoming["provisional_id"]] = resolved
            self.entities.append(
                {
                    "entity_id": resolved.entity_id,
                    "canonical_name": resolved.canonical_name,
                    "description": incoming["description"],
                    "embedding": embeddings[index],
                    "embedding_name": incoming["name"],
                    "chunk_hash": chunk_hash,
                    "document_id": document_id,
                    "document_path": document_path,
                }
            )
        return results

    async def resolve_relationships(
        self,
        session: AsyncSession,
        relationships: Sequence[RelationshipInput],
        chunk_hash: str,
        document_id: uuid.UUID | None,
        document_path: str | None,
    ) -> list[RelationshipResolutionResult]:
        r = self.resolver
        await session.commit()
        results = []
        from graph_core.services.graph_rag.entity_resolver import (
            RelationshipResolutionResult,
        )

        for incoming in relationships:

            async def persist_relationship(writer):
                kind = await r._resolve_rel_type(writer, incoming["rel_type"])
                # Serialize create/find by semantic key; never infer under this lock.
                lock_id = uuid.uuid5(
                    r._collection_id,
                    f"relationship:{incoming['source_entity_id']}:"
                    f"{incoming['target_entity_id']}:{kind.relationship_type_id}",
                )
                await r._acquire_entity_lock(writer, lock_id)
                rel = (
                    await writer.execute(
                        select(GraphRelationship).where(
                            GraphRelationship.source_entity_id
                            == incoming["source_entity_id"],
                            GraphRelationship.target_entity_id
                            == incoming["target_entity_id"],
                            GraphRelationship.relationship_type_id
                            == kind.relationship_type_id,
                        )
                    )
                ).scalar_one_or_none()
                is_new = rel is None
                if rel is None:
                    rel = GraphRelationship(
                        id=uuid.uuid4(),
                        collection_id=r._collection_id,
                        source_entity_id=incoming["source_entity_id"],
                        target_entity_id=incoming["target_entity_id"],
                        relationship_type_id=kind.relationship_type_id,
                        rel_type=kind.canonical_type,
                        weight=0,
                        keywords=[],
                    )
                    writer.add(rel)
                source = await writer.get(GraphEntity, incoming["source_entity_id"])
                target = await writer.get(GraphEntity, incoming["target_entity_id"])
                if source is None or target is None:
                    raise RuntimeError(
                        "Relationship endpoint disappeared; retry the chunk"
                    )
                work = {
                    **incoming,
                    "relationship_id": rel.id,
                    "source_name": source.canonical_name,
                    "target_name": target.canonical_name,
                    "rel_type": kind.canonical_type,
                    "chunk_hash": chunk_hash,
                    "document_id": document_id,
                    "document_path": document_path,
                }
                return work, is_new

            work, is_new = await persist_transaction(session.bind, persist_relationship)
            self.relationships.append(work)
            results.append(
                RelationshipResolutionResult(is_new, work["relationship_id"])
            )
        return results

    async def flush(self, session: AsyncSession) -> None:
        """Score entity descriptions and all relationship assessments jointly.

        Recheck evidence under the write lock. A concurrent new passage triggers a
        fresh snapshot and re-score, never a score attached to unseen evidence.
        Successfully applied groups are removed before retrying changed groups.
        """
        await session.commit()
        for attempt in range(3):
            plans, questions = await self._support_plans(session.bind)
            if not plans:
                self.entities.clear()
                self.relationships.clear()
                return
            decisions = (
                await self.resolver._decisions.support_many(questions)
                if questions
                else {}
            )
            await self._embed_changes(plans, decisions)
            remaining_entities, remaining_relationships = [], []
            for plan in plans:
                r = self.resolver

                async def persist_support(writer):
                    await r._acquire_entity_lock(writer, plan["id"])
                    model = (
                        EntityDescription
                        if plan["kind"] == "entity"
                        else RelationshipDescription
                    )
                    owner = (
                        model.entity_id
                        if plan["kind"] == "entity"
                        else model.relationship_id
                    )
                    rows = (
                        (await writer.execute(select(model).where(owner == plan["id"])))
                        .scalars()
                        .all()
                    )
                    if snapshot(rows) != plan["snapshot"]:
                        return False
                    await self._apply_support(writer, plan, rows, decisions)
                    return True

                if not await persist_transaction(session.bind, persist_support):
                    if plan["kind"] == "entity":
                        remaining_entities.extend(plan["work"])
                    else:
                        remaining_relationships.extend(plan["work"])
            self.entities, self.relationships = (
                remaining_entities,
                remaining_relationships,
            )
            if not self.entities and not self.relationships:
                return
        raise RuntimeError(
            "Source evidence kept changing while scoring; retry the chunk"
        )

    async def _embed_changes(self, plans, decisions):
        """Reuse entity vectors; batch remaining embeddings outside locks."""
        from graph_core.models.rel_types import relationship_embedding_text

        targets, texts = [], []
        for plan in plans:
            for change in plan["changes"]:
                if not change["new"] or "embedding" in change:
                    continue
                if plan["kind"] == "entity":
                    if decisions[change["question_id"]].choice != "supported":
                        continue
                    text = f"{plan['canonical_name']}: {change['description']}"
                else:
                    text = relationship_embedding_text(
                        plan["source_name"],
                        plan["target_name"],
                        plan["rel_type"],
                        change["description"],
                        change["keywords"],
                    )
                targets.append(change)
                texts.append(text)
        if not texts:
            return
        provider = self.resolver._embedding
        embeddings = (
            await provider.embed_documents(texts)
            if hasattr(provider, "embed_documents")
            else [await provider.embed_query(text) for text in texts]
        )
        if len(embeddings) != len(targets):
            raise RuntimeError(
                "Embedding provider returned the wrong number of vectors"
            )
        for change, embedding in zip(targets, embeddings):
            change["embedding"] = embedding

    async def _support_plans(self, bind):
        plans, questions = [], []
        groups = defaultdict(list)
        for work in self.entities:
            groups[("entity", work["entity_id"])].append(work)
        for work in self.relationships:
            groups[("relationship", work["relationship_id"])].append(work)
        async with AsyncSessionLocal(bind=bind) as reader:
            for (kind, owner_id), work_items in groups.items():
                model = (
                    EntityDescription if kind == "entity" else RelationshipDescription
                )
                owner = model.entity_id if kind == "entity" else model.relationship_id
                rows = (
                    (await reader.execute(select(model).where(owner == owner_id)))
                    .scalars()
                    .all()
                )
                states = {
                    (row.document_id, row.description): {
                        "row_id": row.id,
                        "description": row.description,
                        "document_id": row.document_id,
                        "document_path": row.document_path,
                        "evidence": list(row.source_evidence or []),
                        "keywords": getattr(row, "keywords", []) or [],
                        "new": False,
                    }
                    for row in rows
                }
                changed = set()
                plan = {
                    "kind": kind,
                    "id": owner_id,
                    "snapshot": snapshot(rows),
                    "work": work_items,
                    "changes": [],
                    **{
                        key: value
                        for key, value in work_items[0].items()
                        if key
                        in {"canonical_name", "source_name", "target_name", "rel_type"}
                    },
                }
                claim = (
                    {"entity": plan["canonical_name"]}
                    if kind == "entity"
                    else {
                        "source": plan["source_name"],
                        "target": plan["target_name"],
                        "predicate": plan["rel_type"],
                    }
                )
                for work in work_items:
                    if kind == "entity" and not work["description"]:
                        continue
                    key = (work["document_id"], work["description"])
                    state = states.setdefault(
                        key,
                        {
                            "row_id": uuid.uuid4(),
                            "description": work["description"],
                            "document_id": work["document_id"],
                            "document_path": work["document_path"],
                            "evidence": [],
                            "keywords": work.get("keywords", []),
                            "new": True,
                        },
                    )
                    if (
                        kind == "entity"
                        and work["embedding_name"] == plan["canonical_name"]
                    ):
                        state["embedding"] = work["embedding"]
                    evidence = {
                        "chunk_hash": work["chunk_hash"],
                        "document_id": str(work["document_id"])
                        if work["document_id"]
                        else None,
                        "document_path": work["document_path"],
                        "source_passage": self.resolver._source_text,
                    }
                    if kind == "relationship":
                        evidence.update(
                            original_source_name=work.get("original_source_name")
                            or plan["source_name"],
                            original_target_name=work.get("original_target_name")
                            or plan["target_name"],
                            extracted_description=work["description"],
                        )
                    if any(
                        evidence_key(item) == evidence_key(evidence)
                        for item in state["evidence"]
                    ):
                        continue
                    changed.add(key)
                    if kind == "relationship":
                        question_id = f"support_{len(questions)}"
                        questions.append(
                            {
                                "id": question_id,
                                "claim": {**claim, "description": work["description"]},
                                "evidence": [evidence],
                            }
                        )
                        # Remove this internal pointer before submitting evidence.
                        evidence["_passage_question_id"] = question_id
                    state["evidence"].append(evidence)
                for key in changed:
                    state = states[key]
                    question_id = (
                        state["evidence"][0]["_passage_question_id"]
                        if kind == "relationship" and len(state["evidence"]) == 1
                        else f"support_{len(questions)}"
                    )
                    if kind == "entity" or len(state["evidence"]) > 1:
                        questions.append(
                            {
                                "id": question_id,
                                "claim": {**claim, "description": state["description"]},
                                "evidence": state["evidence"],
                            }
                        )
                    plan["changes"].append({**state, "question_id": question_id})
                if kind == "relationship":
                    rel = await reader.get(GraphRelationship, owner_id)
                    if changed or rel.confidence is None:
                        unique = {
                            evidence_key(item): item
                            for state in states.values()
                            for item in state["evidence"]
                        }
                        plan["question_id"] = f"support_{len(questions)}"
                        questions.append(
                            {
                                "id": plan["question_id"],
                                "claim": claim,
                                "evidence": list(unique.values()),
                            }
                        )
                if plan["changes"] or "question_id" in plan:
                    plans.append(plan)
        # Evidence snapshots have no model scores (or internal question pointers).
        for question in questions:
            question["evidence"] = [
                {
                    key: value
                    for key, value in item.items()
                    if key != "_passage_question_id"
                }
                for item in question["evidence"]
            ]
        return plans, questions

    async def _apply_support(self, writer, plan, rows, decisions):
        r = self.resolver
        by_id = {row.id: row for row in rows}
        for change in plan["changes"]:
            decision = decisions[change["question_id"]]
            evidence = []
            for original in change["evidence"]:
                item = dict(original)
                passage_id = item.pop("_passage_question_id", None)
                if passage_id is not None:
                    passage = decisions[passage_id]
                    item.update(
                        support_confidence=passage.probabilities["supported"],
                        support_assessment=passage.trace(
                            "relationship_passage_support"
                        ),
                    )
                evidence.append(item)
            model = (
                EntityDescription
                if plan["kind"] == "entity"
                else RelationshipDescription
            )
            row = by_id.get(change["row_id"])
            if row is None:
                owner_key = (
                    "entity_id" if plan["kind"] == "entity" else "relationship_id"
                )
                row = model(
                    id=change["row_id"],
                    **{owner_key: plan["id"]},
                    description=change["description"],
                    document_id=change["document_id"],
                    document_path=change["document_path"],
                )
                writer.add(row)
                rows.append(row)
            row.source_evidence = evidence
            row.source_chunk_hashes = sorted({item["chunk_hash"] for item in evidence})
            row.weight = len(evidence)
            row.confidence = decision.probabilities["supported"]
            row.score_metadata = decision.trace(f"{plan['kind']}_description_support")
            if plan["kind"] == "relationship":
                row.keywords = change["keywords"]
                if change["new"]:
                    await r._vstore.upsert_relationship_embedding(
                        relationship_id=plan["id"],
                        collection_id=r._collection_id,
                        source_name=plan["source_name"],
                        target_name=plan["target_name"],
                        description=change["description"],
                        embedding=change["embedding"],
                        document_id=change["document_id"],
                        document_path=change["document_path"],
                        session=writer,
                    )
            elif change["new"] and decision.choice == "supported":
                entity = await writer.get(GraphEntity, plan["id"])
                if entity is None:
                    raise RuntimeError("Entity disappeared while scoring")
                count = entity.description_count or 0
                old = await r._vstore.get_entity_centroid(
                    entity.id, r._collection_id, session=writer
                )
                embedding = change["embedding"]
                centroid = (
                    [(a * count + b) / (count + 1) for a, b in zip(old, embedding)]
                    if old
                    else embedding
                )
                await writer.flush()
                await r._vstore.upsert_entity_embedding(
                    entity_id=entity.id,
                    collection_id=r._collection_id,
                    name=entity.canonical_name,
                    description=change["description"],
                    description_id=row.id,
                    embedding=embedding,
                    document_id=change["document_id"],
                    document_path=change["document_path"],
                    session=writer,
                )
                await r._vstore.upsert_entity_centroid(
                    entity_id=entity.id,
                    collection_id=r._collection_id,
                    canonical_name=entity.canonical_name,
                    primary_type=entity.primary_type,
                    description_count=count + 1,
                    embedding=centroid,
                    session=writer,
                )
                entity.description_count = count + 1
        if plan["kind"] == "relationship":
            unique = {
                evidence_key(item): item
                for row in rows
                for item in row.source_evidence or []
            }
            rel = await writer.get(GraphRelationship, plan["id"])
            if rel is None:
                raise RuntimeError("Relationship disappeared while scoring")
            decision = decisions[plan["question_id"]]
            rel.confidence = decision.probabilities["supported"]
            supported = {
                evidence_key(item)
                for row in rows
                for item in row.source_evidence or []
                if (item.get("support_assessment") or {}).get("choice") == "supported"
            }
            rel.support_count = len(supported)
            rel.score_metadata = {
                **decision.trace("relationship_support"),
                "claim": {
                    "source": plan["source_name"],
                    "target": plan["target_name"],
                    "predicate": plan["rel_type"],
                },
                "evidence_keys": [list(key) for key in unique],
            }
            rel.weight = round(rel.confidence * 100)
            rel.keywords = sorted(
                set(rel.keywords or [])
                | {keyword for item in plan["work"] for keyword in item["keywords"]}
            )

"""Final passage-level selection by the decision model, against the original question."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from graph_core.database import AsyncSessionLocal
from graph_core.decisions import GraphDecisions
from graph_core.decisions.graph import relevance_score, relevant
from graph_core.models.collection import Collection
from graph_core.models.graph_rag import (
    EntityDescription,
    GraphEntity,
    GraphRelationship,
    RelationshipDescription,
)

if TYPE_CHECKING:
    from graph_core.services.graph.query.graph_rag import GraphQueryState


async def build_context(
    state: GraphQueryState,
    collection: Collection,
    question: str,
    *,
    derived_context: str = "",
    document_ids: list[uuid.UUID] | None = None,
) -> tuple[str, list[str], list[str], str]:
    if not question.strip():
        raise ValueError("Decision-model context ranking requires the original question")
    candidates: list[dict[str, Any]] = []
    descriptions_by_id: dict[str, str] = {}
    async with AsyncSessionLocal() as session:
        ranked_entities = sorted(
            state.discovered_entity_ids,
            key=lambda eid: state.entity_relevance.get(eid, 0),
            reverse=True,
        )
        for eid in ranked_entities[:30]:
            try:
                entity_id = uuid.UUID(eid)
            except ValueError:
                continue
            entity = await session.get(GraphEntity, entity_id)
            if entity is None or entity.collection_id != collection.id:
                continue
            conditions = [
                EntityDescription.entity_id == entity.id,
                EntityDescription.score_metadata["choice"].as_string() == "supported",
            ]
            if document_ids:
                conditions.append(EntityDescription.document_id.in_(document_ids))
            rows = (
                (
                    await session.execute(
                        select(EntityDescription)
                        .where(*conditions)
                        .order_by(EntityDescription.weight.desc())
                        .limit(4)
                    )
                )
                .scalars()
                .all()
            )
            # Score individual descriptions. A single relevant passage must not
            # license inclusion of unrelated/contaminated neighboring passages.
            for row in rows:
                key = "e_" + row.id.hex
                candidates.append(
                    {
                        "id": key,
                        "graph_id": str(entity.id),
                        "kind": "entity",
                        "name": entity.canonical_name,
                        "type": entity.primary_type,
                        "descriptions": [row.description],
                        "description_id": str(row.id),
                        "document_id": str(row.document_id)
                        if row.document_id
                        else None,
                        "source_support_confidence": row.confidence,
                    }
                )
                descriptions_by_id[key] = (
                    f"{entity.canonical_name} ({entity.primary_type or 'unknown'}): {row.description}"
                )
        for rid in state.traversed_rel_ids[:80]:
            try:
                relationship_id = uuid.UUID(rid)
            except ValueError:
                continue
            rel = await session.get(GraphRelationship, relationship_id)
            if rel is None or rel.collection_id != collection.id:
                continue
            if (rel.score_metadata or {}).get("choice") != "supported":
                continue
            src = await session.get(GraphEntity, rel.source_entity_id)
            tgt = await session.get(GraphEntity, rel.target_entity_id)
            if src is None or tgt is None:
                continue
            conditions = [
                RelationshipDescription.relationship_id == rel.id,
                RelationshipDescription.score_metadata["choice"].as_string() == "supported",
            ]
            if document_ids:
                conditions.append(RelationshipDescription.document_id.in_(document_ids))
            rows = (
                (
                    await session.execute(
                        select(RelationshipDescription)
                        .where(*conditions)
                        .order_by(RelationshipDescription.weight.desc())
                        .limit(4)
                    )
                )
                .scalars()
                .all()
            )
            label = f"{src.canonical_name} -[{rel.rel_type}]-> {tgt.canonical_name}"
            for row in rows:
                key = "r_" + row.id.hex
                candidates.append(
                    {
                        "id": key,
                        "graph_id": str(rel.id),
                        "kind": "relationship",
                        "name": label,
                        "descriptions": [row.description],
                        "description_id": str(row.id),
                        "document_id": str(row.document_id)
                        if row.document_id
                        else None,
                        "source_support_confidence": row.confidence,
                        "source_entity_id": str(src.id),
                        "target_entity_id": str(tgt.id),
                    }
                )
                descriptions_by_id[key] = f"{label}: {row.description}"
    if derived_context:
        candidates.append(
            {
                "id": "derived",
                "graph_id": "derived",
                "kind": "derived",
                "name": "Derived understanding",
                "descriptions": [derived_context],
            }
        )
        descriptions_by_id["derived"] = derived_context
    decisions = await GraphDecisions().relevance(question, candidates)
    ranked = sorted(
        (c for c in candidates if relevant(decisions[c["id"]])),
        key=lambda c: relevance_score(decisions[c["id"]]),
        reverse=True,
    )

    def select_groups(
        kind: str, limit: int
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        selected = []
        scores = {}
        for candidate in ranked:
            if candidate["kind"] != kind:
                continue
            graph_id = candidate["graph_id"]
            if graph_id not in scores and len(scores) >= limit:
                continue
            scores[graph_id] = max(
                scores.get(graph_id, 0), relevance_score(decisions[candidate["id"]])
            )
            selected.append(candidate)
        return selected, scores

    entity_candidates, entity_scores = select_groups("entity", 10)
    rel_candidates, relationship_scores = select_groups("relationship", 20)
    derived = [c for c in ranked if c["kind"] == "derived"]
    included_ids = {c["id"] for c in entity_candidates + rel_candidates + derived}
    state.relevance_decisions = [
        {
            "id": c["id"],
            "kind": c["kind"],
            "name": c["name"],
            "graph_id": c["graph_id"],
            "description_id": c.get("description_id"),
            "document_id": c.get("document_id"),
            "source_confidence": c.get("source_support_confidence"),
            "evaluation_question": question,
            "included": c["id"] in included_ids,
            **decisions[c["id"]].trace("query_relevance"),
        }
        for c in candidates
    ]
    # Downstream route profiles/projections may use only native accepted scores,
    # not embedding similarities or fabricated anchor relevance.
    state.discovered_entity_ids = set(entity_scores)
    for candidate in rel_candidates:
        state.discovered_entity_ids.update(
            [candidate["source_entity_id"], candidate["target_entity_id"]]
        )
    state.entity_relevance = entity_scores
    state.traversed_rel_ids = list(relationship_scores)
    state.rel_score_cache = relationship_scores
    state.rel_combined_score_cache = dict(relationship_scores)
    entities = list(dict.fromkeys(c["name"] for c in entity_candidates))
    relationships = list(dict.fromkeys(c["name"] for c in rel_candidates))
    entity_text = "\n".join(descriptions_by_id[c["id"]] for c in entity_candidates)
    relationship_text = "\n\n".join(descriptions_by_id[c["id"]] for c in rel_candidates)
    context = "Context:\n"
    if derived:
        context += "Derived Understanding:\n" + derived_context + "\n"
    context += f"Entities:\n{entity_text or '(none)'}\nRelationships:\n{relationship_text or '(none)'}"
    return context, entities, relationships, relationship_text

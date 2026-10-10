"""Staged query selection: nodes, top-40 edge ranking, then binary edge gate."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from graph_core.database import AsyncSessionLocal
from graph_core.decisions import GraphDecisions
from graph_core.decisions.graph import QUERY_MAX_EDGES, passes_query_gate
from graph_core.models.collection import Collection
from graph_core.models.graph_rag import (
    EntityDescription,
    GraphEntity,
    GraphRelationship,
    RelationshipDescription,
)

if TYPE_CHECKING:
    from graph_core.services.graph.query.graph_rag import GraphQueryState


async def load_entity_candidate(session, entity, document_ids=None):
    """Hydrate actual extracted descriptions, never substitute a matching label."""
    conditions = [EntityDescription.entity_id == entity.id]
    if document_ids:
        conditions.append(EntityDescription.document_id.in_(document_ids))
    rows = (
        (
            await session.execute(
                select(EntityDescription)
                .where(*conditions)
                .order_by(EntityDescription.weight.desc(), EntityDescription.id)
                .limit(1)
            )
        )
        .scalars()
        .all()
    )
    return {
        "id": "e_" + entity.id.hex,
        "graph_id": str(entity.id),
        "kind": "entity",
        "name": entity.canonical_name,
        "type": entity.primary_type,
        "descriptions": [row.description for row in rows],
        "provenance": [
            {
                "description_id": str(row.id),
                "document_id": str(row.document_id) if row.document_id else None,
            }
            for row in rows
        ],
    }


async def build_context(
    state: GraphQueryState,
    collection: Collection,
    question: str,
    *,
    derived_context: str = "",
    document_ids: list[uuid.UUID] | None = None,
) -> tuple[str, list[str], list[str], str]:
    if not question.strip():
        raise ValueError(
            "Decision-model context ranking requires the original question"
        )
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    async with AsyncSessionLocal() as session:
        node_ids = (
            state.anchor_entity_ids
            if state.node_gate_complete
            else state.discovered_entity_ids
        )
        for eid in sorted(node_ids):
            try:
                entity_id = uuid.UUID(eid)
            except ValueError:
                continue
            entity = await session.get(GraphEntity, entity_id)
            if entity is None or entity.collection_id != collection.id:
                continue
            candidate = await load_entity_candidate(session, entity, document_ids)
            if candidate["descriptions"]:
                nodes.append(candidate)

        for rid in dict.fromkeys(state.traversed_rel_ids):
            try:
                relationship_id = uuid.UUID(rid)
            except ValueError:
                continue
            rel = await session.get(GraphRelationship, relationship_id)
            if rel is None or rel.collection_id != collection.id:
                continue
            src = await session.get(GraphEntity, rel.source_entity_id)
            tgt = await session.get(GraphEntity, rel.target_entity_id)
            if (
                src is None
                or tgt is None
                or src.collection_id != collection.id
                or tgt.collection_id != collection.id
            ):
                continue
            conditions = [RelationshipDescription.relationship_id == rel.id]
            if document_ids:
                conditions.append(RelationshipDescription.document_id.in_(document_ids))
            row = (
                (
                    await session.execute(
                        select(RelationshipDescription)
                        .where(*conditions)
                        .order_by(
                            RelationshipDescription.weight.desc(),
                            RelationshipDescription.id,
                        )
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )
            if row is None:
                continue
            edges.append(
                {
                    "id": "r_" + rel.id.hex,
                    "graph_id": str(rel.id),
                    "kind": "relationship",
                    "name": f"{src.canonical_name} -[{rel.rel_type}]-> {tgt.canonical_name}",
                    "source": src.canonical_name,
                    "relationship": rel.rel_type,
                    "target": tgt.canonical_name,
                    "description": row.description,
                    "description_id": str(row.id),
                    "document_id": str(row.document_id) if row.document_id else None,
                    "source_confidence": row.confidence,
                    "source_entity_id": str(src.id),
                    "target_entity_id": str(tgt.id),
                }
            )

    scorer = GraphDecisions()
    traces = list(state.relevance_decisions)
    if not state.node_gate_complete:
        node_decisions = await scorer.entity_relevance(question, nodes)
        for candidate in nodes:
            decision = node_decisions[candidate["id"]]
            traces.append(
                {
                    **candidate,
                    "evaluation_question": question,
                    "included": passes_query_gate(decision),
                    **decision.trace("query_nodes"),
                }
            )
        nodes = [c for c in nodes if passes_query_gate(node_decisions[c["id"]])]
        state.entity_relevance = {
            c["graph_id"]: node_decisions[c["id"]].probabilities["true"] for c in nodes
        }
        state.anchor_entity_ids = {c["graph_id"] for c in nodes}
        state.node_gate_complete = True

    anchor_names = {c["graph_id"]: c["name"] for c in nodes}
    for edge in edges:
        edge["connected_anchors"] = [
            anchor_names[eid]
            for eid in (edge["source_entity_id"], edge["target_entity_id"])
            if eid in anchor_names
        ]
    scores = await scorer.edge_scores(question, edges)
    ranked = sorted(edges, key=lambda c: (-scores[c["id"]].score, c["graph_id"]))
    top = ranked[:QUERY_MAX_EDGES]
    gates = await scorer.edge_relevance(question, top)
    retained = [c for c in top if passes_query_gate(gates[c["id"]])]
    included = {c["id"] for c in retained}
    shortlisted = {c["id"] for c in top}
    for candidate in ranked:
        traces.append(
            {
                **candidate,
                "evaluation_question": question,
                "shortlisted": candidate["id"] in shortlisted,
                "included": candidate["id"] in included,
                **scores[candidate["id"]].trace("query_edge_scores"),
            }
        )
    for candidate in top:
        traces.append(
            {
                **candidate,
                "evaluation_question": question,
                "included": candidate["id"] in included,
                **gates[candidate["id"]].trace("query_edge_gate"),
            }
        )
    state.relevance_decisions = traces
    state.discovered_entity_ids = {c["graph_id"] for c in nodes}
    state.entity_relevance = {
        eid: state.entity_relevance.get(eid, 0.0) for eid in state.discovered_entity_ids
    }
    for edge in retained:
        for eid in (edge["source_entity_id"], edge["target_entity_id"]):
            state.discovered_entity_ids.add(eid)
            state.entity_relevance[eid] = max(
                state.entity_relevance.get(eid, 0.0),
                gates[edge["id"]].probabilities["true"],
            )
    state.traversed_rel_ids = [c["graph_id"] for c in retained]
    state.rel_score_cache = {c["graph_id"]: scores[c["id"]].score for c in retained}
    state.rel_combined_score_cache = dict(state.rel_score_cache)
    entities = list(dict.fromkeys(c["name"] for c in nodes))
    relationships = [c["name"] for c in retained]
    entity_text = "\n".join(
        f"{c['name']} ({c['type'] or 'unknown'}): " + " | ".join(c["descriptions"])
        for c in nodes
    )
    relationship_text = "\n\n".join(
        f"{c['name']}: {c['description']}" for c in retained
    )
    context = "Context:\n"
    if derived_context:
        context += "Derived Understanding:\n" + derived_context + "\n"
    context += f"Entities:\n{entity_text or '(none)'}\nRelationships:\n{relationship_text or '(none)'}"
    return context, entities, relationships, relationship_text

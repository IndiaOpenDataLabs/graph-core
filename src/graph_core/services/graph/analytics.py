"""Offline analytics over the canonical collection graph.

Builds role-similarity candidate regions over the base graph and induces
semantic concepts from them via LLM or deterministic fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import aliased

from graph_core.database import AsyncSessionLocal
from graph_core.llm import LocalEchoLLMProvider
from graph_core.llm.interface import LLMProvider
from graph_core.models.collection import Collection
from graph_core.models.graph_rag import (
    EntityAlias,
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
)
from graph_core.models.rel_types import rel_types_for_domain


@dataclass(slots=True)
class NodeRecord:
    id: uuid.UUID
    name: str


@dataclass(slots=True)
class RelationshipRecord:
    id: uuid.UUID
    source_id: uuid.UUID
    source_name: str
    target_id: uuid.UUID
    target_name: str
    rel_type: str
    weight: int


_CODE_REL_TYPES = {value.upper() for value in rel_types_for_domain("code")}
_NON_CODE_REL_TYPES = {
    value.upper()
    for domain in ("general", "books", "personal")
    for value in rel_types_for_domain(domain)
}
_CODE_ONLY_REL_TYPES = _CODE_REL_TYPES - _NON_CODE_REL_TYPES


def _is_code_like_collection(relationship_records: list[dict[str, Any]]) -> bool:
    rel_types = {
        str(rel.get("rel_type") or "").upper()
        for rel in relationship_records
        if str(rel.get("rel_type") or "").strip()
    }
    if not rel_types:
        return False
    if rel_types & _CODE_ONLY_REL_TYPES:
        return True
    code_like_common = {
        "DEFINES",
        "DEPENDS_ON",
        "READS",
        "WRITES",
        "RETURNS",
        "LOOPS_OVER",
        "GUARDS",
        "DECORATES",
        "RAISES",
        "CATCHES",
    }
    return len(rel_types & code_like_common) >= 3


def _code_concept_prompt_guidance() -> str:
    return (
        "This is a code collection. Write the concept description in compact "
        "pseudo-code-flavored prose that preserves execution semantics. Capture "
        "conditions, decisions, ordering, loops, splits, merges, or retries when "
        "the evidence supports them. Conditions may be English, but the wording "
        "should feel code-like, using cues such as IF, WHEN, THEN, ELSE, FOR EACH, "
        "WHILE, TRY/CATCH, RETURNS, SPLITS INTO, or MERGES WITH. Do not invent "
        "control flow beyond the evidence."
    )


def _format_connects_to_description(
    *,
    source_label: str,
    source_desc: str,
    target_label: str,
    target_desc: str,
    top_rel_types: list[str],
    direct_count: int,
    path_count: int,
    path_score: float,
    boundary_names: list[str],
    bridge_names: list[str],
    code_like: bool,
) -> str:
    if not code_like:
        description_parts = [
            f"{source_label} ({source_desc}) connects to {target_label} ({target_desc}) "
            f"through {direct_count} direct cross-edge(s) dominated by {', '.join(top_rel_types)}"
        ]
        if path_count:
            description_parts.append(
                f"and {path_count} short directed path(s) with cumulative path score {path_score}"
            )
        if boundary_names:
            description_parts.append(
                f"boundary entities: {', '.join(boundary_names)}"
            )
        if bridge_names:
            description_parts.append(
                f"bridge entities: {', '.join(bridge_names)}"
            )
        return "; ".join(description_parts).strip() + "."

    steps = [
        (
            f"WHEN {source_label} is active, THEN flow reaches {target_label} "
            f"mainly via {', '.join(top_rel_types)}; direct_cross_edges={direct_count}"
        )
    ]
    if source_desc or target_desc:
        steps.append(
            f"CONTEXT: {source_label}={source_desc or 'n/a'}; {target_label}={target_desc or 'n/a'}"
        )
    if boundary_names:
        steps.append(f"USING boundary_entities=[{', '.join(boundary_names)}]")
    if path_count:
        steps.append(
            f"FOR short_paths in range({path_count}): cumulative_path_score={path_score}"
        )
    if bridge_names:
        steps.append(f"BRIDGE VIA [{', '.join(bridge_names)}]")
    return "; ".join(steps).strip() + "."

def _build_role_similarity_groups(
    nodes: list[NodeRecord],
    relationships: list[RelationshipRecord],
    *,
    overlap_min: int = 2,
    cosine_min: float = 0.0,
    jaccard_min: float = 0.0,
    min_signature: int = 1,
) -> list[dict[str, Any]]:
    if not relationships:
        return []

    out_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    in_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    node_name_by_id = {str(node.id): node.name for node in nodes}
    relationship_records_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_node_ids: set[str] = set()

    for rel in relationships:
        source_id = str(rel.source_id)
        target_id = str(rel.target_id)
        rel_type = str(rel.rel_type or "RELATES_TO").upper()
        all_node_ids.add(source_id)
        all_node_ids.add(target_id)
        out_pairs[source_id].add((rel_type, target_id))
        in_pairs[target_id].add((source_id, rel_type))
        relationship_records_by_node[source_id].append(
            {
                "source_id": source_id,
                "source_name": rel.source_name,
                "target_id": target_id,
                "target_name": rel.target_name,
                "rel_type": rel_type,
                "weight": float(rel.weight or 0.0),
                "direction": "out",
                "relationship_id": str(rel.id),
            }
        )
        relationship_records_by_node[target_id].append(
            {
                "source_id": source_id,
                "source_name": rel.source_name,
                "target_id": target_id,
                "target_name": rel.target_name,
                "rel_type": rel_type,
                "weight": float(rel.weight or 0.0),
                "direction": "in",
                "relationship_id": str(rel.id),
            }
        )

    signatures: dict[str, set[tuple[str, str]]] = {}
    token_index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for node_id in all_node_ids:
        signature = out_pairs[node_id] | in_pairs[node_id]
        if len(signature) < min_signature:
            continue
        signatures[node_id] = signature
        for token in signature:
            token_index[token].add(node_id)

    overlap_counts: Counter[tuple[str, str]] = Counter()
    for nodes_with_token in token_index.values():
        members = sorted(nodes_with_token)
        for left, right in combinations(members, 2):
            overlap_counts[(left, right)] += 1

    similarity_graph = nx.Graph()
    similarity_graph.add_nodes_from(signatures.keys())
    pair_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for (left, right), overlap in overlap_counts.items():
        if overlap < overlap_min:
            continue
        left_signature = signatures[left]
        right_signature = signatures[right]
        union = left_signature | right_signature
        cosine = overlap / math.sqrt(len(left_signature) * len(right_signature))
        jaccard = overlap / len(union) if union else 0.0
        if cosine < cosine_min or jaccard < jaccard_min:
            continue
        pair_metrics[(left, right)] = {
            "a": left,
            "b": right,
            "overlap": overlap,
            "cosine": round(cosine, 6),
            "jaccard": round(jaccard, 6),
            "size_a": len(left_signature),
            "size_b": len(right_signature),
        }
        similarity_graph.add_edge(
            left,
            right,
            overlap=overlap,
            cosine=cosine,
            jaccard=jaccard,
        )

    if similarity_graph.number_of_edges() == 0:
        return []

    ranked_groups: list[dict[str, Any]] = []
    for idx, clique in enumerate(nx.find_cliques(similarity_graph)):
        if len(clique) < 2:
            continue
        clique_ids = sorted(clique)
        metrics = []
        for left, right in combinations(clique_ids, 2):
            metrics.append(pair_metrics[tuple(sorted((left, right)))])
        avg_cosine = sum(float(metric["cosine"]) for metric in metrics) / len(metrics)
        avg_jaccard = sum(float(metric["jaccard"]) for metric in metrics) / len(metrics)
        total_overlap = sum(int(metric["overlap"]) for metric in metrics)
        node_names = [node_name_by_id.get(node_id, node_id) for node_id in clique_ids]

        rel_counter: Counter[str] = Counter()
        representative_relationships: list[dict[str, Any]] = []
        seen_relationships: set[str] = set()
        for node_id in clique_ids:
            grouped_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for rel in relationship_records_by_node.get(node_id, []):
                grouped_by_type[str(rel["rel_type"])].append(rel)
                rel_counter[str(rel["rel_type"])] += 1
            for rel_type, items in sorted(
                grouped_by_type.items(),
                key=lambda kv: (-len(kv[1]), kv[0]),
            ):
                items = sorted(
                    items,
                    key=lambda item: (
                        -float(item["weight"]),
                        str(item["source_name"]),
                        str(item["target_name"]),
                    ),
                )
                rel = items[0]
                relationship_id = str(rel["relationship_id"])
                if relationship_id in seen_relationships:
                    continue
                seen_relationships.add(relationship_id)
                representative_relationships.append(rel)

        ranked_groups.append(
            {
                "group_id": f"role:{idx}",
                "kind": "role_clique",
                "size": len(clique_ids),
                "node_ids": clique_ids,
                "node_names": node_names,
                "avg_cosine": round(avg_cosine, 6),
                "avg_jaccard": round(avg_jaccard, 6),
                "total_overlap": total_overlap,
                "pair_metrics": metrics,
                "top_rel_types": [rel_type for rel_type, _ in rel_counter.most_common(8)],
                "representative_edges": representative_relationships[:24],
            }
        )

    ranked_groups.sort(
        key=lambda item: (
            int(item["size"]),
            float(item["avg_cosine"]),
            float(item["avg_jaccard"]),
            int(item["total_overlap"]),
        ),
        reverse=True,
    )
    return ranked_groups


def _shortest_paths_between_sets(
    adjacency: dict[uuid.UUID, list[tuple[uuid.UUID, str, float]]],
    starts: set[uuid.UUID],
    targets: set[uuid.UUID],
    *,
    max_depth: int = 4,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """BFS shortest paths between two node sets, used for concept-to-concept path analysis."""
    from collections import deque
    results: list[dict[str, Any]] = []
    for start in starts:
        queue: deque[tuple[uuid.UUID, list[uuid.UUID], list[str], float]] = deque(
            [(start, [start], [], 0.0)]
        )
        seen_depth: dict[uuid.UUID, int] = {start: 0}
        while queue:
            current, path_nodes, path_rels, score = queue.popleft()
            depth = len(path_rels)
            if depth >= max_depth:
                continue
            for neighbor, rel_type, weight in adjacency.get(current, []):
                next_nodes = [*path_nodes, neighbor]
                next_rels = [*path_rels, rel_type]
                next_score = score + float(weight)
                if neighbor in targets:
                    results.append(
                        {
                            "start": start,
                            "end": neighbor,
                            "nodes": next_nodes,
                            "rel_types": next_rels,
                            "hop_count": len(next_rels),
                            "path_score": round(next_score, 6),
                        }
                    )
                    continue
                next_depth = depth + 1
                if next_depth >= max_depth:
                    continue
                if neighbor not in seen_depth or next_depth < seen_depth[neighbor]:
                    seen_depth[neighbor] = next_depth
                    queue.append((neighbor, next_nodes, next_rels, next_score))
    results.sort(
        key=lambda item: (
            -float(item["path_score"]),
            int(item["hop_count"]),
            tuple(str(value) for value in item["rel_types"]),
            tuple(str(value) for value in item["nodes"]),
        )
    )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for result in results:
        key = tuple(str(value) for value in result["nodes"]) + tuple(
            str(value) for value in result["rel_types"]
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
        if len(deduped) >= limit:
            break
    return deduped


async def _load_graph_records(
    collection_id: uuid.UUID,
) -> tuple[
    Collection,
    list[NodeRecord],
    list[RelationshipRecord],
    dict[str, list[str]],
]:
    async with AsyncSessionLocal() as session:
        collection = await session.get(Collection, collection_id)
        if not collection:
            raise ValueError(f"Collection {collection_id} not found")

        nodes = (
            await session.execute(
                select(GraphEntity.id, GraphEntity.canonical_name).where(
                    GraphEntity.collection_id == collection_id
                )
            )
        ).all()

        source_entity = aliased(GraphEntity)
        target_entity = aliased(GraphEntity)
        relationships = (
            await session.execute(
                select(
                    GraphRelationship.id,
                    GraphRelationship.source_entity_id,
                    source_entity.canonical_name,
                    GraphRelationship.target_entity_id,
                    target_entity.canonical_name,
                    GraphRelationshipType.canonical_type,
                    GraphRelationship.weight,
                )
                .join(
                    source_entity,
                    source_entity.id == GraphRelationship.source_entity_id,
                )
                .join(
                    target_entity,
                    target_entity.id == GraphRelationship.target_entity_id,
                )
                .join(
                    GraphRelationshipType,
                    GraphRelationshipType.id == GraphRelationship.relationship_type_id,
                )
                .where(GraphRelationship.collection_id == collection_id)
            )
        ).all()

        alias_rows = (
            await session.execute(
                select(EntityAlias.entity_id, EntityAlias.alias_name).where(
                    EntityAlias.collection_id == collection_id
                )
            )
        ).all()

    aliases_by_entity_id: dict[str, list[str]] = defaultdict(list)
    for entity_id, alias_name in alias_rows:
        alias = str(alias_name or "").strip()
        if not alias:
            continue
        aliases_by_entity_id[str(entity_id)].append(alias)

    return (
        collection,
        [NodeRecord(id=node_id, name=name) for node_id, name in nodes],
        [
            RelationshipRecord(
                id=rel_id,
                source_id=source_id,
                source_name=source_name,
                target_id=target_id,
                target_name=target_name,
                rel_type=rel_type,
                weight=int(weight or 0),
            )
            for (
                rel_id,
                source_id,
                source_name,
                target_id,
                target_name,
                rel_type,
                weight,
            ) in relationships
        ],
        {
            entity_id: sorted({alias for alias in aliases})
            for entity_id, aliases in aliases_by_entity_id.items()
        },
    )


def build_collection_analysis(
    nodes: list[NodeRecord],
    relationships: list[RelationshipRecord],
) -> dict[str, Any]:
    role_groups = _build_role_similarity_groups(nodes, relationships)
    return {
        "totals": {
            "entities": len(nodes),
            "relationships": len(relationships),
            "role_groups": len(role_groups),
        },
        "role_groups": role_groups,
    }


async def analyze_collection_graph(
    collection_id: uuid.UUID,
) -> dict[str, Any]:
    collection, nodes, relationships, aliases_by_entity_id = await _load_graph_records(
        collection_id
    )
    analysis = build_collection_analysis(
        nodes,
        relationships,
    )
    analysis["collection"] = {
        "id": str(collection.id),
        "name": collection.name,
        "namespace_id": str(collection.namespace_id),
        "strategy": str(collection.strategy),
    }
    analysis["entity_aliases_by_id"] = aliases_by_entity_id
    analysis["relationship_records"] = [
        {
            "id": str(rel.id),
            "source_id": str(rel.source_id),
            "source_name": rel.source_name,
            "target_id": str(rel.target_id),
            "target_name": rel.target_name,
            "rel_type": rel.rel_type,
            "weight": rel.weight,
        }
        for rel in relationships
    ]
    return analysis


async def build_collection_understanding(
    analysis: dict[str, Any],
    llm_provider: LLMProvider | None = None,
) -> dict[str, list[dict[str, Any]]]:
    collection = analysis["collection"]
    max_deterministic_link_pairs = 64
    max_deterministic_meta_edges = 96
    entity_aliases_by_id: dict[str, list[str]] = dict(
        analysis.get("entity_aliases_by_id") or {}
    )
    relationship_records: list[dict[str, Any]] = list(
        analysis.get("relationship_records") or []
    )
    is_code_like = _is_code_like_collection(relationship_records)
    node_name_by_id: dict[str, str] = {}
    for rel in relationship_records:
        node_name_by_id[str(rel["source_id"])] = str(rel["source_name"])
        node_name_by_id[str(rel["target_id"])] = str(rel["target_name"])
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    seen_edge_ids: set[str] = set()
    seen_entity_refs: set[str] = set()

    def derived_chunk_hash(*parts: object) -> str:
        raw = "::".join(str(part) for part in parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def entity_ref_node(name: str) -> str:
        normalized = "_".join(name.strip().lower().split())
        return f"derived:entity:{normalized}"

    def ensure_entity_ref(
        name: str,
        *,
        supporting_ids: list[str] | None = None,
    ) -> str:
        node_id = entity_ref_node(name)
        if node_id in seen_entity_refs:
            return node_id
        seen_entity_refs.add(node_id)
        supporting_ids = supporting_ids or []
        aliases = sorted(
            {
                str(alias).strip()
                for source_id in supporting_ids
                for alias in entity_aliases_by_id.get(str(source_id), [])
                if str(alias).strip()
            }
        )
        nodes.append(
            {
                "id": node_id,
                "name": name,
                "canonical_name": name,
                "collection_id": collection["id"],
                "object_type": "entity",
                "primary_type": "base_entity_ref",
                "type": "base_entity_ref",
                "description": f"Reference to base graph entity: {name}.",
                "aliases": aliases,
                "source_ids": supporting_ids,
            }
        )
        return node_id

    chunk_index = 0

    def next_chunk_index() -> int:
        nonlocal chunk_index
        current = chunk_index
        chunk_index += 1
        return current

    def slug(text: str) -> str:
        return "_".join(part for part in text.strip().lower().split() if part)[:96]

    def add_edge(
        source_id: str,
        target_id: str,
        *,
        rel_type: str,
        description: str,
        source_ids: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> None:
        edge_id = f"{source_id}__{rel_type}__{target_id}"
        if edge_id in seen_edge_ids:
            return
        seen_edge_ids.add(edge_id)
        edges.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "id": edge_id,
                "collection_id": collection["id"],
                "object_type": "relationship",
                "rel_type": rel_type,
                "description": description,
                "keywords": keywords or [],
                "source_ids": source_ids or [],
            }
        )

    def normalize_map(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_value = max(scores.values()) or 0.0
        if max_value <= 0.0:
            return {key: 0.0 for key in scores}
        return {key: float(value) / float(max_value) for key, value in scores.items()}

    candidate_regions: list[dict[str, Any]] = []
    role_groups = list(analysis.get("role_groups") or [])
    for idx, group in enumerate(role_groups, start=1):
        if int(group.get("size", 0)) < 2:
            continue
        representative_edges = group.get("representative_edges", [])
        entity_names = list(group.get("node_names", []))[:24]
        rel_types = list(group.get("top_rel_types", []))
        pair_metrics = list(group.get("pair_metrics", []))
        candidate_regions.append(
            {
                "region_id": f"role_group_{idx}",
                "kind": "role_clique",
                "title": f"Role clique of size {group['size']}: {', '.join(entity_names[:6])}",
                "description": (
                    f"Role-similarity clique of size {group['size']} with average cosine "
                    f"{group['avg_cosine']} and average jaccard {group['avg_jaccard']}; "
                    f"total typed-signature overlap {group['total_overlap']}. "
                    f"Members: {', '.join(entity_names) or 'none'}. "
                    f"Dominant relation types: {', '.join(rel_types) or 'none'}."
                ),
                "source_ids": list(group.get("node_ids", [])),
                "entity_names": entity_names,
                "rel_types": rel_types,
                "representative_edges": representative_edges,
                "pair_metrics": pair_metrics,
            }
        )

    fallback_concepts = []
    for region in candidate_regions:
        fallback_concepts.append(
            {
                "label": region["title"],
                "concept_type": region["kind"],
                "description": region["description"],
                "aliases": [],
                "importance_reason": f"Derived from {region['kind']} candidate {region['region_id']}.",
                "evidence_region_ids": [region["region_id"]],
                "member_entity_names": region["entity_names"][:8],
            }
        )

    induced_concepts = list(fallback_concepts)
    if llm_provider and not isinstance(llm_provider, LocalEchoLLMProvider) and candidate_regions:
        concept_schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "concept_type": {"type": "string", "enum": ["theme", "process", "role", "pattern", "tension", "flow", "concept"]},
                "description": {"type": "string"},
                "aliases": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "importance_reason": {"type": "string"},
                "member_entity_names": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "label",
                "concept_type",
                "description",
                "aliases",
                "importance_reason",
                "member_entity_names",
            ],
        }

        async def induce_region_concept(region: dict[str, Any]) -> dict[str, Any]:
            rel_type = region["rel_types"][0] if region["rel_types"] else "RELATES_TO"
            directed_edges = (
                "; ".join(
                    (
                        f"{edge['source_name']} -[{rel_type}]-> "
                        f"{edge['target_name']} "
                        f"(weight={edge.get('weight', 0)}, "
                        f"count={edge.get('relationship_count', 1)})"
                    )
                    for edge in region.get("representative_edges", [])[:5]
                )
                or "none"
            )
            pair_metrics_text = (
                "; ".join(
                    (
                        f"{metric['a']}~{metric['b']} "
                        f"(overlap={metric['overlap']}, cosine={metric['cosine']}, "
                        f"jaccard={metric['jaccard']})"
                    )
                    for metric in region.get("pair_metrics", [])[:12]
                )
                or "none"
            )
            code_guidance = f"{_code_concept_prompt_guidance()}\n\n" if is_code_like else ""
            prompt = (
                "You are inducing one reusable semantic concept from a role-similarity clique in a knowledge graph.\n"
                "The member entities are grouped because they occupy similar typed graph positions.\n"
                "Do not return a mechanical label like cluster, graph region, connector, bridge, or clique.\n"
                "Infer the higher-level concept, role class, family, pattern, or shared abstraction that these members instantiate together.\n"
                "Prefer labels drawn directly from the collection's own terminology, especially source-language, tradition-specific, or text-native terms when they fit the evidence.\n"
                "If a well-established corpus term fits the pattern, use that term as the label instead of inventing a generic English abstraction.\n"
                "Only fall back to invented English labels when no source-grounded term is adequate.\n"
                "Avoid generic labels like force-user, principle, framework, pattern, agent, or mediator unless the evidence truly does not support a more native term.\n"
                "Use aliases to provide short alternate phrasings or English glosses when helpful, rather than putting the generic gloss in the primary label.\n\n"
                f"{code_guidance}"
                "Also provide a few short aliases or alternate phrasings for the concept when they would help later resolution.\n\n"
                f"Collection: {collection['name']}\n"
                f"Candidate id: {region['region_id']}\n"
                f"Candidate title: {region['title']}\n"
                f"Candidate description: {region['description']}\n"
                f"Entities: {', '.join(region['entity_names'][:16]) or 'none'}\n"
                f"Relation types: {', '.join(region['rel_types']) or 'none'}\n"
                f"Pairwise role-similarity evidence: {pair_metrics_text}\n"
                f"Representative neighborhood edges: {directed_edges}\n"
            )
            try:
                concept = await llm_provider.structured_extract(
                    prompt=prompt,
                    schema=concept_schema,
                )
                concept["evidence_region_ids"] = [region["region_id"]]
                return concept
            except Exception:
                return {
                    "label": region["title"],
                    "concept_type": region["kind"],
                    "description": region["description"],
                    "aliases": [],
                    "importance_reason": (
                        f"Fallback concept for relation type {rel_type} "
                        f"from candidate {region['region_id']}."
                    ),
                    "evidence_region_ids": [region["region_id"]],
                    "member_entity_names": region["entity_names"][:8],
                }

        induced_concepts = list(
            await asyncio.gather(
                *(induce_region_concept(region) for region in candidate_regions)
            )
        )
    region_lookup = {region["region_id"]: region for region in candidate_regions}
    concept_id_by_label: dict[str, str] = {}
    concept_source_ids: dict[str, list[str]] = {}
    concept_labels_by_id: dict[str, str] = {}
    concept_descriptions_by_id: dict[str, str] = {}
    concept_region_ids_by_id: dict[str, list[str]] = {}
    for concept in induced_concepts:
        label = str(concept.get("label") or "").strip()
        if not label:
            continue
        node_id = f"derived:concept:{slug(label)}"
        evidence_ids = [
            str(value)
            for value in concept.get("evidence_region_ids", [])
            if str(value).strip() in region_lookup
        ]
        source_ids = sorted(
            {
                source_id
                for region_id in evidence_ids
                for source_id in region_lookup[region_id]["source_ids"]
                if str(source_id).strip()
            }
        )
        description = (
            f"{str(concept.get('description') or '').strip()} "
            f"Why it matters: {str(concept.get('importance_reason') or '').strip()}"
        ).strip()
        concept_aliases = sorted(
            {
                str(value).strip()
                for value in concept.get("aliases", [])
                if str(value).strip()
            }
        )
        nodes.append(
            {
                "id": node_id,
                "name": label,
                "canonical_name": label,
                "collection_id": collection["id"],
                "object_type": "entity",
                "primary_type": str(concept.get("concept_type") or "concept"),
                "type": "derived_concept",
                "description": description,
                "aliases": concept_aliases,
                "source_ids": source_ids,
            }
        )
        chunk_content = description
        if concept_aliases:
            chunk_content = f"{description}\nAliases: {', '.join(concept_aliases)}".strip()
        chunks.append(
            {
                "chunk_hash": derived_chunk_hash(collection["id"], "concept", label),
                "chunk_index": next_chunk_index(),
                "content": chunk_content,
                "metadata": {
                    "memory_type": "derived_graph",
                    "derived_kind": "concept",
                    "derived_id": node_id,
                    "object_type": "entity",
                    "canonical_name": label,
                    "concept_type": str(concept.get("concept_type") or "concept"),
                    "aliases": concept_aliases,
                    "collection_id": collection["id"],
                },
            }
        )
        concept_id_by_label[label] = node_id
        concept_labels_by_id[node_id] = label
        concept_descriptions_by_id[node_id] = description
        concept_source_ids[node_id] = source_ids
        concept_region_ids_by_id[node_id] = evidence_ids
        member_names = [
            str(value).strip()
            for value in concept.get("member_entity_names", [])
            if str(value).strip()
        ]
        for name in member_names[:10]:
            ref_id = ensure_entity_ref(name, supporting_ids=source_ids)
            add_edge(
                node_id,
                ref_id,
                rel_type="EVIDENCED_BY",
                description=f"Concept {label} is evidenced by entity {name}.",
                source_ids=source_ids,
            )

    node_sets_by_concept: dict[str, set[str]] = {
        concept_id: set(source_ids)
        for concept_id, source_ids in concept_source_ids.items()
        if source_ids
    }
    memberships_by_node_id: dict[str, list[str]] = defaultdict(list)
    for concept_id, source_ids in node_sets_by_concept.items():
        for source_id in source_ids:
            memberships_by_node_id[source_id].append(concept_id)

    pair_aggregates: dict[tuple[str, str], dict[str, Any]] = {}

    def pair_bucket(source_id: str, target_id: str) -> dict[str, Any]:
        key = (source_id, target_id)
        bucket = pair_aggregates.get(key)
        if bucket is None:
            bucket = {
                "source_id": source_id,
                "target_id": target_id,
                "direct_count": 0,
                "direct_weight": 0.0,
                "rel_counter": Counter(),
                "boundary_counter": Counter(),
                "source_ids": set(),
            }
            pair_aggregates[key] = bucket
        return bucket

    for rel in relationship_records:
        source_members = memberships_by_node_id.get(str(rel["source_id"]), [])
        target_members = memberships_by_node_id.get(str(rel["target_id"]), [])
        if not source_members or not target_members:
            continue
        for source_concept_id in source_members:
            for target_concept_id in target_members:
                if source_concept_id == target_concept_id:
                    continue
                bucket = pair_bucket(source_concept_id, target_concept_id)
                weight = float(rel.get("weight") or 0.0)
                rel_type = str(rel.get("rel_type") or "RELATES_TO").upper()
                bucket["direct_count"] += 1
                bucket["direct_weight"] += weight
                bucket["rel_counter"][rel_type] += 1
                bucket["boundary_counter"][str(rel.get("source_name") or "")] += 1
                bucket["boundary_counter"][str(rel.get("target_name") or "")] += 1
                bucket["source_ids"].add(str(rel["source_id"]))
                bucket["source_ids"].add(str(rel["target_id"]))

    path_adjacency: dict[uuid.UUID, list[tuple[uuid.UUID, str, float]]] = defaultdict(list)
    for rel in relationship_records:
        path_adjacency[uuid.UUID(str(rel["source_id"]))].append(
            (
                uuid.UUID(str(rel["target_id"])),
                str(rel.get("rel_type") or "RELATES_TO").upper(),
                float(rel.get("weight") or 0.0),
            )
        )

    ranked_pairs = sorted(
        pair_aggregates.values(),
        key=lambda item: (
            item["direct_weight"],
            item["direct_count"],
            len(item["rel_counter"]),
        ),
        reverse=True,
    )[:max_deterministic_link_pairs]

    for bucket in ranked_pairs:
        source_concept_id = str(bucket["source_id"])
        target_concept_id = str(bucket["target_id"])
        source_nodes = {
            uuid.UUID(node_id)
            for node_id in node_sets_by_concept.get(source_concept_id, set())
        }
        target_nodes = {
            uuid.UUID(node_id)
            for node_id in node_sets_by_concept.get(target_concept_id, set())
        }
        if not source_nodes or not target_nodes:
            continue
        paths = _shortest_paths_between_sets(
            path_adjacency,
            source_nodes,
            target_nodes,
            max_depth=4,
            limit=6,
        )
        bridge_counter = Counter()
        path_rel_counter = Counter()
        path_source_ids: set[str] = set()
        for path in paths:
            node_ids = [str(node_id) for node_id in path.get("nodes", [])]
            for node_id in node_ids[1:-1]:
                bridge_counter[node_name_by_id.get(node_id, node_id)] += 1
                path_source_ids.add(node_id)
            for rel_type in path.get("rel_types", []):
                path_rel_counter[str(rel_type)] += 1
            path_source_ids.update(node_ids)

        combined_rel_counter = Counter(bucket["rel_counter"])
        combined_rel_counter.update(path_rel_counter)
        top_rel_types = [rel_type for rel_type, _ in combined_rel_counter.most_common(3)]
        if not top_rel_types:
            continue
        boundary_names = [name for name, _ in bucket["boundary_counter"].most_common(4) if name]
        bridge_names = [name for name, _ in bridge_counter.most_common(4) if name]
        source_label = concept_labels_by_id.get(source_concept_id, source_concept_id)
        target_label = concept_labels_by_id.get(target_concept_id, target_concept_id)
        source_desc = concept_descriptions_by_id.get(source_concept_id, "")
        target_desc = concept_descriptions_by_id.get(target_concept_id, "")
        path_count = len(paths)
        path_score = round(sum(float(path["path_score"]) for path in paths), 6) if paths else 0.0
        edge_source_ids = sorted(
            set(concept_source_ids.get(source_concept_id, []))
            | set(concept_source_ids.get(target_concept_id, []))
            | set(str(value) for value in bucket["source_ids"])
            | path_source_ids
        )
        add_edge(
            source_concept_id,
            target_concept_id,
            rel_type="CONNECTS_TO",
            description=_format_connects_to_description(
                source_label=source_label,
                source_desc=source_desc,
                target_label=target_label,
                target_desc=target_desc,
                top_rel_types=top_rel_types,
                direct_count=int(bucket["direct_count"]),
                path_count=path_count,
                path_score=path_score,
                boundary_names=boundary_names,
                bridge_names=bridge_names,
                code_like=is_code_like,
            ),
            source_ids=edge_source_ids,
            keywords=top_rel_types,
        )
        if len(
            [edge for edge in edges if "__EVIDENCED_BY__" not in edge["id"]]
        ) >= max_deterministic_meta_edges:
            break

    return {
        "nodes": nodes,
        "edges": edges,
        "chunks": chunks,
        "candidate_region_count": len(candidate_regions),
    }

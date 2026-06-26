"""Offline analytics over the canonical collection graph.

Builds role-similarity candidate regions over the base graph and induces
semantic concepts from them via LLM or deterministic fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import uuid
from collections.abc import Awaitable, Callable
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
    primary_type: str = ""


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


# Enhancement defaults favor broader, burner-like recall.
_ROLE_GROUP_OVERLAP_MIN = 2
_ROLE_GROUP_COSINE_MIN = 0.2
_ROLE_GROUP_JACCARD_MIN = 0.1
_ROLE_GROUP_MIN_SIGNATURE = 1

_EDGE_FAMILY_ORDER: tuple[str, ...] = (
    "definitional",
    "taxonomic",
    "compositional",
    "causal",
    "regulatory",
    "therapeutic",
    "constraint",
    "temporal",
    "supporting_evidence",
    "representation",
    "application_use",
    "location_path",
    "bibliographic",
    "associative",
    "generic_other",
)

_ROLE_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "part",
    "the",
    "to",
    "with",
}

_FLOW_TEMPLATES: dict[str, list[dict[str, object]]] = {
    "definition_to_representation": [
        {"step": "definition_or_taxonomy", "families": {"definitional", "taxonomic", "compositional"}},
        {"step": "representation_or_expression", "families": {"representation", "supporting_evidence", "application_use"}},
    ],
    "constraint_to_complexity": [
        {"step": "constraint_or_contrast", "families": {"constraint"}},
        {"step": "extension_or_generalization", "families": {"taxonomic", "causal", "representation"}},
    ],
    "application_to_theory": [
        {"step": "application_or_use", "families": {"application_use", "regulatory", "therapeutic"}},
        {"step": "evidence_or_theory", "families": {"supporting_evidence", "representation", "definitional"}},
    ],
    "causal_evidence_flow": [
        {"step": "cause_or_mechanism", "families": {"causal", "regulatory", "therapeutic"}},
        {"step": "support_or_evidence", "families": {"supporting_evidence", "representation", "definitional"}},
    ],
}


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


def _edge_family_for(
    *,
    rel_type: str,
    source_name: str,
    target_name: str,
) -> str:
    text = " ".join(
        [
            rel_type.upper().replace("_", " "),
            source_name.lower(),
            target_name.lower(),
        ]
    )
    rel = rel_type.upper()

    if any(token in rel for token in ("AUTHOR", "PUBLISHER", "PUBLICATION", "CITES")):
        return "bibliographic"
    if any(token in rel for token in ("CONTRA", "LIMIT", "AVOID", "PROHIBIT", "RESTRICT")):
        return "constraint"
    if any(token in rel for token in ("PRECEDE", "FOLLOW", "BEFORE", "AFTER", "DURATION", "SEQUENCE")):
        return "temporal"
    if any(token in rel for token in ("LOCATED", "TRAVEL", "PASS", "EXTEND", "PATH", "THROUGH", "NEAR", "BETWEEN", "ALONG")):
        return "location_path"
    if any(token in rel for token in ("PART", "CONTAIN", "INCLUDE", "COMPOSE", "CHAPTER", "COMPONENT")):
        return "compositional"
    if any(token in rel for token in ("IS_A", "INSTANCE", "VARIATION", "SUBTYPE", "CLASSIF", "CATEGOR", "LINEAGE")):
        return "taxonomic"
    if any(token in rel for token in ("DEFINE", "ATTRIBUTE", "QUALITY", "CHARACTER", "IDENTIF", "INDICATE")):
        return "definitional"
    if any(token in rel for token in ("SUPPORT", "EVIDENCE", "INFORM", "REFERENCE", "DISCUSS", "COVER", "TOPIC", "EXAMPLE", "STUDIED")):
        return "supporting_evidence"
    if any(token in rel for token in ("CONNECT", "CORRESPOND", "DESCRIBE", "EXPLAIN", "REPRESENT", "MAP", "SYMBOL")):
        return "representation"
    if any(token in rel for token in ("USE", "UTILIZ", "APPL", "REQUIRE", "INVOLVE", "TARGET", "GUIDE", "FACILITATE", "ENABLE", "ASSIST")):
        return "application_use"
    if any(token in rel for token in ("CONTROL", "REGULAT", "BALANCE", "MAINTAIN", "GOVERN", "STIMULAT", "ACTIVAT", "AWAKEN", "ENHANCE")):
        return "regulatory"
    if any(token in rel for token in ("TREAT", "ALLEVIAT", "CALM", "IMPROVE", "REDUCE", "ELIMINAT", "PURIF", "STRENGTHEN", "BENEFIT")):
        return "therapeutic"
    if any(token in rel for token in ("CAUSE", "LEAD", "RESULT", "INFLUENCE", "PRODUCE", "INDUCE", "TRIGGER", "CREATE", "AFFECT", "ALTER", "DEVELOP")):
        return "causal"
    if any(token in text for token in ("practice", "technique", "method", "uses", "used in")):
        return "application_use"
    if any(token in text for token in ("effect", "outcome", "state", "sensation", "manifests")):
        return "causal"
    if any(token in rel for token in ("ASSOCIATED", "RELATED", "INTERACT", "COMBINED", "CO_EQUAL")):
        return "associative"
    return "generic_other"


def _tokens_for_role_name(name: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9]{2,}", name.lower())
        if token not in _ROLE_TOKEN_STOPWORDS and not token.isdigit()
    ]


def _primary_family(counts: Counter[str]) -> str:
    non_generic = Counter(
        {
            family: count
            for family, count in counts.items()
            if family != "generic_other" and count > 0
        }
    )
    source = non_generic or counts
    if not source:
        return "generic_other"
    return max(
        source,
        key=lambda family: (
            source[family],
            -_EDGE_FAMILY_ORDER.index(family)
            if family in _EDGE_FAMILY_ORDER
            else -len(_EDGE_FAMILY_ORDER),
        ),
    )


def _direction_role(out_count: int, in_count: int) -> str:
    total = out_count + in_count
    if total == 0:
        return "isolated"
    if out_count >= in_count * 2:
        return "source"
    if in_count >= out_count * 2:
        return "sink"
    return "bridge"


def _dynamic_role_label(
    *,
    primary_family: str,
    direction_role: str,
    top_tokens: list[str],
) -> str:
    token_text = " / ".join(top_tokens[:3]) if top_tokens else "mixed terms"
    return f"{primary_family.replace('_', ' ').title()} {direction_role.title()}: {token_text}"


def _json_from_llm_text(text: str) -> Any:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return json.loads(content)


def _find_template_paths_for_anchor(
    relationships: list[RelationshipRecord],
    *,
    start_name: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    out_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rel in relationships:
        raw_rel_type = str(rel.rel_type or "RELATES_TO").upper()
        family = _edge_family_for(
            rel_type=raw_rel_type,
            source_name=rel.source_name,
            target_name=rel.target_name,
        )
        out_edges[rel.source_name].append(
            {
                "source": rel.source_name,
                "target": rel.target_name,
                "rel_type": raw_rel_type,
                "edge_family": family,
                "weight": float(rel.weight or 0.0),
            }
        )

    paths: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[tuple[str, str, str], ...]]] = set()
    for template_name, template in _FLOW_TEMPLATES.items():
        if len(template) != 2:
            continue
        first_allowed = set(template[0]["families"])
        second_allowed = set(template[1]["families"])
        first_edges = [
            edge for edge in out_edges.get(start_name, [])
            if edge["edge_family"] in first_allowed
        ]
        first_edges.sort(
            key=lambda edge: (-float(edge["weight"]), str(edge["edge_family"]), str(edge["target"]))
        )
        for first in first_edges[:12]:
            second_edges = [
                edge for edge in out_edges.get(str(first["target"]), [])
                if edge["edge_family"] in second_allowed
            ]
            second_edges.sort(
                key=lambda edge: (-float(edge["weight"]), str(edge["edge_family"]), str(edge["target"]))
            )
            for second in second_edges[:8]:
                key = (
                    template_name,
                    (
                        (first["source"], first["rel_type"], first["target"]),
                        (second["source"], second["rel_type"], second["target"]),
                    ),
                )
                if key in seen:
                    continue
                seen.add(key)
                paths.append(
                    {
                        "template": template_name,
                        "steps": [str(template[0]["step"]), str(template[1]["step"])],
                        "nodes": [first["source"], first["target"], second["target"]],
                        "edges": [first, second],
                        "score": round(float(first["weight"]) + float(second["weight"]), 3),
                    }
                )
    paths.sort(
        key=lambda item: (
            -float(item["score"]),
            str(item["template"]),
            tuple(str(node) for node in item["nodes"]),
        )
    )
    return paths[:limit]


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
    overlap_min: int = _ROLE_GROUP_OVERLAP_MIN,
    cosine_min: float = _ROLE_GROUP_COSINE_MIN,
    jaccard_min: float = _ROLE_GROUP_JACCARD_MIN,
    min_signature: int = _ROLE_GROUP_MIN_SIGNATURE,
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
        all_node_ids.add(source_id)
        all_node_ids.add(target_id)
        out_pairs[source_id].add(("out", target_id))
        in_pairs[target_id].add(("in", source_id))
        relationship_records_by_node[source_id].append(
            {
                "source_id": source_id,
                "source_name": rel.source_name,
                "target_id": target_id,
                "target_name": rel.target_name,
                "rel_type": str(rel.rel_type or "RELATES_TO").upper(),
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
                "rel_type": str(rel.rel_type or "RELATES_TO").upper(),
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


def _build_dynamic_anchor_regions(
    nodes: list[NodeRecord],
    relationships: list[RelationshipRecord],
    *,
    role_count: int = 100,
    min_bucket_size: int = 5,
    anchors_per_role: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not relationships:
        return [], []

    node_name_by_id = {str(node.id): node.name for node in nodes}
    incident_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    out_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    in_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    incident_rel_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    incident_rels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_node_ids: set[str] = set()

    for rel in relationships:
        source_id = str(rel.source_id)
        target_id = str(rel.target_id)
        raw_rel_type = str(rel.rel_type or "RELATES_TO").upper()
        family = _edge_family_for(
            rel_type=raw_rel_type,
            source_name=rel.source_name,
            target_name=rel.target_name,
        )
        weight = float(rel.weight or 0.0)
        all_node_ids.update((source_id, target_id))
        out_family_counts[source_id][family] += 1
        in_family_counts[target_id][family] += 1
        for node_id, direction in ((source_id, "out"), (target_id, "in")):
            incident_family_counts[node_id][family] += 1
            incident_rel_type_counts[node_id][raw_rel_type] += 1
            incident_rels[node_id].append(
                {
                    "source_id": source_id,
                    "source_name": rel.source_name,
                    "target_id": target_id,
                    "target_name": rel.target_name,
                    "rel_type": raw_rel_type,
                    "edge_family": family,
                    "weight": weight,
                    "direction": direction,
                    "relationship_id": str(rel.id),
                }
            )

    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for node_id in sorted(all_node_ids):
        node_name = node_name_by_id.get(node_id, node_id)
        family = _primary_family(incident_family_counts[node_id])
        out_count = sum(out_family_counts[node_id].values())
        in_count = sum(in_family_counts[node_id].values())
        direction = _direction_role(out_count, in_count)
        bucket = buckets.setdefault(
            (family, direction),
            {
                "primary_family": family,
                "direction_role": direction,
                "nodes": [],
                "family_counts": Counter(),
                "rel_type_counts": Counter(),
                "token_counts": Counter(),
                "total_degree": 0,
            },
        )
        degree = out_count + in_count
        bucket["nodes"].append(
            {
                "node_id": node_id,
                "node_name": node_name,
                "degree": degree,
                "out_degree": out_count,
                "in_degree": in_count,
                "edge_family_counts": dict(incident_family_counts[node_id].most_common()),
                "rel_type_counts": dict(incident_rel_type_counts[node_id].most_common(8)),
            }
        )
        bucket["family_counts"].update(incident_family_counts[node_id])
        bucket["rel_type_counts"].update(incident_rel_type_counts[node_id])
        bucket["token_counts"].update(_tokens_for_role_name(node_name))
        bucket["total_degree"] += degree

    role_profiles: list[dict[str, Any]] = []
    for (family, direction), bucket in buckets.items():
        bucket_nodes = bucket["nodes"]
        if len(bucket_nodes) < min_bucket_size:
            continue
        top_tokens = [token for token, _ in bucket["token_counts"].most_common(8)]
        role_id = f"{family}_{direction}"
        role_profiles.append(
            {
                "role_id": role_id,
                "label": _dynamic_role_label(
                    primary_family=family,
                    direction_role=direction,
                    top_tokens=top_tokens,
                ),
                "primary_family": family,
                "direction_role": direction,
                "node_count": len(bucket_nodes),
                "total_degree": bucket["total_degree"],
                "top_tokens": top_tokens,
                "edge_family_counts": dict(bucket["family_counts"].most_common()),
                "rel_type_counts": dict(bucket["rel_type_counts"].most_common(12)),
                "sample_nodes": [
                    item["node_name"]
                    for item in sorted(
                        bucket_nodes,
                        key=lambda item: (-int(item["degree"]), str(item["node_name"])),
                    )[:12]
                ],
                "_nodes": bucket_nodes,
            }
        )
    role_profiles.sort(
        key=lambda item: (
            item["primary_family"] == "generic_other",
            -int(item["node_count"]),
            -int(item["total_degree"]),
            str(item["role_id"]),
        )
    )
    selected_profiles = role_profiles[: max(0, role_count)]

    candidate_regions: list[dict[str, Any]] = []
    selected_anchor_ids: set[str] = set()
    for profile in selected_profiles:
        ranked_nodes = sorted(
            profile["_nodes"],
            key=lambda item: (-int(item["degree"]), str(item["node_name"])),
        )
        kept = 0
        for anchor in ranked_nodes:
            anchor_id = str(anchor["node_id"])
            anchor_name = str(anchor["node_name"])
            if anchor_id in selected_anchor_ids:
                continue
            selected_anchor_ids.add(anchor_id)
            kept += 1
            rels = sorted(
                incident_rels.get(anchor_id, []),
                key=lambda item: (
                    -float(item["weight"]),
                    str(item["edge_family"]),
                    str(item["source_name"]),
                    str(item["target_name"]),
                ),
            )
            representative_edges = rels[:16]
            source_ids = {anchor_id}
            entity_names = {anchor_name}
            for rel in representative_edges[:10]:
                source_ids.add(str(rel["source_id"]))
                source_ids.add(str(rel["target_id"]))
                entity_names.add(str(rel["source_name"]))
                entity_names.add(str(rel["target_name"]))
            template_paths = _find_template_paths_for_anchor(
                relationships,
                start_name=anchor_name,
                limit=3,
            )
            for path in template_paths:
                entity_names.update(str(node) for node in path.get("nodes", []))
            role_id = str(profile["role_id"])
            role_profile = {
                key: value
                for key, value in profile.items()
                if key != "_nodes"
            }
            candidate_regions.append(
                {
                    "region_id": f"dynamic_{role_id}_{len(candidate_regions) + 1}",
                    "kind": "dynamic_anchor",
                    "title": f"{anchor_name} [{role_profile['label']}]",
                    "description": (
                        f"Dynamic anchor '{anchor_name}' selected from role "
                        f"{role_profile['label']} ({role_id}). "
                        f"Primary edge family: {profile['primary_family']}; "
                        f"directional role: {profile['direction_role']}; "
                        f"degree: {anchor['degree']}."
                    ),
                    "source_ids": sorted(source_ids),
                    "entity_names": sorted(entity_names),
                    "rel_types": list(anchor["rel_type_counts"].keys()),
                    "representative_edges": representative_edges,
                    "pair_metrics": [],
                    "anchor": anchor_name,
                    "anchor_id": anchor_id,
                    "dynamic_role": role_profile,
                    "template_paths": template_paths,
                }
            )
            if kept >= anchors_per_role:
                break

    diagnostics = [
        {
            key: value
            for key, value in profile.items()
            if key != "_nodes"
        }
        for profile in selected_profiles
    ]
    return candidate_regions, diagnostics


async def _refine_dynamic_role_profiles(
    collection_name: str,
    role_profiles: list[dict[str, Any]],
    llm_provider: LLMProvider | None,
) -> dict[str, dict[str, str]]:
    if not llm_provider or isinstance(llm_provider, LocalEchoLLMProvider) or not role_profiles:
        return {}
    compact_profiles = [
        {
            "role_id": profile["role_id"],
            "structural_label": profile["label"],
            "primary_family": profile["primary_family"],
            "direction_role": profile["direction_role"],
            "node_count": profile["node_count"],
            "top_tokens": profile["top_tokens"][:8],
            "top_rel_types": list(profile["rel_type_counts"])[:8],
            "sample_nodes": profile["sample_nodes"][:10],
        }
        for profile in role_profiles
    ]
    schema = {
        "type": "object",
        "properties": {
            "roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role_id": {"type": "string"},
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                        "anchor_selection_hint": {"type": "string"},
                    },
                    "required": ["role_id", "label", "description", "anchor_selection_hint"],
                },
            }
        },
        "required": ["roles"],
    }
    prompt = (
        "You name graph role buckets for concept induction. "
        "The buckets were created structurally from edge-family and directionality. "
        "Convert each bucket into a concise, domain-meaningful human role label while preserving role_id exactly. "
        "Prefer ontology roles such as Graph Representations, Foundational Definitions, Empirical Evidence, "
        "Applications, Constraints, Components, or domain-specific equivalents when sample nodes support them.\n\n"
        f"Collection: {collection_name}\n"
        f"Role buckets: {json.dumps(compact_profiles, ensure_ascii=True)}"
    )
    try:
        parsed = await llm_provider.structured_extract(prompt=prompt, schema=schema)
    except Exception:
        try:
            raw = await llm_provider.chat(
                [
                    {
                        "role": "system",
                        "content": "Return only valid JSON with key roles.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            parsed = _json_from_llm_text(raw)
        except Exception:
            return {}
    roles = parsed.get("roles") if isinstance(parsed, dict) else parsed
    if not isinstance(roles, list):
        return {}
    valid_role_ids = {str(profile["role_id"]) for profile in role_profiles}
    refinements: dict[str, dict[str, str]] = {}
    for role in roles:
        if not isinstance(role, dict):
            continue
        role_id = str(role.get("role_id") or "")
        label = str(role.get("label") or "").strip()
        if role_id not in valid_role_ids or not label:
            continue
        refinements[role_id] = {
            "label": label,
            "description": str(role.get("description") or "").strip(),
            "anchor_selection_hint": str(role.get("anchor_selection_hint") or "").strip(),
        }
    return refinements


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
                select(
                    GraphEntity.id,
                    GraphEntity.canonical_name,
                    GraphEntity.primary_type,
                ).where(
                    GraphEntity.collection_id == collection_id
                )
            )
        ).all()

        non_ref_entity_ids = {
            entity_id
            for entity_id, _, primary_type in nodes
            if str(primary_type or "").strip() != "base_entity_ref"
        }

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
        [
            NodeRecord(id=node_id, name=name, primary_type=str(primary_type or ""))
            for node_id, name, primary_type in nodes
            if node_id in non_ref_entity_ids
        ],
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
            if source_id in non_ref_entity_ids and target_id in non_ref_entity_ids
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
    analysis["node_records"] = [
        {
            "id": str(node.id),
            "name": node.name,
            "primary_type": node.primary_type,
        }
        for node in nodes
    ]
    return analysis


async def build_collection_understanding(
    analysis: dict[str, Any],
    llm_provider: LLMProvider | None = None,
    region_batch_size: int = 1,
    on_region_concept: Callable[[dict[str, Any], dict[str, Any]], Awaitable[None]]
    | None = None,
    on_meta_edge: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    collection = analysis["collection"]
    max_deterministic_link_pairs = 64
    max_deterministic_meta_edges = 96
    min_concept_link_direct_count = 2
    min_concept_link_direct_weight = 3.0
    min_concept_link_distinct_rel_types = 2
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
    ) -> dict[str, Any] | None:
        edge_id = f"{source_id}__{rel_type}__{target_id}"
        if edge_id in seen_edge_ids:
            return None
        seen_edge_ids.add(edge_id)
        edge = {
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
        edges.append(edge)
        return edge

    def normalize_map(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_value = max(scores.values()) or 0.0
        if max_value <= 0.0:
            return {key: 0.0 for key in scores}
        return {key: float(value) / float(max_value) for key, value in scores.items()}

    candidate_regions: list[dict[str, Any]] = []
    role_profiles: list[dict[str, Any]] = []
    analysis_nodes = [
        NodeRecord(
            id=uuid.UUID(str(node["id"])),
            name=str(node["name"]),
            primary_type=str(node.get("primary_type") or ""),
        )
        for node in analysis.get("node_records", [])
        if str(node.get("id") or "").strip()
    ]
    analysis_relationships = [
        RelationshipRecord(
            id=uuid.UUID(str(rel["id"])),
            source_id=uuid.UUID(str(rel["source_id"])),
            source_name=str(rel["source_name"]),
            target_id=uuid.UUID(str(rel["target_id"])),
            target_name=str(rel["target_name"]),
            rel_type=str(rel.get("rel_type") or "RELATES_TO"),
            weight=int(rel.get("weight") or 0),
        )
        for rel in relationship_records
        if str(rel.get("id") or "").strip()
    ]
    if analysis_nodes and analysis_relationships:
        candidate_regions, role_profiles = _build_dynamic_anchor_regions(
            analysis_nodes,
            analysis_relationships,
            role_count=100,
            min_bucket_size=5,
            anchors_per_role=1,
        )
        role_refinements = await _refine_dynamic_role_profiles(
            str(collection["name"]),
            role_profiles,
            llm_provider,
        )
        if role_refinements:
            for profile in role_profiles:
                refinement = role_refinements.get(str(profile["role_id"]))
                if refinement:
                    profile["structural_label"] = profile["label"]
                    profile["label"] = refinement["label"]
                    profile["description"] = refinement["description"]
                    profile["anchor_selection_hint"] = refinement["anchor_selection_hint"]
            for region in candidate_regions:
                role = dict(region.get("dynamic_role") or {})
                refinement = role_refinements.get(str(role.get("role_id")))
                if not refinement:
                    continue
                role["structural_label"] = role.get("label")
                role["label"] = refinement["label"]
                role["description"] = refinement["description"]
                role["anchor_selection_hint"] = refinement["anchor_selection_hint"]
                region["dynamic_role"] = role
                region["title"] = f"{region.get('anchor')} [{role['label']}]"
                region["description"] = (
                    f"Dynamic anchor '{region.get('anchor')}' selected from role "
                    f"{role['label']} ({role.get('role_id')}). "
                    f"{role.get('description') or ''}"
                ).strip()

    if not candidate_regions:
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
                        f"total neighborhood overlap {group['total_overlap']}. "
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
    streamed_regions = False
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
            dynamic_role = dict(region.get("dynamic_role") or {})
            template_paths_text = (
                json.dumps(region.get("template_paths", [])[:3], ensure_ascii=True)
                if region.get("template_paths")
                else "none"
            )
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
                "You are inducing one reusable semantic concept from a candidate region in a knowledge graph.\n"
                "The candidate may be a dynamic anchor neighborhood or a role-similarity clique.\n"
                "If a dynamic role label is present, center the concept on that functional role and its local evidence.\n"
                "Do not return a mechanical label like cluster, graph region, connector, bridge, or clique.\n"
                "Infer the higher-level concept, role class, family, pattern, or shared abstraction that these members instantiate together.\n"
                "Prefer labels drawn directly from the collection's own terminology, especially source-language, tradition-specific, or text-native terms when they fit the evidence.\n"
                "If a well-established corpus term fits the pattern, use that term as the label instead of inventing a generic English abstraction.\n"
                "Only fall back to invented English labels when no source-grounded term is adequate.\n"
                "Avoid generic labels like force-user, principle, framework, pattern, agent, or mediator unless the evidence truly does not support a more native term.\n"
                "Use aliases to provide short alternate phrasings or English glosses when helpful, rather than putting the generic gloss in the primary label.\n\n"
                f"{code_guidance}"
                "Also provide a few short aliases or alternate phrasings for the concept when they would help later resolution.\n"
                "The `importance_reason` must be rich and concrete. Do not just say that the members are related or define a lifecycle.\n"
                "Explain the actual sequence, role split, or operational interplay that the member entities capture, in enough detail that a reader could understand what is happening without reopening the code.\n"
                "Name the concrete responsibilities, transitions, inputs, outputs, error paths, and state changes implied by the members when the evidence supports them.\n"
                "Do not describe the answer as a graph, clique, cluster, or evidence chain. Describe the underlying mechanism or workflow itself.\n\n"
                f"Collection: {collection['name']}\n"
                f"Candidate id: {region['region_id']}\n"
                f"Candidate kind: {region.get('kind')}\n"
                f"Candidate title: {region['title']}\n"
                f"Candidate description: {region['description']}\n"
                f"Anchor: {region.get('anchor') or 'none'}\n"
                f"Dynamic role: {json.dumps(dynamic_role, ensure_ascii=True) if dynamic_role else 'none'}\n"
                f"Entities: {', '.join(region['entity_names'][:16]) or 'none'}\n"
                f"Relation types: {', '.join(region['rel_types']) or 'none'}\n"
                f"Pairwise role-similarity evidence: {pair_metrics_text}\n"
                f"Representative neighborhood edges: {directed_edges}\n"
                f"Template traversal paths: {template_paths_text}\n"
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

        induced_concepts: list[dict[str, Any] | None] = [None] * len(candidate_regions)
        batch_size = max(1, int(region_batch_size))

        async def induce_region_concept_at(
            index: int,
            region: dict[str, Any],
        ) -> tuple[int, dict[str, Any]]:
            concept = await induce_region_concept(region)
            return index, concept

        pending: set[asyncio.Task[tuple[int, dict[str, Any]]]] = set()
        next_index = 0
        while next_index < len(candidate_regions) and len(pending) < batch_size:
            region = candidate_regions[next_index]
            pending.add(
                asyncio.create_task(
                    induce_region_concept_at(next_index, region)
                )
            )
            next_index += 1

        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                index, concept = task.result()
                region = candidate_regions[index]
                induced_concepts[index] = concept
                while next_index < len(candidate_regions) and len(pending) < batch_size:
                    next_region = candidate_regions[next_index]
                    pending.add(
                        asyncio.create_task(
                            induce_region_concept_at(next_index, next_region)
                        )
                    )
                    next_index += 1
                if on_region_concept is not None:
                    await on_region_concept(region, concept)
        streamed_regions = True
    region_concepts: list[dict[str, Any]] = [
        {"region": region, "concept": concept}
        for region, concept in zip(candidate_regions, induced_concepts, strict=False)
    ]
    if on_region_concept is not None and not streamed_regions:
        for region_entry in region_concepts:
            await on_region_concept(region_entry["region"], region_entry["concept"])
    region_lookup = {region["region_id"]: region for region in candidate_regions}
    concept_id_by_label: dict[str, str] = {}
    concept_source_ids: dict[str, list[str]] = {}
    concept_labels_by_id: dict[str, str] = {}
    concept_descriptions_by_id: dict[str, str] = {}
    concept_region_ids_by_id: dict[str, list[str]] = {}
    concept_label_keys = {
        str(region_entry["concept"].get("label") or "").strip().casefold()
        for region_entry in region_concepts
        if str(region_entry["concept"].get("label") or "").strip()
    }
    for region_entry in region_concepts:
        concept = region_entry["concept"]
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
            and str(value).strip().casefold() not in concept_label_keys
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
        (
            bucket
            for bucket in pair_aggregates.values()
            if int(bucket["direct_count"]) >= min_concept_link_direct_count
            or float(bucket["direct_weight"]) >= min_concept_link_direct_weight
            or len(bucket["rel_counter"]) >= min_concept_link_distinct_rel_types
        ),
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
        created_edge = add_edge(
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
        if created_edge is not None and on_meta_edge is not None:
            await on_meta_edge(created_edge)
        if len(
            [edge for edge in edges if "__EVIDENCED_BY__" not in edge["id"]]
        ) >= max_deterministic_meta_edges:
            break

    return {
        "nodes": nodes,
        "edges": edges,
        "chunks": chunks,
        "regions": region_concepts,
        "candidate_region_count": len(candidate_regions),
    }

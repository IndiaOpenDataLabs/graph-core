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
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

import networkx as nx
from scipy.sparse.linalg import eigsh
from sqlalchemy import delete, insert, select, text
from sqlalchemy.orm import aliased

from graph_core.database import AsyncSessionLocal
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.llm import LocalEchoLLMProvider
from graph_core.llm.interface import LLMProvider
from graph_core.models.collection import Collection
from graph_core.models.graph_rag import (
    EntityAlias,
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
    RelationshipDescription,
)
from graph_core.models.incremental_graph import (
    GraphCommunity,
    GraphCommunityMembership,
    GraphComponentMetric,
    GraphNodeMetric,
    GraphProjectionSnapshot,
    GraphVersion,
)
from graph_core.models.rel_types import rel_types_for_domain
from graph_core.services.graph.semantic_frames import (
    FrameArgumentInput,
    SemanticFrameInput,
    deterministic_frame_id,
    replace_community_frames,
)


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
    description: str = ""
    predicate_properties: dict[str, Any] | None = None


@dataclass(slots=True)
class AssertionRecord:
    id: uuid.UUID
    source_concept_id: uuid.UUID
    source_concept_name: str
    source_concept_type: str
    target_concept_id: uuid.UUID
    target_concept_name: str
    target_concept_type: str
    rel_type: str
    weight: int
    source_role: str
    target_role: str
    source_mention_id: uuid.UUID
    target_mention_id: uuid.UUID
    description: str = ""
    assertion_id: uuid.UUID | None = None
    assertion_name: str = ""
    context_id: uuid.UUID | None = None
    context_name: str = ""
    source_document: str = ""
    source_sections: tuple[str, ...] = ()


_CODE_REL_TYPES = {value.upper() for value in rel_types_for_domain("code")}
_NON_CODE_REL_TYPES = {
    value.upper()
    for domain in ("general", "books", "personal")
    for value in rel_types_for_domain(domain)
}
_CODE_ONLY_REL_TYPES = _CODE_REL_TYPES - _NON_CODE_REL_TYPES
_CONTEXT_SCAFFOLD_NODE_TYPES = {"CONTEXT", "ASSERTION"}
_CONTEXT_SCAFFOLD_REL_TYPES = {
    "HAS_ASSERTION",
    "HAS_SUBJECT_MENTION",
    "HAS_OBJECT_MENTION",
    "DENOTES",
}
_SOURCE_HIERARCHY_NODE_TYPES = {
    "SOURCE_FOLDER",
    "SOURCE_DOCUMENT",
    "SOURCE_SECTION",
}
_SOURCE_HIERARCHY_REL_TYPES = {"CONTAINS", "HAS_SECTION", "HAS_CONTEXT"}
_COMMUNITY_EXCLUDED_NODE_TYPES = {
    "ASSERTION",
    "CONDITION",
    "CONTEXT",
    "EVIDENCE_CHUNK",
    "EXCEPTION",
    "PROPOSITION",
    "RULE",
    "SCOPE",
    "SOURCE_DOCUMENT",
    "SOURCE_FOLDER",
    "SOURCE_SECTION",
}

_PROVENANCE_NODE_TYPES = {
    "ASSERTION",
    "CONTEXT",
    "EVIDENCE_CHUNK",
    "SOURCE_DOCUMENT",
    "SOURCE_FOLDER",
    "SOURCE_SECTION",
}
_REASONING_NODE_TYPES = {"CONDITION", "EXCEPTION", "PROPOSITION", "RULE", "SCOPE"}
_REASONING_EDGE_TYPES = {
    "ANTECEDENT_OF",
    "APPLIES_TO",
    "BLOCKS",
    "CONCLUDES",
    "CONDITION",
    "EXCEPTION",
    "OBJECT",
    "SUBJECT",
    "SUPPORTS",
}
_PROJECTION_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "semantic_entities_directed",
        "directed": True,
        "node_policy": "semantic_entities",
        "edge_policy": "non_reasoning",
        "parallel_edges": "sum_weight",
        "self_loops": "exclude",
    },
    {
        "name": "semantic_affinity_undirected",
        "directed": False,
        "node_policy": "semantic_entities",
        "edge_policy": "non_reasoning",
        "parallel_edges": "sum_weight",
        "self_loops": "exclude",
    },
    {
        "name": "rule_dependency_directed",
        "directed": True,
        "node_policy": "reasoning_and_arguments",
        "edge_policy": "reasoning_only",
        "parallel_edges": "sum_weight",
        "self_loops": "exclude",
    },
    {
        "name": "causal_semantics_directed",
        "directed": True,
        "node_policy": "semantic_entities",
        "edge_policy": "non_reasoning",
        "property_filter": {"causal": "causal"},
        "parallel_edges": "sum_weight",
        "self_loops": "exclude",
    },
    {
        "name": "temporal_semantics_directed",
        "directed": True,
        "node_policy": "semantic_entities",
        "edge_policy": "non_reasoning",
        "property_filter": {"temporal": "temporal"},
        "parallel_edges": "sum_weight",
        "self_loops": "exclude",
    },
)


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


def _is_context_concept_type(primary_type: str) -> bool:
    return primary_type.upper().startswith("CONCEPT_")


def _is_context_mention_type(primary_type: str) -> bool:
    return primary_type.upper().startswith("MENTION_")


def _has_context_scaffold(nodes: list[NodeRecord]) -> bool:
    primary_types = {node.primary_type.upper() for node in nodes}
    return bool(primary_types & _CONTEXT_SCAFFOLD_NODE_TYPES) or any(
        _is_context_mention_type(primary_type)
        or _is_context_concept_type(primary_type)
        for primary_type in primary_types
    )


def _source_hierarchy_by_context(
    node_by_id: dict[uuid.UUID, NodeRecord],
    relationships: list[RelationshipRecord],
) -> dict[uuid.UUID, dict[str, Any]]:
    children_by_source: dict[uuid.UUID, list[RelationshipRecord]] = defaultdict(list)
    for rel in relationships:
        rel_type = str(rel.rel_type or "").upper()
        if rel_type in _SOURCE_HIERARCHY_REL_TYPES:
            children_by_source[rel.source_id].append(rel)

    hierarchy: dict[uuid.UUID, dict[str, Any]] = {}

    def walk(
        node_id: uuid.UUID,
        *,
        document: str,
        sections: tuple[str, ...],
        seen: set[uuid.UUID],
    ) -> None:
        if node_id in seen:
            return
        node = node_by_id.get(node_id)
        if node is None:
            return
        node_type = str(node.primary_type or "").upper()
        next_document = document
        next_sections = sections
        if node_type == "SOURCE_DOCUMENT":
            next_document = node.name
        elif node_type == "SOURCE_SECTION":
            next_sections = (*sections, node.name)
        elif node_type == "CONTEXT":
            current = hierarchy.get(node_id)
            candidate = {
                "source_document": next_document,
                "source_sections": next_sections,
            }
            if current is None or (
                not current.get("source_document")
                and candidate.get("source_document")
            ) or len(candidate["source_sections"]) > len(current["source_sections"]):
                hierarchy[node_id] = candidate
            return

        next_seen = {*seen, node_id}
        for rel in children_by_source.get(node_id, []):
            walk(
                rel.target_id,
                document=next_document,
                sections=next_sections,
                seen=next_seen,
            )

    for node in node_by_id.values():
        if str(node.primary_type or "").upper() in _SOURCE_HIERARCHY_NODE_TYPES:
            walk(node.id, document="", sections=(), seen=set())
    return hierarchy


def _project_context_scaffold_graph(
    nodes: list[NodeRecord],
    relationships: list[RelationshipRecord],
) -> tuple[list[NodeRecord], list[RelationshipRecord], list[AssertionRecord]]:
    """Return the semantic concept graph from context/assertion/mention scaffolding."""
    node_by_id = {node.id: node for node in nodes}
    concept_ids = {
        node.id
        for node in nodes
        if _is_context_concept_type(node.primary_type)
    }
    mention_ids = {
        node.id
        for node in nodes
        if _is_context_mention_type(node.primary_type)
    }
    if not concept_ids or not mention_ids:
        return nodes, relationships, []

    concept_id_by_mention_id: dict[uuid.UUID, uuid.UUID] = {}
    assertion_id_by_mention_id: dict[uuid.UUID, uuid.UUID] = {}
    context_id_by_assertion_id: dict[uuid.UUID, uuid.UUID] = {}
    context_id_by_context_child_id: dict[uuid.UUID, uuid.UUID] = {}
    source_hierarchy = _source_hierarchy_by_context(node_by_id, relationships)
    for rel in relationships:
        rel_type = str(rel.rel_type or "").upper()
        if (
            rel_type == "DENOTES"
            and rel.source_id in mention_ids
            and rel.target_id in concept_ids
        ):
            concept_id_by_mention_id[rel.source_id] = rel.target_id
        elif (
            rel_type in {"HAS_SUBJECT_MENTION", "HAS_OBJECT_MENTION"}
            and node_by_id.get(rel.source_id) is not None
            and node_by_id[rel.source_id].primary_type.upper() == "ASSERTION"
            and rel.target_id in mention_ids
        ):
            assertion_id_by_mention_id[rel.target_id] = rel.source_id
        elif (
            rel_type == "HAS_ASSERTION"
            and node_by_id.get(rel.source_id) is not None
            and node_by_id[rel.source_id].primary_type.upper() == "CONTEXT"
            and node_by_id.get(rel.target_id) is not None
            and node_by_id[rel.target_id].primary_type.upper() == "ASSERTION"
        ):
            context_id_by_assertion_id[rel.target_id] = rel.source_id
        elif (
            rel_type == "HAS_CONTEXT"
            and node_by_id.get(rel.target_id) is not None
            and node_by_id[rel.target_id].primary_type.upper() == "CONTEXT"
        ):
            context_id_by_context_child_id[rel.target_id] = rel.target_id

    projected_relationships: list[RelationshipRecord] = []
    assertion_records: list[AssertionRecord] = []
    for rel in relationships:
        rel_type = str(rel.rel_type or "RELATES_TO").upper()
        if rel_type in _CONTEXT_SCAFFOLD_REL_TYPES:
            continue
        if rel.source_id not in mention_ids or rel.target_id not in mention_ids:
            continue
        source_concept_id = concept_id_by_mention_id.get(rel.source_id)
        target_concept_id = concept_id_by_mention_id.get(rel.target_id)
        if (
            source_concept_id is None
            or target_concept_id is None
            or source_concept_id == target_concept_id
        ):
            continue
        source_node = node_by_id.get(source_concept_id)
        target_node = node_by_id.get(target_concept_id)
        if source_node is None or target_node is None:
            continue
        projected_relationships.append(
            RelationshipRecord(
                id=rel.id,
                source_id=source_concept_id,
                source_name=source_node.name,
                target_id=target_concept_id,
                target_name=target_node.name,
                rel_type=rel_type,
                weight=rel.weight,
                description=rel.description,
            )
        )
        assertion_id = (
            assertion_id_by_mention_id.get(rel.source_id)
            or assertion_id_by_mention_id.get(rel.target_id)
        )
        assertion_node = node_by_id.get(assertion_id) if assertion_id else None
        context_id = (
            context_id_by_assertion_id.get(assertion_id)
            if assertion_id is not None
            else None
        )
        if context_id is None:
            context_id = context_id_by_context_child_id.get(rel.source_id)
        if context_id is None:
            context_id = context_id_by_context_child_id.get(rel.target_id)
        context_node = node_by_id.get(context_id) if context_id else None
        hierarchy = source_hierarchy.get(context_id, {}) if context_id else {}
        assertion_records.append(
            AssertionRecord(
                id=rel.id,
                source_concept_id=source_concept_id,
                source_concept_name=source_node.name,
                source_concept_type=source_node.primary_type,
                target_concept_id=target_concept_id,
                target_concept_name=target_node.name,
                target_concept_type=target_node.primary_type,
                rel_type=rel_type,
                weight=rel.weight,
                source_role="source",
                target_role="target",
                source_mention_id=rel.source_id,
                target_mention_id=rel.target_id,
                description=rel.description,
                assertion_id=assertion_id,
                assertion_name=assertion_node.name if assertion_node else "",
                context_id=context_id,
                context_name=context_node.name if context_node else "",
                source_document=str(hierarchy.get("source_document") or ""),
                source_sections=tuple(
                    str(section)
                    for section in hierarchy.get("source_sections", ())
                    if str(section).strip()
                ),
            )
        )

    return (
        [node for node in nodes if node.id in concept_ids],
        [
            rel
            for rel in projected_relationships
            if str(rel.rel_type or "").upper() not in _SOURCE_HIERARCHY_REL_TYPES
        ],
        assertion_records,
    )


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
                "description": rel.description,
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
                "description": rel.description,
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
                    "description": rel.description,
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


def _concept_kind(primary_type: str, name: str = "") -> str:
    raw = str(primary_type or "").strip().upper()
    if raw.startswith("CONCEPT_"):
        return raw.removeprefix("CONCEPT_").lower() or "concept"
    if ":" in name:
        return name.split(":", 1)[0].strip().lower() or "concept"
    return raw.lower() or "concept"


def _build_facet_candidate_regions(
    assertion_records: list[dict[str, Any]],
    *,
    max_facets_per_referent: int = 6,
    max_regions: int = 120,
) -> list[dict[str, Any]]:
    if not assertion_records:
        return []

    assertions_by_referent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    facet_buckets: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def add_view(
        *,
        record: dict[str, Any],
        referent_id: str,
        referent_name: str,
        referent_type: str,
        counterpart_id: str,
        counterpart_name: str,
        counterpart_type: str,
        role: str,
    ) -> None:
        rel_type = str(record.get("rel_type") or "RELATES_TO").upper()
        family = _edge_family_for(
            rel_type=rel_type,
            source_name=str(record.get("source_concept_name") or ""),
            target_name=str(record.get("target_concept_name") or ""),
        )
        counterpart_kind = _concept_kind(counterpart_type, counterpart_name)
        key = (referent_id, role, family, counterpart_kind)
        bucket = facet_buckets.setdefault(
            key,
            {
                "referent_id": referent_id,
                "referent_name": referent_name,
                "referent_type": referent_type,
                "role": role,
                "edge_family": family,
                "counterpart_kind": counterpart_kind,
                "assertions": [],
                "counterpart_ids": set(),
                "counterpart_names": set(),
                "rel_counter": Counter(),
                "context_names": set(),
                "source_documents": set(),
                "source_sections": set(),
                "score": 0.0,
            },
        )
        bucket["assertions"].append(record)
        bucket["counterpart_ids"].add(counterpart_id)
        bucket["counterpart_names"].add(counterpart_name)
        bucket["rel_counter"][rel_type] += 1
        context_name = str(record.get("context_name") or "").strip()
        if context_name:
            bucket["context_names"].add(context_name)
        source_document = str(record.get("source_document") or "").strip()
        if source_document:
            bucket["source_documents"].add(source_document)
        for section in record.get("source_sections", []) or []:
            section_name = str(section or "").strip()
            if section_name:
                bucket["source_sections"].add(section_name)
        bucket["score"] += float(record.get("weight") or 0.0) + 1.0

    for record in assertion_records:
        source_id = str(record.get("source_concept_id") or "").strip()
        target_id = str(record.get("target_concept_id") or "").strip()
        if not source_id or not target_id or source_id == target_id:
            continue
        assertions_by_referent[source_id].append(record)
        assertions_by_referent[target_id].append(record)
        add_view(
            record=record,
            referent_id=source_id,
            referent_name=str(record.get("source_concept_name") or source_id),
            referent_type=str(record.get("source_concept_type") or ""),
            counterpart_id=target_id,
            counterpart_name=str(record.get("target_concept_name") or target_id),
            counterpart_type=str(record.get("target_concept_type") or ""),
            role="source",
        )
        add_view(
            record=record,
            referent_id=target_id,
            referent_name=str(record.get("target_concept_name") or target_id),
            referent_type=str(record.get("target_concept_type") or ""),
            counterpart_id=source_id,
            counterpart_name=str(record.get("source_concept_name") or source_id),
            counterpart_type=str(record.get("source_concept_type") or ""),
            role="target",
        )

    ranked_buckets = sorted(
        facet_buckets.values(),
        key=lambda bucket: (
            -len(bucket["assertions"]),
            -float(bucket["score"]),
            str(bucket["referent_name"]),
            str(bucket["edge_family"]),
            str(bucket["counterpart_kind"]),
        ),
    )
    kept_by_referent: Counter[str] = Counter()
    regions: list[dict[str, Any]] = []
    for bucket in ranked_buckets:
        referent_id = str(bucket["referent_id"])
        if len(assertions_by_referent[referent_id]) < 2:
            continue
        if kept_by_referent[referent_id] >= max_facets_per_referent:
            continue
        kept_by_referent[referent_id] += 1

        rel_types = [rel_type for rel_type, _ in bucket["rel_counter"].most_common()]
        counterpart_names = sorted(str(name) for name in bucket["counterpart_names"])
        counterpart_ids = sorted(str(node_id) for node_id in bucket["counterpart_ids"])
        representative_edges = []
        assertion_ids = []
        source_ids = {referent_id, *counterpart_ids}
        for record in bucket["assertions"]:
            assertion_id = str(record.get("assertion_id") or "").strip()
            if assertion_id:
                assertion_ids.append(assertion_id)
            representative_edges.append(
                {
                    "source_id": str(record.get("source_concept_id") or ""),
                    "source_name": str(record.get("source_concept_name") or ""),
                    "target_id": str(record.get("target_concept_id") or ""),
                    "target_name": str(record.get("target_concept_name") or ""),
                    "rel_type": str(record.get("rel_type") or "RELATES_TO").upper(),
                    "weight": float(record.get("weight") or 0.0),
                    "relationship_id": str(record.get("id") or ""),
                    "assertion_id": assertion_id,
                    "assertion_name": str(record.get("assertion_name") or ""),
                    "context_name": str(record.get("context_name") or ""),
                    "source_document": str(record.get("source_document") or ""),
                    "source_sections": list(record.get("source_sections") or []),
                    "description": str(record.get("description") or ""),
                }
            )
        role = str(bucket["role"])
        family_label = str(bucket["edge_family"]).replace("_", " ")
        title = (
            f"{bucket['referent_name']} as {role} in "
            f"{family_label} {bucket['counterpart_kind']} relations"
        )
        regions.append(
            {
                "region_id": (
                    f"facet_{len(regions) + 1}_"
                    f"{hashlib.md5(title.encode('utf-8')).hexdigest()[:10]}"
                ),
                "kind": "referent_facet",
                "title": title,
                "description": (
                    f"{bucket['referent_name']} appears as {role} across "
                    f"{len(bucket['assertions'])} assertion(s), mainly via "
                    f"{', '.join(rel_types) or 'RELATES_TO'} toward "
                    f"{bucket['counterpart_kind']} counterpart(s): "
                    f"{', '.join(counterpart_names[:8])}."
                ),
                "source_ids": sorted(source_ids),
                "entity_names": [str(bucket["referent_name"]), *counterpart_names],
                "rel_types": rel_types,
                "representative_edges": representative_edges[:16],
                "pair_metrics": [],
                "anchor": str(bucket["referent_name"]),
                "anchor_id": referent_id,
                "facet_profile": {
                    "referent_id": referent_id,
                    "referent_name": str(bucket["referent_name"]),
                    "referent_type": str(bucket["referent_type"]),
                    "role": role,
                    "edge_family": str(bucket["edge_family"]),
                    "counterpart_kind": str(bucket["counterpart_kind"]),
                    "counterpart_names": counterpart_names[:12],
                    "rel_type_counts": dict(bucket["rel_counter"].most_common()),
                    "assertion_ids": sorted(set(assertion_ids)),
                    "context_names": sorted(bucket["context_names"])[:8],
                    "source_documents": sorted(bucket["source_documents"])[:8],
                    "source_sections": sorted(bucket["source_sections"])[:12],
                },
            }
        )
        if len(regions) >= max_regions:
            break
    return regions


def _build_shared_class_candidate_regions(
    assertion_records: list[dict[str, Any]],
    *,
    min_members: int = 2,
    max_regions: int = 80,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for record in assertion_records:
        source_id = str(record.get("source_concept_id") or "").strip()
        target_id = str(record.get("target_concept_id") or "").strip()
        source_name = str(record.get("source_concept_name") or source_id).strip()
        target_name = str(record.get("target_concept_name") or target_id).strip()
        if not source_id or not target_id or source_id == target_id:
            continue
        rel_type = str(record.get("rel_type") or "RELATES_TO").upper()
        key = (rel_type, target_id)
        bucket = buckets.setdefault(
            key,
            {
                "rel_type": rel_type,
                "target_id": target_id,
                "target_name": target_name,
                "target_type": str(record.get("target_concept_type") or ""),
                "members": {},
                "assertions": [],
                "source_documents": set(),
                "source_sections": set(),
                "score": 0.0,
            },
        )
        bucket["members"][source_id] = {
            "id": source_id,
            "name": source_name,
            "type": str(record.get("source_concept_type") or ""),
        }
        bucket["assertions"].append(record)
        source_document = str(record.get("source_document") or "").strip()
        if source_document:
            bucket["source_documents"].add(source_document)
        for section in record.get("source_sections", []) or []:
            section_name = str(section or "").strip()
            if section_name:
                bucket["source_sections"].add(section_name)
        bucket["score"] += float(record.get("weight") or 0.0) + 1.0

    ranked = sorted(
        buckets.values(),
        key=lambda bucket: (
            -len(bucket["members"]),
            -float(bucket["score"]),
            str(bucket["rel_type"]),
            str(bucket["target_name"]),
        ),
    )
    regions: list[dict[str, Any]] = []
    for bucket in ranked:
        members = list(bucket["members"].values())
        if len(members) < min_members:
            continue
        member_names = sorted(str(member["name"]) for member in members)
        member_ids = sorted(str(member["id"]) for member in members)
        representative_edges = []
        assertion_ids = []
        for record in bucket["assertions"]:
            assertion_id = str(record.get("assertion_id") or "").strip()
            if assertion_id:
                assertion_ids.append(assertion_id)
            representative_edges.append(
                {
                    "source_id": str(record.get("source_concept_id") or ""),
                    "source_name": str(record.get("source_concept_name") or ""),
                    "target_id": str(record.get("target_concept_id") or ""),
                    "target_name": str(record.get("target_concept_name") or ""),
                    "rel_type": str(record.get("rel_type") or "RELATES_TO").upper(),
                    "weight": float(record.get("weight") or 0.0),
                    "relationship_id": str(record.get("id") or ""),
                    "assertion_id": assertion_id,
                    "assertion_name": str(record.get("assertion_name") or ""),
                    "context_name": str(record.get("context_name") or ""),
                    "source_document": str(record.get("source_document") or ""),
                    "source_sections": list(record.get("source_sections") or []),
                    "description": str(record.get("description") or ""),
                }
            )
        rel_type = str(bucket["rel_type"])
        target_name = str(bucket["target_name"])
        title = f"{', '.join(member_names[:8])} as {target_name} members"
        regions.append(
            {
                "region_id": (
                    f"shared_class_{len(regions) + 1}_"
                    f"{hashlib.md5(title.encode('utf-8')).hexdigest()[:10]}"
                ),
                "kind": "shared_class",
                "title": title,
                "description": (
                    f"{len(member_names)} entities share {rel_type} assertions "
                    f"to {target_name}: {', '.join(member_names[:12])}."
                ),
                "source_ids": sorted({*member_ids, str(bucket["target_id"])}),
                "entity_names": [*member_names, target_name],
                "rel_types": [rel_type],
                "representative_edges": representative_edges[:24],
                "pair_metrics": [],
                "anchor": target_name,
                "anchor_id": str(bucket["target_id"]),
                "shared_class_profile": {
                    "class_id": str(bucket["target_id"]),
                    "class_name": target_name,
                    "class_type": str(bucket["target_type"]),
                    "member_ids": member_ids,
                    "member_names": member_names,
                    "rel_type": rel_type,
                    "assertion_ids": sorted(set(assertion_ids)),
                    "source_documents": sorted(bucket["source_documents"])[:8],
                    "source_sections": sorted(bucket["source_sections"])[:12],
                },
            }
        )
        if len(regions) >= max_regions:
            break
    return regions


def _region_evidence_keys(region: dict[str, Any]) -> tuple[set[str], set[str]]:
    assertion_ids: set[str] = set()
    context_names: set[str] = set()
    profile = region.get("facet_profile") or region.get("shared_class_profile") or {}
    if isinstance(profile, dict):
        assertion_ids.update(
            str(value).strip()
            for value in profile.get("assertion_ids", [])
            if str(value).strip()
        )
        context_names.update(
            str(value).strip()
            for value in profile.get("context_names", [])
            if str(value).strip()
        )
        context_names.update(
            str(value).strip()
            for value in profile.get("source_documents", [])
            if str(value).strip()
        )
        context_names.update(
            str(value).strip()
            for value in profile.get("source_sections", [])
            if str(value).strip()
        )
    for edge in region.get("representative_edges", []):
        if not isinstance(edge, dict):
            continue
        assertion_id = str(
            edge.get("assertion_id") or edge.get("relationship_id") or ""
        ).strip()
        if assertion_id:
            assertion_ids.add(assertion_id)
        context_name = str(edge.get("context_name") or "").strip()
        if context_name:
            context_names.add(context_name)
        source_document = str(edge.get("source_document") or "").strip()
        if source_document:
            context_names.add(source_document)
        for section in edge.get("source_sections", []) or []:
            section_name = str(section or "").strip()
            if section_name:
                context_names.add(section_name)
    return assertion_ids, context_names


def _meta_relation_type_for_evidence(rel_types: list[str]) -> str:
    normalized = {str(rel_type).strip().upper() for rel_type in rel_types if rel_type}
    if not normalized:
        return "CONNECTS_TO"
    if normalized & {
        "IS_A",
        "INSTANCE_OF",
        "TYPE_OF",
        "SUBCLASS_OF",
        "INHERITS_FROM",
        "IMPLEMENTS",
        "EXTENDS",
    }:
        return "SPECIALIZES"
    if normalized & {
        "PART_OF",
        "HAS_PART",
        "CONTAINS",
        "BELONGS_TO",
        "INCLUDES",
        "COMPOSED_OF",
    }:
        return "PART_OF"
    if normalized & {
        "CAUSES",
        "ENABLES",
        "TRIGGERS",
        "PRODUCES",
        "GENERATES",
        "CREATES",
        "LEADS_TO",
        "RESULTS_IN",
    }:
        return "ENABLES"
    if normalized & {
        "GUIDES",
        "INFORMS",
        "EXPLAINS",
        "DESCRIBES",
        "TEACHES",
        "SUPPORTS",
        "EVIDENCES",
    }:
        return "INFORMS"
    if normalized & {
        "CALLS",
        "USES",
        "DEPENDS_ON",
        "REQUIRES",
        "IMPORTS",
        "READS",
        "WRITES",
    }:
        return "DEPENDS_ON"
    return "CONNECTS_TO"


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
    list[AssertionRecord],
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
                    GraphRelationshipType.inferred_properties,
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
        description_rows = (
            await session.execute(
                select(
                    RelationshipDescription.relationship_id,
                    RelationshipDescription.description,
                )
                .join(
                    GraphRelationship,
                    GraphRelationship.id == RelationshipDescription.relationship_id,
                )
                .where(GraphRelationship.collection_id == collection_id)
            )
        ).all()

    aliases_by_entity_id: dict[str, list[str]] = defaultdict(list)
    for entity_id, alias_name in alias_rows:
        alias = str(alias_name or "").strip()
        if not alias:
            continue
        aliases_by_entity_id[str(entity_id)].append(alias)
    description_by_relationship_id: dict[uuid.UUID, str] = {}
    for relationship_id, description in description_rows:
        text = str(description or "").strip()
        if not text:
            continue
        current = description_by_relationship_id.get(relationship_id, "")
        if len(text) > len(current):
            description_by_relationship_id[relationship_id] = text

    loaded_nodes = [
        NodeRecord(id=node_id, name=name, primary_type=str(primary_type or ""))
        for node_id, name, primary_type in nodes
        if node_id in non_ref_entity_ids
    ]
    loaded_relationships = [
        RelationshipRecord(
            id=rel_id,
            source_id=source_id,
            source_name=source_name,
            target_id=target_id,
            target_name=target_name,
            rel_type=rel_type,
            weight=int(weight or 0),
            description=description_by_relationship_id.get(rel_id, ""),
            predicate_properties=dict(predicate_properties or {}),
        )
        for (
            rel_id,
            source_id,
            source_name,
            target_id,
            target_name,
            rel_type,
            weight,
            predicate_properties,
        ) in relationships
        if source_id in non_ref_entity_ids and target_id in non_ref_entity_ids
    ]
    assertion_records: list[AssertionRecord] = []
    if _has_context_scaffold(loaded_nodes):
        loaded_nodes, loaded_relationships, assertion_records = _project_context_scaffold_graph(
            loaded_nodes,
            loaded_relationships,
        )
    loaded_node_ids = {str(node.id) for node in loaded_nodes}
    return (
        collection,
        loaded_nodes,
        loaded_relationships,
        assertion_records,
        {
            entity_id: sorted({alias for alias in aliases})
            for entity_id, aliases in aliases_by_entity_id.items()
            if entity_id in loaded_node_ids
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


def _build_semantic_communities(
    nodes: list[NodeRecord],
    relationships: list[RelationshipRecord],
) -> list[dict[str, Any]]:
    """Compute deterministic weighted communities for navigation, not evidence."""
    graph = nx.Graph()
    node_by_id = {
        node.id: node
        for node in nodes
        if node.primary_type.upper() not in _COMMUNITY_EXCLUDED_NODE_TYPES
        and not node.primary_type.upper().startswith("MENTION_")
    }
    graph.add_nodes_from(node_by_id)
    for relationship in relationships:
        if (
            relationship.source_id not in node_by_id
            or relationship.target_id not in node_by_id
        ):
            continue
        if relationship.source_id == relationship.target_id:
            continue
        weight = max(1.0, float(relationship.weight or 1))
        if graph.has_edge(relationship.source_id, relationship.target_id):
            graph[relationship.source_id][relationship.target_id]["weight"] += weight
        else:
            graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                weight=weight,
            )
    if graph.number_of_edges() == 0:
        return []
    pagerank = nx.pagerank(graph, weight="weight")
    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        seed=0,
    )
    results: list[dict[str, Any]] = []
    for members in communities:
        if len(members) < 2:
            continue
        member_set = set(members)
        ordered = sorted(
            member_set,
            key=lambda node_id: (-pagerank.get(node_id, 0.0), str(node_id)),
        )
        predicate_counts = Counter(
            relationship.rel_type
            for relationship in relationships
            if relationship.source_id in member_set
            and relationship.target_id in member_set
        )
        results.append(
            {
                "member_ids": [str(node_id) for node_id in ordered],
                "member_names": [node_by_id[node_id].name for node_id in ordered],
                "predicate_families": [
                    predicate for predicate, _ in predicate_counts.most_common(8)
                ],
            }
        )
    results.sort(key=lambda item: (-len(item["member_ids"]), item["member_ids"][0]))
    return results


def _projection_spec_hash(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _build_projection(
    nodes: list[NodeRecord],
    relationships: list[RelationshipRecord],
    spec: dict[str, Any],
) -> nx.Graph | nx.DiGraph:
    graph: nx.Graph | nx.DiGraph = nx.DiGraph() if spec["directed"] else nx.Graph()

    def include_node(node: NodeRecord) -> bool:
        node_type = node.primary_type.upper()
        if node_type in _PROVENANCE_NODE_TYPES or node_type.startswith("MENTION_"):
            return False
        if spec["node_policy"] == "semantic_entities":
            return node_type not in _REASONING_NODE_TYPES
        return True

    included = {node.id for node in nodes if include_node(node)}
    property_filter = dict(spec.get("property_filter") or {})
    if not property_filter:
        graph.add_nodes_from(included)
    for relationship in relationships:
        if relationship.source_id == relationship.target_id:
            continue
        if relationship.source_id not in included or relationship.target_id not in included:
            continue
        reasoning_edge = relationship.rel_type.upper() in _REASONING_EDGE_TYPES
        if spec["edge_policy"] == "reasoning_only" and not reasoning_edge:
            continue
        if spec["edge_policy"] == "non_reasoning" and reasoning_edge:
            continue
        properties = relationship.predicate_properties or {}
        if any(properties.get(name) != value for name, value in property_filter.items()):
            continue
        graph.add_nodes_from((relationship.source_id, relationship.target_id))
        weight = max(1, int(relationship.weight or 1))
        if graph.has_edge(relationship.source_id, relationship.target_id):
            graph[relationship.source_id][relationship.target_id]["weight"] += weight
        else:
            graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                weight=weight,
            )
        if (
            graph.is_directed()
            and (
                properties.get("symmetry") == "symmetric"
                or properties.get("directionality") == "undirected"
            )
            and not graph.has_edge(relationship.target_id, relationship.source_id)
        ):
            graph.add_edge(
                relationship.target_id,
                relationship.source_id,
                weight=weight,
                inferred_symmetric=True,
            )
    return graph


def _ranked_metric(values: dict[uuid.UUID, float]) -> list[tuple[uuid.UUID, float, int]]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], str(item[0])))
    return [
        (node_id, float(value), rank)
        for rank, (node_id, value) in enumerate(ordered, 1)
    ]


def _spectral_diagnostics(graph: nx.Graph) -> dict[str, float | str]:
    if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
        return {"status": "insufficient_graph"}
    largest_nodes = max(nx.connected_components(graph), key=len)
    component = graph.subgraph(largest_nodes)
    if component.number_of_nodes() < 3:
        return {"status": "insufficient_component"}
    try:
        laplacian = nx.normalized_laplacian_matrix(component, weight="weight")
        eigenvalues = sorted(
            float(value)
            for value in eigsh(
                laplacian,
                k=2,
                which="SM",
                return_eigenvectors=False,
            )
        )
    except Exception as exc:
        return {"status": "failed", "reason": type(exc).__name__}
    return {
        "status": "computed",
        "largest_component_nodes": float(component.number_of_nodes()),
        "fiedler_value": eigenvalues[1],
        "spectral_gap": eigenvalues[1] - eigenvalues[0],
    }


def _harmonic_centrality(graph: nx.Graph) -> tuple[dict[uuid.UUID, float], int]:
    node_count = graph.number_of_nodes()
    if node_count <= 10_000:
        return nx.harmonic_centrality(graph), node_count
    sample_size = min(128, node_count)
    sources = sorted(graph.nodes, key=str)[:sample_size]
    values = {node_id: 0.0 for node_id in graph.nodes}
    for source_id in sources:
        for target_id, distance in nx.single_source_shortest_path_length(
            graph, source_id
        ).items():
            if distance:
                values[target_id] += 1.0 / distance
    scale = node_count / sample_size
    return ({node_id: value * scale for node_id, value in values.items()}, sample_size)


def _compute_projection_analytics(graph: nx.Graph | nx.DiGraph) -> dict[str, Any]:
    undirected = graph.to_undirected()
    metrics: dict[str, dict[uuid.UUID, float]] = {}
    strongly_connected: list[set[uuid.UUID]] = []
    if graph.is_directed():
        metrics["in_degree"] = dict(graph.in_degree(weight="weight"))
        metrics["out_degree"] = dict(graph.out_degree(weight="weight"))
        strongly_connected = list(nx.strongly_connected_components(graph))
    else:
        metrics["degree"] = dict(graph.degree(weight="weight"))
    harmonic_sample_size = 0
    if graph.number_of_nodes():
        metrics["pagerank"] = nx.pagerank(graph, weight="weight")
        metrics["harmonic_centrality"], harmonic_sample_size = _harmonic_centrality(
            undirected
        )
        metrics["clustering"] = nx.clustering(undirected, weight="weight")
        if undirected.number_of_edges():
            metrics["core_number"] = {
                node_id: float(value)
                for node_id, value in nx.core_number(undirected).items()
            }
            sample = min(
                graph.number_of_nodes(),
                64 if graph.number_of_nodes() > 100_000 else 256,
            )
            metrics["betweenness_approx"] = nx.betweenness_centrality(
                graph,
                k=sample,
                normalized=True,
                weight=None,
                seed=0,
            )
    articulation = (
        set(nx.articulation_points(undirected))
        if undirected.number_of_edges()
        else set()
    )
    metrics["is_articulation"] = {
        node_id: float(node_id in articulation) for node_id in graph.nodes
    }
    communities = (
        list(nx.community.louvain_communities(undirected, weight="weight", seed=0))
        if undirected.number_of_edges()
        else [{node_id} for node_id in undirected.nodes]
    )
    components = list(nx.connected_components(undirected))
    component_metrics = [
        ("__graph__", "component_count", float(len(components)), {"scc_count": len(strongly_connected)}),
        (
            "__graph__",
            "bridge_count",
            float(len(list(nx.bridges(undirected)))) if undirected.number_of_edges() else 0.0,
            None,
        ),
    ]
    spectral = _spectral_diagnostics(undirected)
    if spectral.get("status") == "computed":
        for metric in ("fiedler_value", "spectral_gap"):
            component_metrics.append(
                ("largest_component", metric, float(spectral[metric]), None)
            )
    return {
        "node_metrics": metrics,
        "communities": communities,
        "component_metrics": component_metrics,
        "diagnostics": {
            "component_count": len(components),
            "scc_count": len(strongly_connected),
            "community_count": len(communities),
            "spectral": spectral,
            "betweenness_sample_size": min(
                graph.number_of_nodes(),
                64 if graph.number_of_nodes() > 100_000 else 256,
            ),
            "harmonic_sample_size": harmonic_sample_size,
        },
    }


def _row_batches(rows: list[dict[str, Any]], size: int = 5000):
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


async def _latest_graph_version(collection_id: uuid.UUID) -> GraphVersion:
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:lock_key))"),
            {"lock_key": f"graph-version:{collection_id}"},
        )
        version = (
            await session.execute(
                select(GraphVersion)
                .where(GraphVersion.collection_id == collection_id)
                .order_by(GraphVersion.version.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if version is None:
            version = GraphVersion(
                collection_id=collection_id,
                version=1,
                status="ready",
                manifest={"source": "legacy_full_graph"},
                published_at=datetime.now(UTC),
            )
            session.add(version)
            await session.flush()
        await session.commit()
        return version


async def _persist_projection(
    version: GraphVersion,
    spec: dict[str, Any],
    graph: nx.Graph | nx.DiGraph,
    analytics: dict[str, Any],
) -> str:
    digest = _projection_spec_hash(spec)
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:lock_key))"),
            {
                "lock_key": (
                    f"graph-projection:{version.id}:{spec['name']}:{digest}"
                )
            },
        )
        existing = (
            await session.execute(
                select(GraphProjectionSnapshot).where(
                    GraphProjectionSnapshot.graph_version_id == version.id,
                    GraphProjectionSnapshot.name == spec["name"],
                    GraphProjectionSnapshot.spec_hash == digest,
                )
            )
        ).scalar_one_or_none()
        if existing is not None and existing.status == "completed":
            return "skipped"
        if existing is not None:
            await session.execute(
                delete(GraphProjectionSnapshot).where(
                    GraphProjectionSnapshot.id == existing.id
                )
            )
        snapshot_id = uuid.uuid4()
        session.add(
            GraphProjectionSnapshot(
                id=snapshot_id,
                graph_version_id=version.id,
                name=spec["name"],
                spec_hash=digest,
                specification=spec,
                status="building",
                node_count=graph.number_of_nodes(),
                edge_count=graph.number_of_edges(),
            )
        )
        await session.flush()
        for metric, values in analytics["node_metrics"].items():
            rows = [
                {
                    "projection_id": snapshot_id,
                    "entity_id": node_id,
                    "metric": metric,
                    "value": value,
                    "rank": rank,
                }
                for node_id, value, rank in _ranked_metric(values)
            ]
            for batch in _row_batches(rows):
                await session.execute(insert(GraphNodeMetric), batch)
        for index, members in enumerate(analytics["communities"], 1):
            community_id = uuid.uuid4()
            session.add(
                GraphCommunity(
                    id=community_id,
                    projection_id=snapshot_id,
                    algorithm="louvain",
                    community_key=f"community-{index}",
                    metadata_json={"size": len(members)},
                )
            )
            await session.flush()
            membership_rows = [
                {
                    "community_id": community_id,
                    "entity_id": node_id,
                    "strength": 1.0,
                }
                for node_id in members
            ]
            for batch in _row_batches(membership_rows):
                await session.execute(insert(GraphCommunityMembership), batch)
        component_rows = [
            {
                "projection_id": snapshot_id,
                "component_key": key,
                "metric": metric,
                "value": value,
                "metadata_json": metadata,
            }
            for key, metric, value, metadata in analytics["component_metrics"]
        ]
        if component_rows:
            await session.execute(insert(GraphComponentMetric), component_rows)
        snapshot = await session.get(GraphProjectionSnapshot, snapshot_id)
        snapshot.status = "completed"
        snapshot.diagnostics = analytics["diagnostics"]
        snapshot.completed_at = datetime.now(UTC)
        await session.commit()
    return "completed"


async def enhance_structural_analytics(
    collection_id: uuid.UUID,
    nodes: list[NodeRecord],
    relationships: list[RelationshipRecord],
) -> dict[str, Any]:
    """Compute and persist deterministic analytics for the latest graph version."""
    version = await _latest_graph_version(collection_id)
    results: dict[str, Any] = {"graph_version": version.version, "projections": {}}
    for spec in _PROJECTION_SPECS:
        graph = _build_projection(nodes, relationships, spec)
        analytics = _compute_projection_analytics(graph)
        status = await _persist_projection(version, spec, graph, analytics)
        results["projections"][spec["name"]] = {
            "status": status,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "diagnostics": analytics["diagnostics"],
        }
    return results


async def analyze_collection_graph(
    collection_id: uuid.UUID,
) -> dict[str, Any]:
    (
        collection,
        nodes,
        relationships,
        assertion_records,
        aliases_by_entity_id,
    ) = await _load_graph_records(collection_id)
    analysis = build_collection_analysis(
        nodes,
        relationships,
    )
    analysis["semantic_communities"] = _build_semantic_communities(
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
            "description": rel.description,
            **(
                {"predicate_properties": rel.predicate_properties}
                if rel.predicate_properties
                else {}
            ),
        }
        for rel in relationships
    ]
    analysis["assertion_records"] = [
        {
            "id": str(record.id),
            "source_concept_id": str(record.source_concept_id),
            "source_concept_name": record.source_concept_name,
            "source_concept_type": record.source_concept_type,
            "target_concept_id": str(record.target_concept_id),
            "target_concept_name": record.target_concept_name,
            "target_concept_type": record.target_concept_type,
            "rel_type": record.rel_type,
            "weight": record.weight,
            "source_role": record.source_role,
            "target_role": record.target_role,
            "source_mention_id": str(record.source_mention_id),
            "target_mention_id": str(record.target_mention_id),
            "assertion_id": str(record.assertion_id) if record.assertion_id else None,
            "assertion_name": record.assertion_name,
            "context_id": str(record.context_id) if record.context_id else None,
            "context_name": record.context_name,
            "source_document": record.source_document,
            "source_sections": list(record.source_sections),
            "description": record.description,
        }
        for record in assertion_records
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


async def enhance_structural_analytics_from_analysis(
    analysis: dict[str, Any],
) -> dict[str, Any]:
    collection_id = uuid.UUID(str(analysis["collection"]["id"]))
    nodes = [
        NodeRecord(
            id=uuid.UUID(str(row["id"])),
            name=str(row["name"]),
            primary_type=str(row.get("primary_type") or ""),
        )
        for row in analysis.get("node_records", [])
    ]
    relationships = [
        RelationshipRecord(
            id=uuid.UUID(str(row["id"])),
            source_id=uuid.UUID(str(row["source_id"])),
            source_name=str(row["source_name"]),
            target_id=uuid.UUID(str(row["target_id"])),
            target_name=str(row["target_name"]),
            rel_type=str(row["rel_type"]),
            weight=int(row.get("weight") or 1),
            description=str(row.get("description") or ""),
            predicate_properties=dict(row.get("predicate_properties") or {}),
        )
        for row in analysis.get("relationship_records", [])
    ]
    return await enhance_structural_analytics(collection_id, nodes, relationships)


async def enhance_semantic_frames(
    analysis: dict[str, Any],
    collection: Collection,
    embedding_provider: EmbeddingProvider,
) -> dict[str, int]:
    """Materialize embedded community navigation frames after graph analytics."""
    frames: list[SemanticFrameInput] = []
    for community in analysis.get("semantic_communities", []):
        member_ids = [uuid.UUID(value) for value in community["member_ids"]]
        member_names = list(community["member_names"])
        predicates = list(community["predicate_families"])
        membership_hash = hashlib.sha256(
            "|".join(sorted(str(value) for value in member_ids)).encode()
        ).hexdigest()
        frame_id = deterministic_frame_id(
            collection.id,
            f"community-summary:{membership_hash}",
        )
        top_names = member_names[:10]
        title = "Community: " + ", ".join(top_names[:4])
        frame_text = (
            f"A semantic community connecting {', '.join(top_names)}. "
            f"Prominent relationship families: {', '.join(predicates) or 'none'}. "
            "Use this frame only to locate relevant propositions; it is not evidence."
        )
        frames.append(
            SemanticFrameInput(
                id=frame_id,
                collection_id=collection.id,
                frame_kind="community_summary",
                title=title,
                frame_text=frame_text,
                content_hash=hashlib.sha256(frame_text.encode()).hexdigest(),
                polarity="unknown",
                modality="navigation",
                executable_status="navigation_only",
                metadata={
                    "member_count": len(member_ids),
                    "predicate_families": predicates,
                },
                arguments=tuple(
                    FrameArgumentInput("member", entity_id, position)
                    for position, entity_id in enumerate(member_ids[:10])
                ),
            )
        )
    persisted, embedded = await replace_community_frames(
        collection,
        embedding_provider,
        frames,
    )
    return {"community_frames": persisted, "community_embeddings": embedded}


async def build_collection_understanding(
    analysis: dict[str, Any],
    llm_provider: LLMProvider | None = None,
    region_batch_size: int = 1,
    on_region_concept: Callable[[dict[str, Any], dict[str, Any]], Awaitable[None]]
    | None = None,
    on_meta_edge: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
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
    assertion_records: list[dict[str, Any]] = list(
        analysis.get("assertion_records") or []
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
    if assertion_records:
        candidate_regions.extend(_build_facet_candidate_regions(assertion_records))
        candidate_regions.extend(
            _build_shared_class_candidate_regions(assertion_records)
        )

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
            description=str(rel.get("description") or ""),
        )
        for rel in relationship_records
        if str(rel.get("id") or "").strip()
    ]
    if analysis_nodes and analysis_relationships:
        dynamic_regions, role_profiles = _build_dynamic_anchor_regions(
            analysis_nodes,
            analysis_relationships,
            role_count=100,
            min_bucket_size=5,
            anchors_per_role=1,
        )
        candidate_regions.extend(dynamic_regions)
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

    if on_progress is not None:
        await on_progress(len(candidate_regions), 0)

    induced_concepts = list(fallback_concepts)
    streamed_regions = False
    if llm_provider and not isinstance(llm_provider, LocalEchoLLMProvider) and candidate_regions:
        concept_schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "concept_type": {
                    "type": "string",
                    "enum": [
                        "referent_facet",
                        "shared_class",
                        "relation_pattern",
                        "process_flow",
                        "tension",
                        "composite_identity",
                        "theme",
                        "role",
                        "concept",
                    ],
                },
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
                        f"count={edge.get('relationship_count', 1)}"
                        f"; assertion={edge.get('assertion_name') or 'n/a'}"
                        f"; evidence={edge.get('description') or 'n/a'})"
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
                "The candidate may be a referent facet, shared class, dynamic anchor neighborhood, or role-similarity clique.\n"
                "Use concept_type=referent_facet for a specific context-supported role, identity, capacity, or aspect of one anchor entity.\n"
                "Use concept_type=shared_class for sibling entities that instantiate the same category, type, or class.\n"
                "Use concept_type=relation_pattern for a repeated relationship shape across different entities.\n"
                "Use concept_type=process_flow for ordered mechanisms, workflows, or causal sequences.\n"
                "Use concept_type=tension for opposing forces, constraints, tradeoffs, or contradictions.\n"
                "Use concept_type=composite_identity for a stable entity identity made from multiple facets.\n"
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
        completed_regions = 0

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
                completed_regions += 1
                if on_progress is not None:
                    await on_progress(len(candidate_regions), completed_regions)
        streamed_regions = True
    region_concepts: list[dict[str, Any]] = [
        {"region": region, "concept": concept}
        for region, concept in zip(candidate_regions, induced_concepts, strict=False)
    ]
    if not streamed_regions:
        completed_regions = 0
        for region_entry in region_concepts:
            if on_region_concept is not None:
                await on_region_concept(
                    region_entry["region"],
                    region_entry["concept"],
                )
            completed_regions += 1
            if on_progress is not None:
                await on_progress(len(candidate_regions), completed_regions)
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
        meta_rel_type = _meta_relation_type_for_evidence(top_rel_types)
        created_edge = add_edge(
            source_concept_id,
            target_concept_id,
            rel_type=meta_rel_type,
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

    evidence_buckets: dict[tuple[str, str], set[str]] = defaultdict(set)
    concept_evidence: dict[str, tuple[set[str], set[str]]] = {}
    for concept_id, region_ids in concept_region_ids_by_id.items():
        assertion_ids: set[str] = set()
        context_names: set[str] = set()
        for region_id in region_ids:
            region = region_lookup.get(region_id)
            if not region:
                continue
            region_assertions, region_contexts = _region_evidence_keys(region)
            assertion_ids.update(region_assertions)
            context_names.update(region_contexts)
        concept_evidence[concept_id] = (assertion_ids, context_names)
        for assertion_id in assertion_ids:
            evidence_buckets[("assertion", assertion_id)].add(concept_id)
        for context_name in context_names:
            evidence_buckets[("context", context_name)].add(concept_id)

    co_occurrence_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for (evidence_kind, evidence_key), concept_ids in evidence_buckets.items():
        ordered_ids = sorted(concept_ids)
        for index, source_concept_id in enumerate(ordered_ids):
            for target_concept_id in ordered_ids[index + 1 :]:
                pair_key = (source_concept_id, target_concept_id)
                bucket = co_occurrence_pairs.setdefault(
                    pair_key,
                    {
                        "source_id": source_concept_id,
                        "target_id": target_concept_id,
                        "assertions": set(),
                        "contexts": set(),
                    },
                )
                if evidence_kind == "assertion":
                    bucket["assertions"].add(evidence_key)
                else:
                    bucket["contexts"].add(evidence_key)

    ranked_co_occurrences = sorted(
        co_occurrence_pairs.values(),
        key=lambda item: (
            len(item["assertions"]) + len(item["contexts"]),
            len(item["assertions"]),
            item["source_id"],
            item["target_id"],
        ),
        reverse=True,
    )
    for bucket in ranked_co_occurrences:
        if len(
            [edge for edge in edges if "__EVIDENCED_BY__" not in edge["id"]]
        ) >= max_deterministic_meta_edges:
            break
        source_concept_id = str(bucket["source_id"])
        target_concept_id = str(bucket["target_id"])
        source_label = concept_labels_by_id.get(source_concept_id, source_concept_id)
        target_label = concept_labels_by_id.get(target_concept_id, target_concept_id)
        shared_assertions = sorted(bucket["assertions"])
        shared_contexts = sorted(bucket["contexts"])
        if not shared_assertions and not shared_contexts:
            continue
        edge_source_ids = sorted(
            set(concept_source_ids.get(source_concept_id, []))
            | set(concept_source_ids.get(target_concept_id, []))
        )
        context_text = ", ".join(shared_contexts[:4]) or "none"
        assertion_text = ", ".join(shared_assertions[:4]) or "none"
        created_edge = add_edge(
            source_concept_id,
            target_concept_id,
            rel_type="CO_OCCURS_WITH",
            description=(
                f"{source_label} and {target_label} are grounded in shared "
                f"context/assertion evidence. Contexts: {context_text}. "
                f"Assertions: {assertion_text}."
            ),
            source_ids=edge_source_ids,
            keywords=["CO_OCCURS_WITH", *shared_contexts[:3]],
        )
        if created_edge is not None and on_meta_edge is not None:
            await on_meta_edge(created_edge)

    return {
        "nodes": nodes,
        "edges": edges,
        "chunks": chunks,
        "regions": region_concepts,
        "candidate_region_count": len(candidate_regions),
    }

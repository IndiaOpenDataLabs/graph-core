"""Relationship-type vocabulary for multi-dimensional graph retrieval.

Each edge in `graph_relationships` / FalkorDB carries a `rel_type` that
selects which dimension of the graph it belongs to. `RELATES_TO` is the
generic fallback that the LLM extractor and any pre-existing data fall
back to; the rest of the vocabulary is domain-specific so the query path
can run per-dimension BFS and merge the results.

A *dimension* in the query layer is just a subset of rel_types. By
default all dimensions are active. `dimension_weights` lets callers
nudge which dimensions matter for a given collection.
"""
from __future__ import annotations

from typing import Final

REL_TYPE_GENERIC: Final = "RELATES_TO"
DEFAULT_REL_TYPE: Final = REL_TYPE_GENERIC
MAX_REL_TYPE_LEN: Final = 64

# Domains: which rel_types are *expected* in each. Used only as a
# vocabulary hint for the LLM extractor; the storage layer is
# dimension-agnostic and accepts any string.
DOMAIN_VOCAB: Final[dict[str, list[str]]] = {
    "general": [
        "RELATES_TO",
        "EXPLAINS",
        "MENTIONED_IN",
        "IS_AN_EXAMPLE_OF",
        "IS_ANALOGY_OF",
        "CAUSES",
        "PART_OF",
        "CONTRADICTS",
        "SUPPORTS",
        "REFERENCES",
    ],
    "books": [
        "RELATES_TO",
        "EXPLAINS",
        "MENTIONED_IN",
        "QUOTES",
        "CITES",
        "IS_AN_EXAMPLE_OF",
        "IS_ANALOGY_OF",
        "CONTRASTS_WITH",
        "SUPPORTS",
        "ELABORATES",
    ],
    "code": [
        "RELATES_TO",
        "CALLS",
        "IMPORTS",
        "DEFINES",
        "IMPLEMENTS",
        "EXTENDS",
        "DEPENDS_ON",
        "TESTS",
        "DOCUMENTS",
        "REFERENCES",
    ],
    "personal": [
        "RELATES_TO",
        "REMEMBERS",
        "MENTIONED",
        "EXPLAINS_TO",
        "OPINION_ABOUT",
        "PREFERS",
        "DECIDED",
        "OWNS",
    ],
}

ALL_DOMAINS: Final[tuple[str, ...]] = tuple(DOMAIN_VOCAB.keys())


def rel_types_for_domain(domain: str | None) -> list[str]:
    """Vocabulary hint list for a given ingestion domain."""
    if not domain:
        return list(DOMAIN_VOCAB["general"])
    return list(DOMAIN_VOCAB.get(domain, DOMAIN_VOCAB["general"]))


def normalize_rel_type(value: str | None) -> str:
    """Coerce a candidate rel_type to a safe, upper-snake string.

    Falls back to ``RELATES_TO`` for missing/empty/oversize input so the
    column is never NULL at the storage layer.
    """
    if not value:
        return DEFAULT_REL_TYPE
    cleaned = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    if not cleaned:
        return DEFAULT_REL_TYPE
    if len(cleaned) > MAX_REL_TYPE_LEN:
        cleaned = cleaned[:MAX_REL_TYPE_LEN]
    if not cleaned[0].isalpha():
        cleaned = "R_" + cleaned
    return cleaned

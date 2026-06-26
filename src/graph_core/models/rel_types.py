"""Relationship-type vocabulary for multi-dimensional graph retrieval.

Each edge in `graph_relationships` / FalkorDB carries a `rel_type` that
selects which dimension of the graph it belongs to. `RELATES_TO` is the
generic fallback that the LLM extractor and any pre-existing data fall
back to; the rest of the vocabulary is domain-specific so the query path
can run per-dimension BFS and merge the results.

A *dimension* in the query layer is just a subset of rel_types. By
default all dimensions are active. `dimension_weights` lets callers
nudge which dimensions matter for a given collection.

Domain-specific vocabularies are now defined in ``domain_config.py``.
This module re-exports them for backward compatibility.
"""
from __future__ import annotations

from typing import Final

from graph_core.models.domain_config import (
    ALL_DOMAIN_NAMES,
    get_domain_config,
)

REL_TYPE_GENERIC: Final = "RELATES_TO"
DEFAULT_REL_TYPE: Final = REL_TYPE_GENERIC
MAX_REL_TYPE_LEN: Final = 64

# Backward-compatible alias — reads from domain_config registry.
ALL_DOMAINS: Final[tuple[str, ...]] = tuple(ALL_DOMAIN_NAMES)


def rel_types_for_domain(domain: str | None) -> list[str]:
    """Vocabulary hint list for a given ingestion domain."""
    return list(get_domain_config(domain).rel_types)


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


def relationship_embedding_text(
    source_name: str,
    target_name: str,
    rel_type: str | None,
    description: str,
    keywords: list[str] | None = None,
) -> str:
    """Canonical text used for relationship embeddings.

    The rel_type is included explicitly so different semantic
    dimensions between the same endpoints can separate in vector space.
    """
    clean_rel_type = normalize_rel_type(rel_type)
    keyword_text = ", ".join(k.strip() for k in (keywords or []) if k.strip())
    text = (
        f"Relationship type: {clean_rel_type}. "
        f"Source: {source_name}. "
        f"Target: {target_name}. "
        f"Description: {description}"
    )
    if keyword_text:
        text += f" Keywords: {keyword_text}."
    return text

"""Helpers for deriving human-readable FalkorDB graph names."""

from __future__ import annotations

import re
import uuid

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _slugify_collection_name(name: str) -> str:
    slug = _NON_ALNUM_RE.sub("_", (name or "").strip().lower()).strip("_")
    if not slug:
        return "collection"
    return slug[:48].strip("_") or "collection"


def collection_graph_name(
    *,
    collection_id: uuid.UUID,
    collection_name: str,
) -> str:
    """Return a readable, unique FalkorDB graph name for a collection."""
    slug = _slugify_collection_name(collection_name)
    return f"collection_{slug}_{collection_id.hex[:8]}"


def legacy_collection_graph_name(collection_id: uuid.UUID) -> str:
    """Return the legacy collection graph name based only on UUID."""
    return f"collection_{collection_id.hex}"

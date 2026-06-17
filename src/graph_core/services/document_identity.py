"""Helpers for stable document identity across ingestion and reingestion."""

from __future__ import annotations

import uuid
from pathlib import PurePosixPath


def normalize_document_path(document_path: str) -> str:
    """Return a stable, slash-separated path token for a document."""
    normalized = str(document_path or "").strip().replace("\\", "/")
    if not normalized:
        return ""
    normalized = PurePosixPath(normalized).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def document_namespace_id(collection_id: uuid.UUID) -> uuid.UUID:
    """Derive a stable namespace UUID for a collection."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"graph-core:collection:{collection_id}")


def document_id_for_path(collection_id: uuid.UUID, document_path: str) -> uuid.UUID:
    """Derive a stable document UUID from collection + normalized path."""
    normalized = normalize_document_path(document_path)
    if not normalized:
        raise ValueError("document_path is required to derive a document_id")
    return uuid.uuid5(document_namespace_id(collection_id), normalized)


def document_id_for_chunk(collection_id: uuid.UUID, chunk_hash: str) -> uuid.UUID:
    """Derive a stable document UUID for a standalone chunk."""
    normalized = str(chunk_hash or "").strip().lower()
    if not normalized:
        raise ValueError("chunk_hash is required to derive a chunk-scoped document_id")
    return uuid.uuid5(document_namespace_id(collection_id), f"chunk:{normalized}")

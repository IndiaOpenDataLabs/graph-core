"""Helpers for replaying namespace FalkorDB graph-name migrations."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from graph_core.models.collection import Collection
from graph_core.storage.graph_names import (
    collection_graph_name,
    legacy_collection_graph_name,
)
from graph_core.storage.graph_storage import FalkorDBGraphStorage


def load_collection_graph_payloads(
    session: Session,
) -> list[tuple[str, UUID, list[str]]]:
    """Load the collection graph rename payloads from the sync Alembic session."""
    result = session.execute(select(Collection))
    payloads: list[tuple[str, UUID, list[str]]] = []
    for collection in result.scalars().all():
        current_graph_name = collection_graph_name(
            namespace_id=collection.namespace_id,
            collection_id=collection.id,
            collection_name=collection.name,
        )
        old_graph_names = [
            collection_graph_name(
                collection_id=collection.id,
                collection_name=collection.name,
            ),
            legacy_collection_graph_name(collection.id),
        ]
        payloads.append(
            (
                current_graph_name,
                collection.namespace_id,
                [name for name in old_graph_names if name != current_graph_name],
            )
        )
    return payloads


async def replay_collection_graph_names(
    payloads: list[tuple[str, UUID, list[str]]],
) -> None:
    """Replay the collection graph rename pass."""
    for current_graph_name, namespace_id, old_graph_names in payloads:
        current_storage = FalkorDBGraphStorage(current_graph_name)
        current_exists = await current_storage.exists()
        current_node_count = await current_storage.node_count() if current_exists else 0

        for old_graph_name in old_graph_names:
            old_storage = FalkorDBGraphStorage(old_graph_name)
            if not await old_storage.exists():
                continue
            if current_exists:
                old_node_count = await old_storage.node_count()
                if current_node_count == 0 and old_node_count > 0:
                    await current_storage.drop()
                    if await old_storage.rename(current_graph_name):
                        break
                elif old_node_count == 0:
                    await old_storage.drop()
                continue
            if await old_storage.rename(current_graph_name):
                break

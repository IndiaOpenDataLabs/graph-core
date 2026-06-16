"""Rename FalkorDB graphs to the namespace-prefixed collection naming scheme."""

from __future__ import annotations

import asyncio

from sqlalchemy import select

from graph_core.database import AsyncSessionLocal
from graph_core.models.namespace import Namespace
from graph_core.services import auth_service
from graph_core.services.graph import GraphService


async def _main() -> None:
    service = GraphService()
    results = await service.migrate_all_collection_graph_names()
    for row in results:
        print(f"{row['collection_id']} {row['collection_name']} -> {row['graph_name']}")

    async with AsyncSessionLocal() as session:
        namespaces = await session.scalars(
            select(Namespace).order_by(Namespace.created_at.asc())
        )
        for namespace in namespaces:
            (
                _,
                credential,
                secret,
            ) = await auth_service.ensure_namespace_falkordb_credential(
                session,
                str(namespace.id),
            )
            if secret:
                print(
                    f"{namespace.id} {namespace.name} credential={credential.id} "
                    f"username={credential.label} created"
                )
            else:
                print(
                    f"{namespace.id} {namespace.name} credential={credential.id} "
                    f"username={credential.label} reused"
                )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()

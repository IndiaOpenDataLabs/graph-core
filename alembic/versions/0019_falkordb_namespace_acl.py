"""backfill namespace falkordb acl and migrate graph names

Revision ID: 0019_falkordb_namespace_acl
Revises: 0018_query_enhance_jobs
Create Date: 2026-06-17
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select

from graph_core.database import AsyncSessionLocal
from graph_core.models.namespace import Namespace
from graph_core.services import auth_service
from graph_core.services.graph import GraphService

# revision identifiers, used by Alembic.
revision = "0019_falkordb_namespace_acl"
down_revision = "0018_query_enhance_jobs"
branch_labels = None
depends_on = None


async def _upgrade_async() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Namespace).order_by(Namespace.created_at.asc())
        )
        namespaces = list(result.scalars().all())
        for namespace in namespaces:
            await auth_service.ensure_namespace_falkordb_credential(
                session,
                str(namespace.id),
            )

    service = GraphService()
    await service.migrate_all_collection_graph_names()


def upgrade() -> None:
    asyncio.run(_upgrade_async())


def downgrade() -> None:
    # Data migration is not safely reversible.
    pass

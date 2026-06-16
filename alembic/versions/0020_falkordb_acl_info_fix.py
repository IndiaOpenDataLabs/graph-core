"""replay namespace falkordb acl provisioning with info permission

Revision ID: 0020_falkordb_acl_info_fix
Revises: 0019_falkordb_namespace_acl
Create Date: 2026-06-17
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select

from graph_core.database import AsyncSessionLocal
from graph_core.models.namespace import Namespace
from graph_core.services import auth_service

# revision identifiers, used by Alembic.
revision = "0020_falkordb_acl_info_fix"
down_revision = "0019_falkordb_namespace_acl"
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


def upgrade() -> None:
    asyncio.run(_upgrade_async())


def downgrade() -> None:
    # Data migration is not safely reversible.
    pass

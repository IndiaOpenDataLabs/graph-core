"""backfill namespace falkordb acl and migrate graph names

Revision ID: 0019_falkordb_namespace_acl
Revises: 0018_query_enhance_jobs
Create Date: 2026-06-17
"""

from __future__ import annotations

import asyncio

from sqlalchemy.orm import Session

from graph_core.migrations.falkordb_acl import (
    load_namespace_acl_payloads,
    replay_namespace_acl_payloads,
)
from graph_core.migrations.falkordb_graph_names import (
    load_collection_graph_payloads,
    replay_collection_graph_names,
)

# revision identifiers, used by Alembic.
revision = "0019_falkordb_namespace_acl"
down_revision = "0018_query_enhance_jobs"
branch_labels = None
depends_on = None


async def _upgrade_async() -> None:
    bind = op.get_bind()
    with Session(bind=bind) as session:
        acl_payloads = load_namespace_acl_payloads(session)
        graph_payloads = load_collection_graph_payloads(session)

    await replay_namespace_acl_payloads(acl_payloads)
    await replay_collection_graph_names(graph_payloads)


def upgrade() -> None:
    asyncio.run(_upgrade_async())


def downgrade() -> None:
    # Data migration is not safely reversible.
    pass

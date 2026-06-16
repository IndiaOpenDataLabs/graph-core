"""assign namespace FalkorDB db keyspaces and replay graphs

Revision ID: 0023_falkordb_namespace_db
Revises: 0022_falkordb_graph_name_replay
Create Date: 2026-06-17
"""

from __future__ import annotations

from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from alembic import op
from graph_core.models.collection import Collection
from graph_core.models.namespace import Namespace
from graph_core.services.auth_service import _sync_namespace_falkordb_metadata

# revision identifiers, used by Alembic.
revision = "0023_falkordb_namespace_db"
down_revision = "0022_falkordb_graph_name_replay"
branch_labels = None
depends_on = None


async def _upgrade_async() -> None:
    bind = op.get_bind()
    with Session(bind=bind) as session:
        result = session.execute(select(Namespace).order_by(Namespace.created_at.asc()))
        namespaces = list(result.scalars().all())
        counts_result = session.execute(
            select(
                Collection.namespace_id,
                func.count(Collection.id),
            ).group_by(Collection.namespace_id)
        )
        collection_counts = {
            namespace_id: int(count)
            for namespace_id, count in counts_result.all()
        }
        used_dbs: set[int] = set()
        ordered_namespaces = sorted(
            namespaces,
            key=lambda namespace: (
                -collection_counts.get(namespace.id, 0),
                namespace.created_at or datetime.min.replace(tzinfo=UTC),
            ),
        )
        for namespace in ordered_namespaces:
            db = namespace.falkordb_db
            if db is None:
                db = 0
                while db in used_dbs:
                    db += 1
                namespace.falkordb_db = db
            used_dbs.add(int(db))
            _sync_namespace_falkordb_metadata(namespace)
        session.commit()


def upgrade() -> None:
    op.add_column(
        "namespaces",
        sa.Column("falkordb_db", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_namespaces_falkordb_db",
        "namespaces",
        ["falkordb_db"],
        unique=True,
    )
    import asyncio

    asyncio.run(_upgrade_async())


def downgrade() -> None:
    # Data migration is not safely reversible.
    op.drop_index("ix_namespaces_falkordb_db", table_name="namespaces")
    op.drop_column("namespaces", "falkordb_db")

"""replay FalkorDB collection graph-name migration

Revision ID: 0022_falkordb_graph_name_replay
Revises: 0021_falkordb_acl_exists_fix
Create Date: 2026-06-17
"""

from __future__ import annotations

import asyncio

from sqlalchemy.orm import Session

from alembic import op
from graph_core.migrations.falkordb_acl import (
    load_namespace_acl_payloads,
    replay_namespace_acl_payloads,
)
from graph_core.migrations.falkordb_graph_names import (
    load_collection_graph_payloads,
    replay_collection_graph_names,
)

# revision identifiers, used by Alembic.
revision = "0022_falkordb_graph_name_replay"
down_revision = "0021_falkordb_acl_exists_fix"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    with Session(bind=bind) as session:
        acl_payloads = load_namespace_acl_payloads(session)
        payloads = load_collection_graph_payloads(session)
    asyncio.run(replay_namespace_acl_payloads(acl_payloads))
    asyncio.run(replay_collection_graph_names(payloads))


def downgrade() -> None:
    # Data migration is not safely reversible.
    pass

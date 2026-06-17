"""replay namespace falkordb acl provisioning with browser probe grants

Revision ID: 0024_falkordb_acl_getuser_fix
Revises: 0023_falkordb_namespace_db
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

# revision identifiers, used by Alembic.
revision = "0024_falkordb_acl_getuser_fix"
down_revision = "0023_falkordb_namespace_db"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    with Session(bind=bind) as session:
        payloads = load_namespace_acl_payloads(session)
    asyncio.run(replay_namespace_acl_payloads(payloads))


def downgrade() -> None:
    # Replay-only data migration; no safe downgrade.
    return None

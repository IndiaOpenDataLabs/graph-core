"""merge migration heads

Revision ID: 0017_merge_migration_heads
Revises: 0016_drop_ns_api_keys, 9f1a2b3c4d5e
Create Date: 2026-06-14
"""

from __future__ import annotations

# revision identifiers, used by Alembic.
revision = "0017_merge_migration_heads"
down_revision = ("0016_drop_ns_api_keys", "9f1a2b3c4d5e")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

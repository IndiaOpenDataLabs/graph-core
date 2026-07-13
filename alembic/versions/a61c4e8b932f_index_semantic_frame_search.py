"""index semantic frame search

Revision ID: a61c4e8b932f
Revises: f17a2c9d4e61
Create Date: 2026-07-13
"""

from __future__ import annotations

from alembic import op

revision = "a61c4e8b932f"
down_revision = "f17a2c9d4e61"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_graph_semantic_frames_search ON graph_semantic_frames "
        "USING gin (to_tsvector('simple', "
        "coalesce(title, '') || ' ' || coalesce(frame_text, '')))"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_graph_semantic_frames_search",
        table_name="graph_semantic_frames",
    )

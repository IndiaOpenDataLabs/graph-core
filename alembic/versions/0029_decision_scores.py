"""Separate native confidence, source support, and identity decision audit.

Existing mixed-unit weights are deliberately NOT converted to confidence.
No aliases or graph endpoints are automatically rewritten.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0029_decision_scores"
down_revision = "0028_add_raw_extraction_identity"
branch_labels = None
depends_on = None


def upgrade():
    for table in ("entity_descriptions", "relationship_descriptions"):
        op.add_column(table, sa.Column("confidence", sa.Float(), nullable=True))
        op.add_column(table, sa.Column("score_metadata", sa.JSON(), nullable=True))
        op.add_column(table, sa.Column("source_evidence", sa.JSON(), nullable=True))
    op.add_column(
        "graph_relationships", sa.Column("confidence", sa.Float(), nullable=True)
    )
    op.add_column(
        "graph_relationships", sa.Column("support_count", sa.Integer(), nullable=True)
    )
    op.add_column(
        "graph_relationships", sa.Column("score_metadata", sa.JSON(), nullable=True)
    )
    op.add_column(
        "entity_aliases", sa.Column("identity_decision", sa.JSON(), nullable=True)
    )
    op.create_table(
        "entity_resolution_decisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "collection_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("incoming_name", sa.String(256), nullable=False),
        sa.Column("source_chunk_hash", sa.String(64), nullable=True),
        sa.Column("source_evidence", sa.JSON(), nullable=False),
        sa.Column("decision", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_entity_resolution_decisions_collection_id",
        "entity_resolution_decisions",
        ["collection_id"],
    )


def downgrade():
    op.drop_table("entity_resolution_decisions")
    op.drop_column("entity_aliases", "identity_decision")
    for column in ("score_metadata", "support_count", "confidence"):
        op.drop_column("graph_relationships", column)
    for table in ("entity_descriptions", "relationship_descriptions"):
        for column in ("source_evidence", "score_metadata", "confidence"):
            op.drop_column(table, column)

"""add_llm_profile_id_to_collections

Revision ID: b433a5717427
Revises: 0006_light_rag_ingest_enum
Create Date: 2026-05-28 08:54:56.770622
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b433a5717427'
down_revision = '0006_light_rag_ingest_enum'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('collections', sa.Column('llm_profile_id', sa.UUID(), nullable=True))
    op.create_index(op.f('ix_collections_llm_profile_id'), 'collections', ['llm_profile_id'], unique=False)
    op.create_foreign_key(None, 'collections', 'profiles', ['llm_profile_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    op.drop_constraint(None, 'collections', type_='foreignkey')
    op.drop_index(op.f('ix_collections_llm_profile_id'), table_name='collections')
    op.drop_column('collections', 'llm_profile_id')

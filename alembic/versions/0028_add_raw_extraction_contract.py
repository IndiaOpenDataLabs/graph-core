"""add extraction contract to raw chunk extractions

Raw extractions are cached so an identical chunk is never re-extracted. The
cache key was (chunk_content_hash, collection_id, domain), which says nothing
about the prompt, schema, or parser shape that produced the payload. A future
contract change would therefore keep serving payloads written under the old
contract, silently degrading extraction instead of failing.

This revision adds ``extraction_contract`` and moves it into the unique key, so
payloads written under different contracts coexist and a lookup can require the
contract it is actually running. Existing rows are backfilled from
``extraction_model`` so a code-domain payload is not re-discovered as generic
(and the other way round), which would force needless re-extraction on the next
ingest.

Revision ID: 0028_add_raw_extraction_contract
Revises: 0027_add_chunk_cancelled_status
Create Date: 2026-09-15
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0028_add_raw_extraction_contract"
down_revision = "0027_add_chunk_cancelled_status"
branch_labels = None
depends_on = None

GENERIC_CONTRACT = "generic-endpoints-v0"
CODE_CONTRACT = "code-taxonomy-v0"


def upgrade() -> None:
    op.add_column(
        "raw_chunk_extractions",
        sa.Column(
            "extraction_contract",
            sa.String(64),
            nullable=False,
            server_default=GENERIC_CONTRACT,
        ),
    )

    # Rows written under the code taxonomy carry extraction_model = 'domain:code'.
    # Backfilling from it keeps every existing row resolvable under the contract
    # that actually produced it.
    op.execute(
        sa.text(
            "UPDATE raw_chunk_extractions "
            "SET extraction_contract = :code_contract "
            "WHERE extraction_model = 'domain:code'"
        ).bindparams(code_contract=CODE_CONTRACT)
    )

    op.drop_constraint(
        "uq_raw_chunk_extractions_hash_collection",
        "raw_chunk_extractions",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_raw_chunk_extractions_hash_collection_contract",
        "raw_chunk_extractions",
        ["chunk_content_hash", "collection_id", "extraction_contract"],
    )


def downgrade() -> None:
    # Collapsing back to (hash, collection) can collide across contracts; keep
    # the newest row per (hash, collection) so the constraint can be restored.
    op.execute(
        sa.text(
            "DELETE FROM raw_chunk_extractions a "
            "USING raw_chunk_extractions b "
            "WHERE a.ctid < b.ctid "
            "AND a.chunk_content_hash = b.chunk_content_hash "
            "AND a.collection_id = b.collection_id"
        )
    )
    op.drop_constraint(
        "uq_raw_chunk_extractions_hash_collection_contract",
        "raw_chunk_extractions",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_raw_chunk_extractions_hash_collection",
        "raw_chunk_extractions",
        ["chunk_content_hash", "collection_id"],
    )
    op.drop_column("raw_chunk_extractions", "extraction_contract")

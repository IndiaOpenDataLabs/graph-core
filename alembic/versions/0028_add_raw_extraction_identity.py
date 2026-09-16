"""add extraction cache identity to raw chunk extractions

Raw extractions are cached so an identical chunk is never re-extracted. The cache
identity was (chunk_content_hash, collection_id) with a domain-derived
``extraction_model`` column, which says nothing about the prompt, schema, or
domain configuration that produced the payload.

Two consequences this revision removes:

* a prompt or schema change keeps serving payloads written under the previous
  family, silently degrading extraction instead of failing;
* distinct domains resolve to distinct prompts but share one row, so the domain
  that is written second cannot be stored at all, and a lookup filtered by domain
  misses it forever.

Adds ``extraction_contract`` (code-shipped prompt/schema family) and
``prompt_fingerprint`` (the domain configuration the run resolved) and moves both
into the unique key.

``extraction_contract`` is backfilled from ``extraction_model``: it is a property
of the deployed code, so it can be inferred for historical rows, and inferring it
prevents a needless full re-extraction of cached code-domain content.
``prompt_fingerprint`` cannot be reconstructed for historical rows, so they are
recorded as ``legacy`` and re-extracted once on the next ingest. That is the
correct cost of not serving a payload whose prompt inputs are unknown.

Revision ID: 0028_add_raw_extraction_identity
Revises: 0027_add_chunk_cancelled_status
Create Date: 2026-09-15
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0028_add_raw_extraction_identity"
down_revision = "0027_add_chunk_cancelled_status"
branch_labels = None
depends_on = None

GENERIC_CONTRACT = "generic-endpoints-v0"
CODE_CONTRACT = "code-taxonomy-v0"
LEGACY_FINGERPRINT = "legacy"

OLD_CONSTRAINT = "uq_raw_chunk_extractions_hash_collection"
NEW_CONSTRAINT = "uq_raw_chunk_extractions_identity"
TABLE = "raw_chunk_extractions"


def upgrade() -> None:
    op.add_column(
        TABLE,
        sa.Column(
            "extraction_contract",
            sa.String(64),
            nullable=False,
            server_default=GENERIC_CONTRACT,
        ),
    )
    op.add_column(
        TABLE,
        sa.Column(
            "prompt_fingerprint",
            sa.String(32),
            nullable=False,
            server_default=LEGACY_FINGERPRINT,
        ),
    )

    # Rows written under the code taxonomy carry extraction_model = 'domain:code'.
    # Backfilling keeps every existing row resolvable under the family that
    # actually produced it.
    op.execute(
        sa.text(
            "UPDATE raw_chunk_extractions "
            "SET extraction_contract = :code_contract "
            "WHERE extraction_model = 'domain:code'"
        ).bindparams(code_contract=CODE_CONTRACT)
    )

    op.drop_constraint(OLD_CONSTRAINT, TABLE, type_="unique")
    op.create_unique_constraint(
        NEW_CONSTRAINT,
        TABLE,
        [
            "chunk_content_hash",
            "collection_id",
            "extraction_contract",
            "prompt_fingerprint",
        ],
    )


def downgrade() -> None:
    # Collapsing back to (hash, collection) collides across identities, so keep
    # one row per (hash, collection) before restoring the old constraint.
    op.execute(
        sa.text(
            "DELETE FROM raw_chunk_extractions a "
            "USING raw_chunk_extractions b "
            "WHERE a.ctid < b.ctid "
            "AND a.chunk_content_hash = b.chunk_content_hash "
            "AND a.collection_id = b.collection_id"
        )
    )
    op.drop_constraint(NEW_CONSTRAINT, TABLE, type_="unique")
    op.create_unique_constraint(
        OLD_CONSTRAINT, TABLE, ["chunk_content_hash", "collection_id"]
    )
    op.drop_column(TABLE, "prompt_fingerprint")
    op.drop_column(TABLE, "extraction_contract")

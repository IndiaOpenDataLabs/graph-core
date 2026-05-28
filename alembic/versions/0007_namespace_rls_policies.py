"""namespace row-level security policies

Revision ID: 0007_namespace_rls_policies
Revises: b433a5717427
Create Date: 2026-05-28
"""  # noqa: E501

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# ruff: noqa: E501
# revision identifiers, used by Alembic.
revision = "0007_namespace_rls_policies"
down_revision = "b433a5717427"
branch_labels = None
depends_on = None


# Tables with direct namespace_id column
direct_tables = [
    "credentials",
    "profiles",
    "collections",
    "jobs",
]

# Tables scoped via collection_id -> collections.namespace_id
via_collection_tables = [
    "ingestion_records",
    "graph_entities",
    "graph_relationships",
    "raw_chunk_extractions",
]

# Tables scoped via entity_id -> graph_entities.collection_id -> collections.namespace_id
via_entity_tables = [
    "entity_descriptions",
    "entity_aliases",
    "entity_types",
]

# Tables scoped via relationship_id -> graph_relationships.collection_id -> collections.namespace_id
via_relationship_tables = [
    "relationship_descriptions",
]

# Tables scoped via job_id -> jobs.namespace_id
via_job_tables = [
    "job_events",
    "ingestion_chunks",
]


def upgrade() -> None:
    # Helper function: when app.current_namespace_id is set (via SET LOCAL),
    # enforce namespace filter. When not set (workers, admin connections),
    # allow all rows — app-layer isolation still applies.
    op.execute(sa.text("""
        CREATE OR REPLACE FUNCTION ns_check(namespace_id uuid)
        RETURNS boolean AS $$
            DECLARE
                ctx uuid;
            BEGIN
                ctx := current_setting('app.current_namespace_id', true)::uuid;
                RETURN ctx IS NULL OR ctx = namespace_id;
            END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """))

    # Helper function for collection-scoped tables
    op.execute(sa.text("""
        CREATE OR REPLACE FUNCTION ns_check_via_collection(collection_id uuid)
        RETURNS boolean AS $$
            DECLARE
                ctx uuid;
            BEGIN
                ctx := current_setting('app.current_namespace_id', true)::uuid;
                IF ctx IS NULL THEN
                    RETURN true;
                END IF;
                RETURN EXISTS (
                    SELECT 1 FROM collections c
                    WHERE c.id = collection_id
                      AND c.namespace_id = ctx
                );
            END;
        $$ LANGUAGE plpgsql STABLE;
    """))

    # Enable RLS on every namespace-scoped table
    for table in (
        direct_tables
        + via_collection_tables
        + via_entity_tables
        + via_relationship_tables
        + via_job_tables
    ):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")

    # --- Policies for tables with direct namespace_id ---
    for table in direct_tables:
        _create_policy(op, table, "ns_select", "SELECT", "ns_check(namespace_id)")
        _create_policy(op, table, "ns_insert", "INSERT", "ns_check(namespace_id)")
        _create_policy(op, table, "ns_update", "UPDATE", "ns_check(namespace_id)")
        _create_policy(op, table, "ns_delete", "DELETE", "ns_check(namespace_id)")

    # --- Policies for tables scoped via collection_id ---
    for table in via_collection_tables:
        _create_policy(op, table, "ns_select", "SELECT", "ns_check_via_collection(collection_id)")
        _create_policy(op, table, "ns_insert", "INSERT", "ns_check_via_collection(collection_id)")
        _create_policy(op, table, "ns_update", "UPDATE", "ns_check_via_collection(collection_id)")
        _create_policy(op, table, "ns_delete", "DELETE", "ns_check_via_collection(collection_id)")

    # --- Policies for tables scoped via entity_id ---
    op.execute("""
        CREATE POLICY ns_select ON entity_descriptions FOR SELECT
            USING (ns_check_via_collection(
                (SELECT e.collection_id FROM graph_entities e WHERE e.id = entity_descriptions.entity_id)
            ))
    """)
    op.execute("""
        CREATE POLICY ns_insert ON entity_descriptions FOR INSERT
            WITH CHECK (ns_check_via_collection(
                (SELECT e.collection_id FROM graph_entities e WHERE e.id = entity_descriptions.entity_id)
            ))
    """)
    op.execute("""
        CREATE POLICY ns_update ON entity_descriptions FOR UPDATE
            USING (ns_check_via_collection(
                (SELECT e.collection_id FROM graph_entities e WHERE e.id = entity_descriptions.entity_id)
            ))
    """)
    op.execute("""
        CREATE POLICY ns_delete ON entity_descriptions FOR DELETE
            USING (ns_check_via_collection(
                (SELECT e.collection_id FROM graph_entities e WHERE e.id = entity_descriptions.entity_id)
            ))
    """)

    for table in ("entity_aliases", "entity_types"):
        op.execute(f"""
            CREATE POLICY ns_select ON {table} FOR SELECT
                USING (ns_check_via_collection(
                    (SELECT e.collection_id FROM graph_entities e WHERE e.id = {table}.entity_id)
                ))
        """)
        op.execute(f"""
            CREATE POLICY ns_insert ON {table} FOR INSERT
                WITH CHECK (ns_check_via_collection(
                    (SELECT e.collection_id FROM graph_entities e WHERE e.id = {table}.entity_id)
                ))
        """)
        op.execute(f"""
            CREATE POLICY ns_update ON {table} FOR UPDATE
                USING (ns_check_via_collection(
                    (SELECT e.collection_id FROM graph_entities e WHERE e.id = {table}.entity_id)
                ))
        """)
        op.execute(f"""
            CREATE POLICY ns_delete ON {table} FOR DELETE
                USING (ns_check_via_collection(
                    (SELECT e.collection_id FROM graph_entities e WHERE e.id = {table}.entity_id)
                ))
        """)

    # --- Policies for tables scoped via relationship_id ---
    op.execute("""
        CREATE POLICY ns_select ON relationship_descriptions FOR SELECT
            USING (ns_check_via_collection(
                (SELECT r.collection_id FROM graph_relationships r WHERE r.id = relationship_descriptions.relationship_id)
            ))
    """)
    op.execute("""
        CREATE POLICY ns_insert ON relationship_descriptions FOR INSERT
            WITH CHECK (ns_check_via_collection(
                (SELECT r.collection_id FROM graph_relationships r WHERE r.id = relationship_descriptions.relationship_id)
            ))
    """)
    op.execute("""
        CREATE POLICY ns_update ON relationship_descriptions FOR UPDATE
            USING (ns_check_via_collection(
                (SELECT r.collection_id FROM graph_relationships r WHERE r.id = relationship_descriptions.relationship_id)
            ))
    """)
    op.execute("""
        CREATE POLICY ns_delete ON relationship_descriptions FOR DELETE
            USING (ns_check_via_collection(
                (SELECT r.collection_id FROM graph_relationships r WHERE r.id = relationship_descriptions.relationship_id)
            ))
    """)

    # --- Policies for tables scoped via job_id ---
    for table in via_job_tables:
        op.execute(f"""
            CREATE POLICY ns_select ON {table} FOR SELECT
                USING (ns_check((SELECT j.namespace_id FROM jobs j WHERE j.id = {table}.job_id)))
        """)
        op.execute(f"""
            CREATE POLICY ns_insert ON {table} FOR INSERT
                WITH CHECK (ns_check((SELECT j.namespace_id FROM jobs j WHERE j.id = {table}.job_id)))
        """)
        op.execute(f"""
            CREATE POLICY ns_update ON {table} FOR UPDATE
                USING (ns_check((SELECT j.namespace_id FROM jobs j WHERE j.id = {table}.job_id)))
        """)
        op.execute(f"""
            CREATE POLICY ns_delete ON {table} FOR DELETE
                USING (ns_check((SELECT j.namespace_id FROM jobs j WHERE j.id = {table}.job_id)))
        """)


def downgrade() -> None:
    # Drop all policies
    all_tables = (
        direct_tables
        + via_collection_tables
        + via_entity_tables
        + via_relationship_tables
        + via_job_tables
    )
    for table in all_tables:
        for suffix in ("ns_select", "ns_insert", "ns_update", "ns_delete"):
            try:
                op.execute(f"DROP POLICY IF EXISTS {suffix} ON {table}")
            except Exception:
                pass

    # Disable RLS
    for table in all_tables:
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    # Drop helper functions
    op.execute("DROP FUNCTION IF EXISTS ns_check(uuid)")
    op.execute("DROP FUNCTION IF EXISTS ns_check_via_collection(uuid)")


def _create_policy(op, table: str, name: str, cmd: str, using: str) -> None:
    if cmd == "INSERT":
        op.execute(
            f"CREATE POLICY {name} ON {table} FOR {cmd} WITH CHECK ({using})"
        )
    else:
        op.execute(
            f"CREATE POLICY {name} ON {table} FOR {cmd} USING ({using})"
        )

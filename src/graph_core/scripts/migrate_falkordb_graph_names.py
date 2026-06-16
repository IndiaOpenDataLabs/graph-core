"""Rename FalkorDB graphs and backfill namespace credentials."""

from __future__ import annotations


def main() -> None:
    from graph_core.scripts.migrate_namespace_falkordb_graph_names import main as _main

    _main()


if __name__ == "__main__":
    main()

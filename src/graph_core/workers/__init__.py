"""Dramatiq workers — thin execution wrappers around GraphService."""

from graph_core.workers.ingestion import run_ingestion

__all__ = ["run_ingestion"]

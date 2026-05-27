"""Dramatiq workers — thin execution wrappers around GraphService."""

# Configure broker before importing actors
import graph_core.broker  # noqa: F401

from graph_core.workers.ingestion import run_ingestion

__all__ = ["run_ingestion"]

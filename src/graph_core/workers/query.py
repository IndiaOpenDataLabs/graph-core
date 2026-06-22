"""Query workers — Dramatiq actors for async collection querying."""

import logging
import uuid

import dramatiq

import graph_core.broker  # noqa: F401
from graph_core.services.graph import GraphService

logger = logging.getLogger(__name__)


@dramatiq.actor(
    queue_name="query_jobs",
    max_retries=0,
    max_age=3600000,
    time_limit=float("inf"),
)
async def run_query(job_id: str):
    """Execute an async query job and persist the answer in job payload."""
    service = GraphService()
    await service.run_query_job(uuid.UUID(job_id))

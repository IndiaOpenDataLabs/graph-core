"""Enhance workers — Dramatiq actors for async collection understanding."""

import logging
import uuid

import dramatiq

import graph_core.broker  # noqa: F401
from graph_core.services.graph import GraphService

logger = logging.getLogger(__name__)


@dramatiq.actor(
    queue_name="enhance_jobs",
    max_retries=3,
    max_age=7200000,
    time_limit=float("inf"),
)
async def run_enhance(job_id: str):
    """Execute an async enhance job and persist the summary in job payload."""
    service = GraphService()
    await service.run_enhance_job(uuid.UUID(job_id))

#!/usr/bin/env python3
"""Operational helper for inspecting and recovering ingestion jobs.

Examples:
  PYTHONPATH=src .venv/bin/python docs/HELPER.py active-jobs
  PYTHONPATH=src .venv/bin/python docs/HELPER.py show-job <job_id>
  PYTHONPATH=src .venv/bin/python docs/HELPER.py finalize-failed <job_id>
  PYTHONPATH=src .venv/bin/python docs/HELPER.py requeue-failed <job_id>
  PYTHONPATH=src .venv/bin/python docs/HELPER.py requeue-processing <job_id>
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqlalchemy import text  # noqa: E402

import graph_core.workers  # noqa: E402,F401
from graph_core.database import AsyncSessionLocal  # noqa: E402
from graph_core.workers.ingestion import run_chunk  # noqa: E402


async def active_jobs() -> None:
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                text(
                    """
                    select id, job_type, status, progress_percent,
                           chunks_total, chunks_completed, created_at
                    from jobs
                    where status in ('pending', 'running')
                    order by created_at desc
                    """
                )
            )
        ).fetchall()
    for row in rows:
        print(row)


async def show_job(job_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                text(
                    """
                    select id, job_type, status, progress_percent, chunks_total,
                           chunks_completed, started_at, completed_at, error
                    from jobs
                    where id = :jid
                    """
                ),
                {"jid": job_id},
            )
        ).fetchone()
        print("job", row)
        counts = (
            await session.execute(
                text(
                    """
                    select status, count(*)
                    from ingestion_chunks
                    where job_id = :jid
                    group by status
                    order by status
                    """
                ),
                {"jid": job_id},
            )
        ).fetchall()
        print("chunk_counts", counts)
        failed = (
            await session.execute(
                text(
                    """
                    select chunk_index, left(coalesce(error, ''), 300)
                    from ingestion_chunks
                    where job_id = :jid and status = 'failed'
                    order by chunk_index
                    """
                ),
                {"jid": job_id},
            )
        ).fetchall()
        for row in failed:
            print("failed", row)


async def finalize_failed(job_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(
            text(
                """
                update jobs
                set status = 'failed',
                    completed_at = :now,
                    error = coalesce(
                        nullif(error, ''),
                        'One or more chunks failed during ingestion'
                    )
                where id = :jid
                  and status = 'running'
                  and not exists (
                    select 1 from ingestion_chunks
                    where job_id = :jid and status in ('pending', 'processing')
                  )
                  and exists (
                    select 1 from ingestion_chunks
                    where job_id = :jid and status = 'failed'
                  )
                """
            ),
            {"jid": job_id, "now": datetime.now(UTC)},
        )
        await session.commit()
    await show_job(job_id)


async def _reset_chunks(job_id: uuid.UUID, from_status: str, note: str) -> int:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                f"""
                update ingestion_chunks
                set status = 'pending',
                    error = coalesce(error, '') ||
                        case
                            when coalesce(error, '') = '' then ''
                            else E'\\n'
                        end ||
                        :note,
                    completed_at = null
                where job_id = :jid
                  and status = '{from_status}'
                """
            ),
            {"jid": job_id, "note": note},
        )
        await session.execute(
            text(
                """
                update jobs
                set status = 'running',
                    error = null
                where id = :jid
                """
            ),
            {"jid": job_id},
        )
        await session.commit()
        return int(result.rowcount or 0)


async def _chunk_indices(job_id: uuid.UUID, status: str) -> list[int]:
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                text(
                    """
                    select chunk_index
                    from ingestion_chunks
                    where job_id = :jid and status = :status
                    order by chunk_index
                    """
                ),
                {"jid": job_id, "status": status},
            )
        ).fetchall()
    return [int(row[0]) for row in rows]


async def requeue_failed(job_id: uuid.UUID) -> None:
    failed_indices = await _chunk_indices(job_id, "failed")
    count = await _reset_chunks(job_id, "failed", "[manual requeue from HELPER.py]")
    for chunk_index in failed_indices:
        run_chunk.send(str(job_id), chunk_index)
    print(f"reset_failed={count} dispatched={len(failed_indices)}")
    await show_job(job_id)


async def requeue_processing(job_id: uuid.UUID) -> None:
    processing_indices = await _chunk_indices(job_id, "processing")
    count = await _reset_chunks(
        job_id,
        "processing",
        "[manual reset from processing in HELPER.py]",
    )
    for chunk_index in processing_indices:
        run_chunk.send(str(job_id), chunk_index)
    print(f"reset_processing={count} dispatched={len(processing_indices)}")
    await show_job(job_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("active-jobs")
    for name in (
        "show-job",
        "finalize-failed",
        "requeue-failed",
        "requeue-processing",
    ):
        cmd = sub.add_parser(name)
        cmd.add_argument("job_id", type=uuid.UUID)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.command == "active-jobs":
        await active_jobs()
    elif args.command == "show-job":
        await show_job(args.job_id)
    elif args.command == "finalize-failed":
        await finalize_failed(args.job_id)
    elif args.command == "requeue-failed":
        await requeue_failed(args.job_id)
    elif args.command == "requeue-processing":
        await requeue_processing(args.job_id)


if __name__ == "__main__":
    asyncio.run(main())

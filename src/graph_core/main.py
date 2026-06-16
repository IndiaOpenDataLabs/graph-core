"""Graph Core - FastAPI application entry point."""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from graph_core.api import chats, collections, ingest, jobs, namespaces, platform, query
from graph_core.database import AsyncSessionLocal, current_namespace_id
from graph_core.mcp.server import admin_mcp, user_mcp
from graph_core.migrations.falkordb_acl import (
    load_namespace_acl_payloads,
    replay_namespace_acl_payloads,
)

logger = logging.getLogger(__name__)


async def _replay_namespace_falkordb_acls() -> None:
    async with AsyncSessionLocal() as session:
        payloads = await session.run_sync(load_namespace_acl_payloads)
    if not payloads:
        logger.info("No namespace FalkorDB ACLs to replay on startup")
        return
    await replay_namespace_acl_payloads(payloads)
    logger.info("Replayed %d namespace FalkorDB ACLs on startup", len(payloads))


@asynccontextmanager
async def lifespan(app: FastAPI):
    import graph_core.workers  # noqa: F401

    del app
    async with admin_mcp.session_manager.run(), user_mcp.session_manager.run():
        await _replay_namespace_falkordb_acls()
        yield


app = FastAPI(
    title="Graph Core",
    description="AI-native knowledge infrastructure platform",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def set_namespace_context(request: Request, call_next):
    """Extract X-Namespace-ID header and set request-scoped contextvar.

    The contextvar is read by NamespacedAsyncSession.begin() to set the
    Postgres app.current_namespace_id session variable for RLS policies.

    New clients should use Authorization: Bearer <ns_key> header.
    """
    ns_header = request.headers.get("x-namespace-id")
    if ns_header:
        try:
            current_namespace_id.set(uuid.UUID(ns_header))
        except ValueError:
            pass
    response = await call_next(request)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# Mount routers
app.include_router(namespaces.router)
app.include_router(platform.router)
app.include_router(collections.router)
app.include_router(chats.router)
app.include_router(ingest.router)
app.include_router(jobs.router)
app.include_router(query.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}

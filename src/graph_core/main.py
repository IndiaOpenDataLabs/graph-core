"""Graph Core — FastAPI application entry point."""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from graph_core.api import chats, collections, ingest, jobs, namespaces, platform, query
from graph_core.database import current_namespace_id
from graph_core.mcp.server import mcp_server_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    import graph_core.workers  # noqa: F401

    del app
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

    DEPRECATED: New clients should use Authorization: Bearer <ns_key> header.
    This middleware remains for backward compatibility.
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


app.mount("/mcp", mcp_server_app())

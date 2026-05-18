"""Graph Core — FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from graph_core.api import collections, ingest, jobs, platform, query
from graph_core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: import workers to register Dramatiq actors
    import graph_core.workers  # noqa: F401
    yield
    # Shutdown


app = FastAPI(
    title="Graph Core",
    description="AI-native knowledge infrastructure platform",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Middleware ──


@app.middleware("http")
async def namespace_from_header(request: Request, call_next):
    """Extract namespace from X-Namespace-ID header for all routes."""
    request.state.namespace_id = request.headers.get("X-Namespace-ID", "")
    response = await call_next(request)
    return response


# ── Error handlers ──


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# ── Mount routers ──

# Platform control plane
app.include_router(platform.router)

# Resources
app.include_router(collections.router)
app.include_router(ingest.router)
app.include_router(jobs.router)
app.include_router(query.router)

# ── Health ──


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}

"""Graph Core — FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from graph_core.api import collections, ingest, jobs, platform, query
from graph_core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    import graph_core.workers  # noqa: F401
    yield


app = FastAPI(
    title="Graph Core",
    description="AI-native knowledge infrastructure platform",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# Mount routers
app.include_router(platform.router)
app.include_router(collections.router)
app.include_router(ingest.router)
app.include_router(jobs.router)
app.include_router(query.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}

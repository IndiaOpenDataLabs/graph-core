"""FastAPI router — ingest endpoints."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from graph_core.api.dependencies import get_namespace_id
from graph_core.services.graph import GraphService
from graph_core.workers.ingestion import run_ingestion


class IngestChunkRequest(BaseModel):
    text: str


class IngestChunkResponse(BaseModel):
    chunk_hash: str
    entity_count: int
    relationship_count: int


class IngestDocRequest(BaseModel):
    text: str


class IngestDocResponse(BaseModel):
    job_id: str
    status: str


router = APIRouter(tags=["ingest"])
service = GraphService()


@router.post("/collections/{collection_id}/ingest/chunk", response_model=IngestChunkResponse)
async def ingest_chunk(
    body: IngestChunkRequest,
    collection_id: uuid.UUID,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
) -> IngestChunkResponse:
    try:
        result = await service.ingest_chunk(body.text, collection_id, namespace_id)
        return IngestChunkResponse(**result.__dict__)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/collections/{collection_id}/ingest/doc", response_model=IngestDocResponse)
async def ingest_document(
    body: IngestDocRequest,
    collection_id: uuid.UUID,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
) -> IngestDocResponse:
    try:
        result = await service.enqueue_document_ingestion(body.text, collection_id, namespace_id)
        run_ingestion.send(str(result.job_id))
        return IngestDocResponse(job_id=str(result.job_id), status=result.status)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

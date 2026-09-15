"""FastAPI router — ingest endpoints."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from graph_core.api.auth import get_namespace_id
from graph_core.api.provider_errors import service_http_errors
from graph_core.services.graph import GraphService
from graph_core.workers.ingestion import run_ingestion


class IngestRequest(BaseModel):
    text: str
    domain: str | None = None
    document_path: str | None = None


class IngestChunkResponse(BaseModel):
    chunk_hash: str
    entity_count: int
    relationship_count: int


class IngestDocResponse(BaseModel):
    job_id: str
    status: str


router = APIRouter(tags=["ingest"])
service = GraphService()


@router.post(
    "/collections/{collection_id}/ingest/chunk",
    response_model=IngestChunkResponse,
)
async def ingest_chunk(
    body: IngestRequest,
    collection_id: uuid.UUID,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
) -> IngestChunkResponse:
    with service_http_errors():
        result = await service.ingest_chunk(
            body.text,
            collection_id,
            namespace_id,
            domain=body.domain,
            document_path=body.document_path,
        )
        return IngestChunkResponse(**result.__dict__)


@router.post(
    "/collections/{collection_id}/ingest/doc",
    response_model=IngestDocResponse,
)
async def ingest_document(
    body: IngestRequest,
    collection_id: uuid.UUID,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
) -> IngestDocResponse:
    with service_http_errors():
        result = await service.enqueue_document_ingestion(
            body.text,
            collection_id,
            namespace_id,
            domain=body.domain,
            document_path=body.document_path,
        )
        run_ingestion.send(str(result.job_id))
        return IngestDocResponse(job_id=str(result.job_id), status=result.status)

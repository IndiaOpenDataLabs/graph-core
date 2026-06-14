"""FastAPI router — query endpoint."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from graph_core.api.auth import get_namespace_id
from graph_core.services.graph import GraphService
from graph_core.workers.query import run_query


class QueryRequest(BaseModel):
    question: str
    mode: str | None = None
    llm_profile_id: uuid.UUID | None = None
    chat_id: uuid.UUID | None = None


class QueryResponse(BaseModel):
    job_id: str
    type: str
    status: str
    collection_id: str
    namespace_id: str


router = APIRouter(tags=["query"])
service = GraphService()


@router.post(
    "/collections/{collection_id}/query",
    response_model=QueryResponse,
    status_code=202,
)
async def query_collection(
    body: QueryRequest,
    collection_id: uuid.UUID,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
) -> QueryResponse:
    try:
        result = await service.enqueue_query_job(
            body.question,
            collection_id,
            namespace_id,
            body.mode,
            llm_profile_id=body.llm_profile_id,
            chat_id=body.chat_id,
        )
        run_query.send(result["job_id"])
        return QueryResponse(**result)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

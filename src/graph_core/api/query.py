"""FastAPI router — query endpoint."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from graph_core.api.dependencies import get_namespace_id
from graph_core.services.graph import GraphService


class QueryRequest(BaseModel):
    question: str
    mode: str | None = None


class QueryResponse(BaseModel):
    response: str
    entities_used: list[str]
    relationships_used: list[str]
    mode: str


router = APIRouter(tags=["query"])
service = GraphService()


@router.post("/collections/{collection_id}/query", response_model=QueryResponse)
async def query_collection(
    body: QueryRequest,
    collection_id: uuid.UUID,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
) -> QueryResponse:
    try:
        result = await service.query(body.question, collection_id, namespace_id, body.mode)
        return QueryResponse(**result.__dict__)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

"""FastAPI router — query endpoint."""

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
    namespace_id: uuid.UUID,
) -> QueryResponse:
    """Query a collection's knowledge graph."""
    try:
        result = await service.query(body.question, collection_id, namespace_id, body.mode)
        return QueryResponse(**result.__dict__)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

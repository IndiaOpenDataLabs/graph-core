"""FastAPI router — collection CRUD."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from graph_core.database import AsyncSession, get_session
from graph_core.services.graph import GraphService


class CreateCollectionRequest(BaseModel):
    name: str
    strategy: str = "vector"
    embedding_profile_id: uuid.UUID | None = None
    default_query_mode: str | None = None


class CreateCollectionResponse(BaseModel):
    id: str
    name: str
    strategy: str
    namespace_id: str
    embedding_profile_id: str | None
    default_query_mode: str | None


router = APIRouter(prefix="/collections", tags=["collections"])
service = GraphService()


@router.post("/")
async def create_collection(
    body: CreateCollectionRequest,
    namespace_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> CreateCollectionResponse:
    """Create a new collection bound to the namespace."""
    try:
        collection = await service.create_collection(
            name=body.name,
            namespace_id=namespace_id,
            strategy=body.strategy,  # type: ignore[arg-type]
            embedding_profile_id=body.embedding_profile_id,
            default_query_mode=body.default_query_mode,
        )
        return _to_response(collection)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/")
async def list_collections(
    namespace_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[CreateCollectionResponse]:
    collections = await service.list_collections(namespace_id)
    return [_to_response(c) for c in collections]


def _to_response(c) -> CreateCollectionResponse:
    return CreateCollectionResponse(
        id=str(c.id),
        name=c.name,
        strategy=c.strategy,
        namespace_id=str(c.namespace_id),
        embedding_profile_id=str(c.embedding_profile_id) if c.embedding_profile_id else None,
        default_query_mode=c.default_query_mode,
    )

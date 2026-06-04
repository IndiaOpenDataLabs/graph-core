"""FastAPI router — collection CRUD."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from graph_core.api.auth import get_namespace_id
from graph_core.database import AsyncSession, get_session
from graph_core.services.graph import GraphService


class CreateCollectionRequest(BaseModel):
    name: str
    strategy: str = "vector"
    embedding_profile_id: uuid.UUID | None = None
    llm_profile_id: uuid.UUID | None = None
    default_query_mode: str | None = None
    gleaning_passes: int = 1


class UpdateCollectionRequest(BaseModel):
    name: str | None = None
    strategy: str | None = None
    embedding_profile_id: uuid.UUID | None = None
    llm_profile_id: uuid.UUID | None = None
    default_query_mode: str | None = None
    gleaning_passes: int | None = None
    clear_llm_profile: bool = False
    clear_default_query_mode: bool = False


class CollectionResponse(BaseModel):
    id: str
    name: str
    strategy: str
    namespace_id: str
    embedding_profile_id: str | None
    llm_profile_id: str | None
    default_query_mode: str | None
    gleaning_passes: int


class EnhanceCollectionResponse(BaseModel):
    status: str
    collection_id: str
    graph_name: str
    node_count: int
    edge_count: int
    chunk_count: int


router = APIRouter(prefix="/collections", tags=["collections"])
service = GraphService()


@router.post("/")
async def create_collection(
    body: CreateCollectionRequest,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> CollectionResponse:
    try:
        collection = await service.create_collection(
            name=body.name,
            namespace_id=namespace_id,
            strategy=body.strategy,  # type: ignore[arg-type]
            embedding_profile_id=body.embedding_profile_id,
            llm_profile_id=body.llm_profile_id,
            default_query_mode=body.default_query_mode,
            gleaning_passes=body.gleaning_passes,
        )
        return _to_response(collection)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/")
async def list_collections(
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[CollectionResponse]:
    collections = await service.list_collections(namespace_id)
    return [_to_response(c) for c in collections]


@router.patch("/{collection_id}")
async def update_collection(
    collection_id: uuid.UUID,
    body: UpdateCollectionRequest,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> CollectionResponse:
    del session
    try:
        collection = await service.update_collection(
            collection_id=collection_id,
            namespace_id=namespace_id,
            name=body.name,
            strategy=body.strategy,  # type: ignore[arg-type]
            embedding_profile_id=body.embedding_profile_id,
            llm_profile_id=body.llm_profile_id,
            default_query_mode=body.default_query_mode,
            gleaning_passes=body.gleaning_passes,
            clear_llm_profile=body.clear_llm_profile,
            clear_default_query_mode=body.clear_default_query_mode,
        )
        return _to_response(collection)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("/{collection_id}")
async def delete_collection(
    collection_id: uuid.UUID,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, str]:
    del session
    try:
        collection = await service.get_collection(collection_id)
        if collection.namespace_id != namespace_id:
            raise PermissionError(
                f"Collection {collection_id} does not belong to namespace "
                f"{namespace_id}"
            )
        await service.delete_collection(collection_id)
        return {"status": "deleted", "id": str(collection_id)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/{collection_id}/enhance")
async def enhance_collection(
    collection_id: uuid.UUID,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> EnhanceCollectionResponse:
    del session
    try:
        result = await service.build_collection_understanding(
            collection_id=collection_id,
            namespace_id=namespace_id,
        )
        derived_graph = result["derived_graph"]
        return EnhanceCollectionResponse(
            status="enhanced",
            collection_id=str(collection_id),
            graph_name=str(derived_graph["graph_name"]),
            node_count=int(derived_graph["node_count"]),
            edge_count=int(derived_graph["edge_count"]),
            chunk_count=int(derived_graph["chunk_count"]),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


def _to_response(c) -> CollectionResponse:
    return CollectionResponse(
        id=str(c.id),
        name=c.name,
        strategy=c.strategy,
        namespace_id=str(c.namespace_id),
        embedding_profile_id=(
            str(c.embedding_profile_id) if c.embedding_profile_id else None
        ),
        llm_profile_id=str(c.llm_profile_id) if c.llm_profile_id else None,
        default_query_mode=c.default_query_mode,
        gleaning_passes=c.gleaning_passes,
    )

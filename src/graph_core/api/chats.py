"""FastAPI router — chat session CRUD for query memory."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from graph_core.api.auth import get_namespace_id
from graph_core.services.graph import GraphService


class CreateChatRequest(BaseModel):
    title: str | None = None


class ChatSessionResponse(BaseModel):
    id: str
    collection_id: str
    title: str | None = None
    turn_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


router = APIRouter(prefix="/collections/{collection_id}/chats", tags=["chats"])
service = GraphService()


@router.post("/", response_model=ChatSessionResponse)
async def create_chat_session(
    collection_id: uuid.UUID,
    body: CreateChatRequest,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
) -> ChatSessionResponse:
    try:
        chat = await service.create_chat_session(
            collection_id=collection_id,
            namespace_id=namespace_id,
            title=body.title,
        )
        return ChatSessionResponse(
            id=str(chat.id),
            collection_id=str(chat.collection_id),
            title=chat.title,
            turn_count=0,
            created_at=chat.created_at.isoformat() if chat.created_at else None,
            updated_at=chat.updated_at.isoformat() if chat.updated_at else None,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/", response_model=list[ChatSessionResponse])
async def list_chat_sessions(
    collection_id: uuid.UUID,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
    limit: int = Query(default=20, ge=1, le=100),
) -> list[ChatSessionResponse]:
    try:
        rows = await service.list_chat_sessions(
            collection_id=collection_id,
            namespace_id=namespace_id,
            limit=limit,
        )
        return [ChatSessionResponse(**row) for row in rows]
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

"""FastAPI router — namespace management and key rotation."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from graph_core.api.auth import AuthContext, get_auth_context
from graph_core.database import AsyncSession, get_session
from graph_core.services import auth_service


class CreateNamespaceRequest(BaseModel):
    name: str


class CreateNamespaceResponse(BaseModel):
    id: str
    name: str
    api_key: str


class NamespaceResponse(BaseModel):
    id: str
    name: str
    api_key_prefix: str | None
    created_at: str | None


class RotateKeyResponse(BaseModel):
    api_key: str


router = APIRouter(prefix="/platform/namespaces", tags=["namespaces"])


@router.post("/", response_model=CreateNamespaceResponse)
async def create_namespace(
    body: CreateNamespaceRequest,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> CreateNamespaceResponse:
    """Create a new namespace. Requires admin JWT."""
    if not auth.is_admin:
        raise HTTPException(status_code=403, detail="Admin key required to create namespaces")

    result = await auth_service.create_namespace_with_key(
        session,
        name=body.name,
    )
    return CreateNamespaceResponse(
        id=str(result.namespace.id),
        name=result.namespace.name,
        api_key=result.api_key,
    )


@router.get("/", response_model=list[NamespaceResponse])
async def list_namespaces(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[NamespaceResponse]:
    """List all namespaces. Requires admin JWT."""
    if not auth.is_admin:
        raise HTTPException(status_code=403, detail="Admin key required to list namespaces")

    namespaces = await auth_service.list_namespaces(session)
    return [
        NamespaceResponse(
            id=str(ns.id),
            name=ns.name,
            api_key_prefix=ns.api_key_prefix,
            created_at=ns.created_at.isoformat() if ns.created_at else None,
        )
        for ns in namespaces
    ]


@router.get("/me", response_model=NamespaceResponse)
async def get_current_namespace(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> NamespaceResponse:
    """Get the current namespace. Works with namespace API key."""
    if not auth.namespace:
        raise HTTPException(status_code=400, detail="Not authenticated with a namespace key")

    return NamespaceResponse(
        id=str(auth.namespace.id),
        name=auth.namespace.name,
        api_key_prefix=auth.namespace.api_key_prefix,
        created_at=auth.namespace.created_at.isoformat() if auth.namespace.created_at else None,
    )


@router.post("/{namespace_id}/rotate-key", response_model=RotateKeyResponse)
async def rotate_namespace_key(
    namespace_id: str,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> RotateKeyResponse:
    """Rotate a namespace's API key. Requires admin JWT."""
    if not auth.is_admin:
        raise HTTPException(status_code=403, detail="Admin key required to rotate keys")

    result = await auth_service.rotate_namespace_key(session, namespace_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Namespace {namespace_id} not found")

    return RotateKeyResponse(api_key=result.api_key)

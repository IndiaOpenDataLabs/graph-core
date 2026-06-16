"""FastAPI router — namespace management and user-token minting."""

from __future__ import annotations

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
    falkordb_db: int
    token_type: str
    scope: str
    token: str
    expires_at: str
    credential_id: str
    falkordb_username: str
    falkordb_secret: str
    falkordb_graph_pattern: str


class NamespaceResponse(BaseModel):
    id: str
    name: str
    created_at: str | None


class IssueUserTokenRequest(BaseModel):
    subject: str | None = None
    expires_in_days: int = 365


class IssueUserTokenResponse(BaseModel):
    namespace_id: str
    namespace_name: str
    token_type: str
    scope: str
    token: str
    expires_at: str


router = APIRouter(prefix="/platform/namespaces", tags=["namespaces"])


@router.post("/", response_model=CreateNamespaceResponse)
async def create_namespace(
    body: CreateNamespaceRequest,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> CreateNamespaceResponse:
    """Create a new namespace. Requires admin JWT."""
    if not auth.is_admin:
        raise HTTPException(
            status_code=403, detail="Admin JWT required to create namespaces"
        )

    result = await auth_service.create_namespace(
        session,
        name=body.name,
    )
    (
        credential_ns,
        credential,
        secret,
    ) = await auth_service.provision_namespace_falkordb_credential(
        session,
        str(result.id),
    )
    token_result = await auth_service.issue_namespace_user_token(
        session,
        str(result.id),
        subject="graph-core-cli",
        expires_in_days=365,
    )
    if token_result is None:
        raise HTTPException(status_code=500, detail="Failed to mint namespace token")
    return CreateNamespaceResponse(
        id=str(result.id),
        name=result.name,
        falkordb_db=int(credential_ns.falkordb_db or 0),
        token_type="user",
        scope="graph-core:user",
        token=token_result[1],
        expires_at=token_result[2].isoformat(),
        credential_id=str(credential.id),
        falkordb_username=credential.label or "",
        falkordb_secret=secret,
        falkordb_graph_pattern=f"tenant:{credential_ns.id}:*",
    )


@router.get("/", response_model=list[NamespaceResponse])
async def list_namespaces(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[NamespaceResponse]:
    """List all namespaces. Requires admin JWT."""
    if not auth.is_admin:
        raise HTTPException(
            status_code=403, detail="Admin JWT required to list namespaces"
        )

    namespaces = await auth_service.list_namespaces(session)
    return [
        NamespaceResponse(
            id=str(ns.id),
            name=ns.name,
            created_at=ns.created_at.isoformat() if ns.created_at else None,
        )
        for ns in namespaces
    ]


@router.get("/me", response_model=NamespaceResponse)
async def get_current_namespace(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> NamespaceResponse:
    """Get the current namespace. Works with a user JWT."""
    if not auth.namespace:
        raise HTTPException(
            status_code=400, detail="Not authenticated with a user token"
        )

    return NamespaceResponse(
        id=str(auth.namespace.id),
        name=auth.namespace.name,
        created_at=auth.namespace.created_at.isoformat()
        if auth.namespace.created_at
        else None,
    )


@router.post("/{namespace_id}/issue-user-token", response_model=IssueUserTokenResponse)
async def issue_user_token(
    namespace_id: str,
    body: IssueUserTokenRequest,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> IssueUserTokenResponse:
    """Issue a long-lived namespace-scoped user JWT. Requires admin JWT."""
    if not auth.is_admin:
        raise HTTPException(
            status_code=403, detail="Admin JWT required to issue user tokens"
        )

    if body.expires_in_days <= 0:
        raise HTTPException(status_code=400, detail="expires_in_days must be positive")

    result = await auth_service.issue_namespace_user_token(
        session,
        namespace_id,
        subject=body.subject,
        expires_in_days=body.expires_in_days,
    )
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Namespace {namespace_id} not found"
        )

    return IssueUserTokenResponse(
        namespace_id=str(result[0].id),
        namespace_name=result[0].name,
        token_type="user",
        scope="graph-core:user",
        token=result[1],
        expires_at=result[2].isoformat(),
    )

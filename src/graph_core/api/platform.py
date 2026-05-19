"""FastAPI router — platform control plane (credentials, profiles, capabilities)."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from graph_core.api.dependencies import get_namespace_id
from graph_core.services.platform import PlatformService


class RegisterCredentialRequest(BaseModel):
    provider: str
    secret: str
    label: str | None = None


class RegisterCredentialResponse(BaseModel):
    credential_id: str
    provider: str
    label: str | None


class CreateProfileRequest(BaseModel):
    kind: str
    provider: str
    model: str
    credential_id: uuid.UUID | None = None
    label: str | None = None
    dimensions: int | None = None
    distance_metric: str | None = None


class ProfileResponse(BaseModel):
    profile_id: str
    kind: str
    provider: str
    model: str
    credential_id: str | None
    label: str | None
    dimensions: int | None
    distance_metric: str | None


router = APIRouter(prefix="/platform", tags=["platform"])
service = PlatformService()


@router.get("/capabilities")
async def get_capabilities(
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
) -> dict:
    """Discover platform capabilities for dynamic clients."""
    embedding_profiles = await service.list_profiles(
        namespace_id=namespace_id,
        kind="embedding",
    )
    llm_profiles = await service.list_profiles(
        namespace_id=namespace_id,
        kind="llm",
    )
    return {
        "embedding_profiles": [
            _to_profile_response(profile).model_dump()
            for profile in embedding_profiles
        ],
        "llm_profiles": [
            _to_profile_response(profile).model_dump()
            for profile in llm_profiles
        ],
        "retrieval_strategies": ["vector", "custom_graph_rag", "light_rag"],
        "max_chunk_size": 16000,
    }


@router.post("/credentials", response_model=RegisterCredentialResponse)
async def register_cred(
    body: RegisterCredentialRequest,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
) -> RegisterCredentialResponse:
    """Register encrypted credential. Returns credential_id for profile binding."""
    credential = await service.register_credential(
        namespace_id=namespace_id,
        provider=body.provider,
        secret=body.secret,
        label=body.label,
    )
    return RegisterCredentialResponse(
        credential_id=str(credential.id),
        provider=credential.provider,
        label=credential.label,
    )


@router.post("/profiles", response_model=ProfileResponse)
async def create_profile(
    body: CreateProfileRequest,
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
) -> ProfileResponse:
    try:
        profile = await service.create_profile(
            namespace_id=namespace_id,
            kind=body.kind,
            provider=body.provider,
            model=body.model,
            credential_id=body.credential_id,
            label=body.label,
            dimensions=body.dimensions,
            distance_metric=body.distance_metric,
        )
        return _to_profile_response(profile)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/embedding-profiles", response_model=list[ProfileResponse])
async def list_embedding_profiles(
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
) -> list[ProfileResponse]:
    profiles = await service.list_profiles(
        namespace_id=namespace_id,
        kind="embedding",
    )
    return [_to_profile_response(profile) for profile in profiles]


@router.get("/llm-profiles", response_model=list[ProfileResponse])
async def list_llm_profiles(
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
) -> list[ProfileResponse]:
    profiles = await service.list_profiles(
        namespace_id=namespace_id,
        kind="llm",
    )
    return [_to_profile_response(profile) for profile in profiles]


def _to_profile_response(profile) -> ProfileResponse:
    return ProfileResponse(
        profile_id=str(profile.id),
        kind=profile.kind,
        provider=profile.provider,
        model=profile.model,
        credential_id=str(profile.credential_id) if profile.credential_id else None,
        label=profile.label,
        dimensions=profile.dimensions,
        distance_metric=profile.distance_metric,
    )

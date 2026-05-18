"""FastAPI router — platform control plane (credentials, profiles, capabilities)."""

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class RegisterCredentialRequest(BaseModel):
    provider: str
    secret: str
    label: str | None = None


class CredentialResponse(BaseModel):
    credential_id: str
    provider: str
    label: str | None


router = APIRouter(prefix="/platform", tags=["platform"])


@router.get("/capabilities")
async def get_capabilities() -> dict:
    """Discover platform capabilities for dynamic clients."""
    return {
        "embedding_profiles": [],
        "llm_profiles": [],
        "retrieval_strategies": ["vector", "custom_graph_rag"],
        "max_chunk_size": 16000,
    }


@router.post("/credentials", response_model=CredentialResponse)
async def register_cred(
    body: RegisterCredentialRequest,
    namespace_id: uuid.UUID,
) -> CredentialResponse:
    """Register encrypted credential. Returns credential_id for profile binding."""
    # TODO: encrypt secret, store in Credential model
    raise NotImplementedError("Credential storage pending encryption implementation")

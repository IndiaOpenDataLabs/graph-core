"""Platform control-plane service."""

import uuid

from sqlalchemy import select

from graph_core.config import settings
from graph_core.database import AsyncSessionLocal
from graph_core.models.credential import Credential
from graph_core.models.profile import Profile
from graph_core.services.crypto import CredentialCrypto


class PlatformService:
    def __init__(self):
        self._crypto = CredentialCrypto()

    async def register_credential(
        self,
        *,
        namespace_id: uuid.UUID,
        provider: str,
        secret: str,
        label: str | None = None,
        base_url: str | None = None,
    ) -> Credential:
        async with AsyncSessionLocal() as session:
            credential = Credential(
                namespace_id=namespace_id,
                provider=provider,
                label=label,
                encrypted_secret=self._crypto.encrypt(secret),
                base_url=base_url,
            )
            session.add(credential)
            await session.commit()
            await session.refresh(credential)
            return credential

    async def create_profile(
        self,
        *,
        namespace_id: uuid.UUID,
        kind: str,
        provider: str,
        model: str,
        credential_id: uuid.UUID | None = None,
        label: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        distance_metric: str | None = None,
        max_concurrent_calls: int | None = None,
    ) -> Profile:
        async with AsyncSessionLocal() as session:
            if credential_id is not None:
                credential = await session.get(Credential, credential_id)
                if not credential or credential.namespace_id != namespace_id:
                    raise ValueError("Credential not found in namespace")

            if kind == "embedding":
                if dimensions is None:
                    raise ValueError(
                        "Embedding profiles require dimensions."
                    )
                distance_metric = distance_metric or settings.default_distance_metric
            if max_concurrent_calls is not None and max_concurrent_calls < 1:
                raise ValueError("max_concurrent_calls must be >= 1")

            profile = Profile(
                namespace_id=namespace_id,
                credential_id=credential_id,
                kind=kind,
                provider=provider,
                model=model,
                label=label,
                base_url=base_url,
                dimensions=dimensions,
                distance_metric=distance_metric,
                max_concurrent_calls=max_concurrent_calls,
            )
            session.add(profile)
            await session.commit()
            await session.refresh(profile)
            return profile

    async def list_profiles(
        self,
        *,
        namespace_id: uuid.UUID,
        kind: str | None = None,
    ) -> list[Profile]:
        async with AsyncSessionLocal() as session:
            statement = select(Profile).where(Profile.namespace_id == namespace_id)
            if kind is not None:
                statement = statement.where(Profile.kind == kind)
            result = await session.execute(statement)
            return list(result.scalars().all())

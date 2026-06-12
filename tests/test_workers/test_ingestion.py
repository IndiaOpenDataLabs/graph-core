import uuid

import pytest

from graph_core.models.profile import Profile
from graph_core.workers.ingestion import _resolve_llm_chunk_scope_and_limit


@pytest.mark.asyncio
async def test_resolve_llm_chunk_scope_and_limit_returns_none_for_missing_job():
    assert await _resolve_llm_chunk_scope_and_limit(uuid.uuid4()) is None


@pytest.mark.asyncio
async def test_resolve_llm_chunk_scope_and_limit_uses_collection_profile(
    db_session,
    test_namespace,
):
    from graph_core.models.collection import Collection
    from graph_core.models.job import Job

    profile = Profile(
        namespace_id=test_namespace.id,
        kind="llm",
        provider="openai",
        model="local-llm",
        label="local-llm",
        max_concurrent_calls=3,
    )
    db_session.add(profile)
    await db_session.commit()
    await db_session.refresh(profile)

    collection = Collection(
        namespace_id=test_namespace.id,
        name="profiled",
        strategy="custom_graph_rag",
        embedding_dimensions=256,
        llm_profile_id=profile.id,
    )
    db_session.add(collection)
    await db_session.commit()
    await db_session.refresh(collection)

    job = Job(
        namespace_id=test_namespace.id,
        collection_id=collection.id,
        job_type="ingest_document",
        status="pending",
        payload={"text": "x"},
    )
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)

    scope_limit = await _resolve_llm_chunk_scope_and_limit(job.id)

    assert scope_limit == (str(profile.id), 3)

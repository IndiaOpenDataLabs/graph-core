import asyncio

import pytest

from graph_core import provider_semaphore


class _FakeSemaphore:
    def __init__(self) -> None:
        self.acquire_count = 0
        self.release_started = asyncio.Event()
        self.released = asyncio.Event()
        self.owner_job_ids: list[str | None] = []

    async def acquire(
        self,
        scope: str,
        limit: int,
        owner_job_id: str | None = None,
    ) -> str | None:
        assert scope == "scope"
        assert limit == 1
        self.acquire_count += 1
        self.owner_job_ids.append(owner_job_id)
        return "token"

    async def try_acquire(
        self,
        scope: str,
        limit: int,
        owner_job_id: str | None = None,
    ) -> str | None:
        return await self.acquire(scope, limit, owner_job_id=owner_job_id)

    async def release(self, scope: str, token: str | None, limit: int) -> None:
        assert scope == "scope"
        assert token == "token"
        assert limit == 1
        self.release_started.set()
        await asyncio.sleep(0.05)
        self.released.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("slot_name", "semaphore_name"),
    [
        ("llm_call_slot", "_llm_semaphore"),
        ("embedding_call_slot", "_embedding_semaphore"),
    ],
)
async def test_provider_slot_release_completes_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    slot_name: str,
    semaphore_name: str,
) -> None:
    fake = _FakeSemaphore()
    monkeypatch.setattr(provider_semaphore, semaphore_name, fake)
    slot = getattr(provider_semaphore, slot_name)

    async def _run() -> None:
        async with slot(scope="scope", max_concurrent_calls=1):
            await asyncio.sleep(10)

    task = asyncio.create_task(_run())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(fake.release_started.wait(), timeout=1)
    await asyncio.wait_for(fake.released.wait(), timeout=1)


@pytest.mark.asyncio
async def test_llm_call_slot_reuses_adopted_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSemaphore()
    monkeypatch.setattr(provider_semaphore, "_llm_semaphore", fake)

    token = await provider_semaphore.reserve_llm_call_slot(
        scope="scope",
        max_concurrent_calls=1,
    )
    assert token == "token"

    async with provider_semaphore.adopt_llm_call_slot(
        scope="scope",
        token=token,
        max_concurrent_calls=1,
    ):
        async with provider_semaphore.llm_call_slot(
            scope="scope",
            max_concurrent_calls=1,
        ):
            pass

    await provider_semaphore.release_llm_call_slot(
        scope="scope",
        token=token,
        max_concurrent_calls=1,
    )
    assert fake.acquire_count == 1


@pytest.mark.asyncio
async def test_provider_job_context_tags_direct_provider_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = _FakeSemaphore()
    fake_embedding = _FakeSemaphore()
    monkeypatch.setattr(provider_semaphore, "_llm_semaphore", fake_llm)
    monkeypatch.setattr(provider_semaphore, "_embedding_semaphore", fake_embedding)

    async with provider_semaphore.provider_job_context("job-123"):
        async with provider_semaphore.llm_call_slot(
            scope="scope",
            max_concurrent_calls=1,
        ):
            pass
        async with provider_semaphore.embedding_call_slot(
            scope="scope",
            max_concurrent_calls=1,
        ):
            pass

    assert fake_llm.owner_job_ids == ["job-123"]
    assert fake_embedding.owner_job_ids == ["job-123"]


@pytest.mark.asyncio
async def test_try_reserve_llm_call_slot_uses_explicit_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSemaphore()
    monkeypatch.setattr(provider_semaphore, "_llm_semaphore", fake)

    token = await provider_semaphore.try_reserve_llm_call_slot(
        scope="scope",
        max_concurrent_calls=1,
        owner_job_id="job-456",
    )

    assert token == "token"
    assert fake.owner_job_ids == ["job-456"]

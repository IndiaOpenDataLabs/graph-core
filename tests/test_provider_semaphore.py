import asyncio

import pytest

from graph_core import provider_semaphore


class _FakeSemaphore:
    def __init__(self) -> None:
        self.release_started = asyncio.Event()
        self.released = asyncio.Event()

    async def acquire(self, scope: str, limit: int) -> str | None:
        assert scope == "scope"
        assert limit == 1
        return "token"

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

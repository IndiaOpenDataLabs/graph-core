from types import SimpleNamespace
import uuid

import pytest

from graph_core.services.graph.query import graph_rag, lightrag, vector


class _AsyncSessionContext:
    def __init__(self, session) -> None:
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _FakeSession:
    def __init__(self, profile) -> None:
        self._profile = profile

    async def get(self, model, profile_id):
        del model, profile_id
        return self._profile


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "resolver_name"),
    [
        (vector, "_resolve_llm_provider"),
        (lightrag, "_resolve_llm_provider"),
        (graph_rag, "_resolve_llm_provider"),
    ],
)
async def test_query_resolvers_use_4096_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
    module,
    resolver_name: str,
) -> None:
    namespace_id = uuid.uuid4()
    profile_id = uuid.uuid4()
    profile = SimpleNamespace(
        id=profile_id,
        namespace_id=namespace_id,
        kind="llm",
        provider="openai",
        model="mlx",
        credential_id=uuid.uuid4(),
        base_url="http://localhost:1234/v1",
        max_concurrent_calls=4,
    )
    captured: dict[str, object] = {}

    async def _resolve_credential(session, profile_obj):
        del session, profile_obj
        return "secret", "http://localhost:1234/v1"

    def _get_llm_provider(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(module, "_resolve_credential", _resolve_credential)
    monkeypatch.setattr(module, "get_llm_provider", _get_llm_provider)
    monkeypatch.setattr(module, "AsyncSessionLocal", lambda: _AsyncSessionContext(_FakeSession(profile)))

    resolver = getattr(module, resolver_name)
    provider = await resolver(namespace_id=namespace_id, llm_profile_id=profile_id)

    assert provider is not None
    assert captured["max_output_tokens"] == 4096

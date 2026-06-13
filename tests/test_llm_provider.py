from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

import graph_core.llm.openai_provider as openai_provider


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self) -> None:
        self.create = AsyncMock(return_value=_FakeResponse('{"ok": true}'))


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeAsyncOpenAI:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.chat = _FakeChat()


@asynccontextmanager
async def _noop_llm_call_slot(*args, **kwargs):
    del args, kwargs
    yield


@pytest.mark.asyncio
async def test_chat_sets_default_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(openai_provider, "llm_call_slot", _noop_llm_call_slot)
    monkeypatch.setattr(
        openai_provider.settings,
        "default_llm_max_output_tokens",
        17,
        raising=False,
    )

    provider = openai_provider.OpenAILLMProvider(
        api_key="test-key",
        model="test-model",
    )

    response = await provider.chat([{"role": "user", "content": "hello"}])

    assert response == '{"ok": true}'
    kwargs = provider._client.chat.completions.create.await_args.kwargs
    assert kwargs["max_tokens"] == 17


@pytest.mark.asyncio
async def test_structured_extract_sets_default_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(openai_provider, "llm_call_slot", _noop_llm_call_slot)
    monkeypatch.setattr(
        openai_provider.settings,
        "default_llm_max_output_tokens",
        23,
        raising=False,
    )

    provider = openai_provider.OpenAILLMProvider(
        api_key="test-key",
        model="test-model",
    )

    result = await provider.structured_extract(
        prompt="extract",
        schema={"title": "result"},
    )

    assert result == {"ok": True}
    kwargs = provider._client.chat.completions.create.await_args.kwargs
    assert kwargs["max_tokens"] == 23

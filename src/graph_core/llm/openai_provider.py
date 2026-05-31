"""LLM providers for OpenAI and offline local fallback."""

import json
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from graph_core.llm.interface import LLMProvider
from graph_core.provider_semaphore import llm_call_slot


class LocalEchoLLMProvider(LLMProvider):
    async def chat(self, messages: list[dict]) -> str:
        return next(
            (
                message["content"]
                for message in reversed(messages)
                if message["role"] == "user"
            ),
            "",
        )

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        yield await self.chat(messages)

    async def structured_extract(self, prompt: str, schema: dict) -> dict:
        return {}


class OpenAILLMProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def chat(self, messages: list[dict]) -> str:
        async with llm_call_slot():
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
        return response.choices[0].message.content or ""

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        async with llm_call_slot():
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    async def structured_extract(self, prompt: str, schema: dict) -> dict:
        messages = [{"role": "user", "content": prompt}]
        async with llm_call_slot():
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.get("title", "schema"),
                            "schema": schema,
                        },
                    },
                )
            except Exception:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format={"type": "json"},
                )

        content = response.choices[0].message.content or ""
        if not content:
            return {}
        return json.loads(content)

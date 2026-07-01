"""LLM providers for OpenAI and offline local fallback."""

import json
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from graph_core.config import settings
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
        profile_id: str | None = None,
        max_concurrent_calls: int | None = None,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.openai_request_timeout_seconds,
            max_retries=max(settings.openai_max_retries, 0),
        )
        self._model = model
        self._profile_id = profile_id
        self._max_concurrent_calls = max_concurrent_calls

    @staticmethod
    def _completion_limits() -> dict[str, int]:
        max_tokens = int(settings.llm_max_output_tokens or 0)
        if max_tokens <= 0:
            return {}
        return {"max_tokens": max_tokens}

    async def chat(self, messages: list[dict]) -> str:
        async with llm_call_slot(
            scope=self._profile_id,
            max_concurrent_calls=self._max_concurrent_calls,
        ):
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                **self._completion_limits(),
            )
        return self._response_text(response)

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        async with llm_call_slot(
            scope=self._profile_id,
            max_concurrent_calls=self._max_concurrent_calls,
        ):
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
                **self._completion_limits(),
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    @staticmethod
    def _response_text(response) -> str:
        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        if content.strip():
            return content
        reasoning_content = getattr(message, "reasoning_content", None) or ""
        return reasoning_content

    async def structured_extract(self, prompt: str, schema: dict) -> dict:
        async def _request(extract_prompt: str, force_json: bool = False):
            request_kwargs = {
                "model": self._model,
                "messages": [{"role": "user", "content": extract_prompt}],
                **self._completion_limits(),
            }
            if force_json:
                request_kwargs["response_format"] = {"type": "json_object"}
            else:
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.get("title", "schema"),
                        "schema": schema,
                    },
                }
            return await self._client.chat.completions.create(**request_kwargs)

        async with llm_call_slot(
            scope=self._profile_id,
            max_concurrent_calls=self._max_concurrent_calls,
        ):
            try:
                response = await _request(prompt)
            except Exception:
                response = await _request(prompt, force_json=True)

        content = self._response_text(response)
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            repair_prompt = (
                f"{prompt}\n\n"
                "The previous response was not valid JSON.\n"
                f"Parse error: {error}.\n"
                "Return only valid JSON that matches the requested schema."
            )
            async with llm_call_slot(
                scope=self._profile_id,
                max_concurrent_calls=self._max_concurrent_calls,
            ):
                repair_response = await _request(repair_prompt, force_json=True)
            repair_content = self._response_text(repair_response)
            if not repair_content:
                return {}
            return json.loads(repair_content)

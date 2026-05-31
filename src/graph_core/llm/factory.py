"""LLM provider selection."""

from graph_core.config import settings
from graph_core.llm.interface import LLMProvider
from graph_core.llm.openai_provider import LocalEchoLLMProvider, OpenAILLMProvider
from graph_core.provider_base_url import normalize_provider_base_url


def get_llm_provider(
    *,
    provider_name: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    provider_name = provider_name or settings.default_llm_provider
    model = model or settings.default_llm_model

    if provider_name == "local_echo":
        return LocalEchoLLMProvider()
    if provider_name == "openai":
        effective_api_key = api_key or settings.openai_api_key
        if not effective_api_key:
            return LocalEchoLLMProvider()
        return OpenAILLMProvider(
            api_key=effective_api_key,
            model=model,
            base_url=normalize_provider_base_url(
                base_url or settings.openai_base_url
            ),
        )

    raise ValueError(f"Unsupported llm provider: {provider_name}")

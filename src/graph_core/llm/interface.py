"""LLM provider interface."""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict]) -> str:
        """Send a chat completion and return the assistant response."""
        ...

    @abstractmethod
    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream chat completion tokens."""
        ...

    @abstractmethod
    async def structured_extract(self, prompt: str, schema: dict) -> dict:
        """Use function calling / JSON mode for structured extraction."""
        ...

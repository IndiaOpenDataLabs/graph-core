"""Token-aware chunking for document ingestion."""

from __future__ import annotations

import tiktoken


class TokenChunker:
    def __init__(self, chunk_size_tokens: int, chunk_overlap_tokens: int):
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens cannot be negative")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

        self._chunk_size = chunk_size_tokens
        self._chunk_overlap = chunk_overlap_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str) -> list[str]:
        tokens = self._encoding.encode(text)
        if not tokens:
            return []

        chunks: list[str] = []
        step = self._chunk_size - self._chunk_overlap

        for start in range(0, len(tokens), step):
            token_slice = tokens[start : start + self._chunk_size]
            if not token_slice:
                continue
            chunks.append(self._encoding.decode(token_slice).strip())
            if start + self._chunk_size >= len(tokens):
                break

        return [chunk for chunk in chunks if chunk]

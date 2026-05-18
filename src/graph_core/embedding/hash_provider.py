"""Deterministic local embedding provider for tests and baseline vector search."""

from __future__ import annotations

import hashlib
import math
import re

from graph_core.embedding.interface import EmbeddingProvider


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class HashEmbeddingProvider(EmbeddingProvider):
    """Simple hashed bag-of-words embedding.

    This is not semantically rich, but it gives stable lexical retrieval without
    requiring external services or credentials.
    """

    def __init__(self, dimensions: int = 256):
        self._dimensions = dimensions

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = TOKEN_RE.findall(text.lower())

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

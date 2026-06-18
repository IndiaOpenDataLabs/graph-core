"""Domain-aware document chunking for ingestion.

Uses recursive, structure-preserving chunking for general text and an
AST-first strategy for code.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import tiktoken
from chonkie import CodeChunker
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from graph_core.models.domain_config import get_domain_config


class DocumentChunker:
    def __init__(self, chunk_size_tokens: int, chunk_overlap_tokens: int):
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens cannot be negative")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be smaller than chunk_size_tokens"
            )

        self._chunk_size = chunk_size_tokens
        self._chunk_overlap = chunk_overlap_tokens
        self._code_overlap = 0
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._length_fn: Callable[[str], int] = self._token_length
        self._prose_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=self._length_fn,
            separators=["\n\n", "\n", " ", ""],
        )
        self._generic_code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._code_overlap,
            length_function=self._length_fn,
            separators=[
                "\nclass ",
                "\ndef ",
                "\nasync def ",
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ntry",
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

    def chunk_text(self, text: str, domain: str | None = None) -> list[str]:
        if not text.strip():
            return []

        if get_domain_config(domain).use_ast_chunking:
            return self._chunk_code(text)
        return self._clean_chunks(self._prose_splitter.split_text(text))

    def _chunk_code(self, text: str) -> list[str]:
        language_name = self._infer_code_language(text)
        if language_name is not None:
            try:
                ast_chunks = self._chunk_code_with_ast(text, language_name)
                if ast_chunks:
                    return ast_chunks
            except Exception:
                pass

            langchain_language = _LANGCHAIN_LANGUAGE_BY_NAME.get(language_name)
            if langchain_language is not None:
                return self._clean_chunks(
                    RecursiveCharacterTextSplitter.from_language(
                        language=langchain_language,
                        chunk_size=self._chunk_size,
                        chunk_overlap=self._code_overlap,
                        length_function=self._length_fn,
                    ).split_text(text)
                )

        return self._clean_chunks(self._generic_code_splitter.split_text(text))

    def _chunk_code_with_ast(self, text: str, language_name: str) -> list[str]:
        chunker = CodeChunker(
            language=language_name,
            tokenizer="gpt2",
            chunk_size=self._chunk_size,
        )
        raw_chunks = [
            str(getattr(chunk, "text", "")).strip()
            for chunk in chunker.chunk(text)
            if str(getattr(chunk, "text", "")).strip()
        ]
        if not raw_chunks:
            return []

        packed: list[str] = []
        buffer = ""
        for chunk in raw_chunks:
            candidate = self._join_code_units(buffer, chunk) if buffer else chunk
            if self._token_length(candidate) <= self._chunk_size:
                buffer = candidate
                continue
            if buffer:
                packed.append(buffer.strip())
            if self._token_length(chunk) <= self._chunk_size:
                buffer = chunk
            else:
                packed.extend(
                    self._clean_chunks(self._generic_code_splitter.split_text(chunk))
                )
                buffer = ""
        if buffer.strip():
            packed.append(buffer.strip())
        return self._clean_chunks(packed)

    def _token_length(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    @staticmethod
    def _clean_chunks(chunks: list[str]) -> list[str]:
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    @staticmethod
    def _join_code_units(left: str, right: str) -> str:
        if not left:
            return right
        if left.endswith("\n\n") or right.startswith("\n\n"):
            return f"{left}{right}"
        if left.endswith("\n") or right.startswith("\n"):
            return f"{left}\n{right}"
        return f"{left}\n\n{right}"

    @staticmethod
    def _infer_code_language(text: str) -> str | None:
        stripped = text.lstrip()
        if not stripped:
            return None
        if re.search(
            (
                r"^\s*from\s+\S+\s+import\s+"
                r"|^\s*def\s+\w+\s*\("
                r"|^\s*class\s+\w+\s*[:(]"
            ),
            text,
            re.MULTILINE,
        ):
            return "python"
        if re.search(r"^\s*package\s+\w+|^\s*func\s+\w+\s*\(", text, re.MULTILINE):
            return "go"
        if re.search(
            r"^\s*use\s+\S+;|^\s*fn\s+\w+\s*\(|^\s*impl\b",
            text,
            re.MULTILINE,
        ):
            return "rust"
        if re.search(
            r"^\s*import\s+[\w.*]+\s*;|^\s*public\s+class\b|^\s*private\s+\w+",
            text,
            re.MULTILINE,
        ):
            return "java"
        if re.search(
            r"^\s*#include\s+[<\"]|std::|^\s*template\s*<",
            text,
            re.MULTILINE,
        ):
            return "cpp"
        if re.search(
            r"^\s*using\s+\S+;|^\s*namespace\s+\S+|^\s*public\s+class\b",
            text,
            re.MULTILINE,
        ):
            return "csharp"
        if re.search(
            (
                r"^\s*function\s+\w+\s*\("
                r"|^\s*const\s+\w+\s*="
                r"|^\s*export\s+default\b"
            ),
            text,
            re.MULTILINE,
        ):
            if re.search(
                r":\s*[A-Z][A-Za-z0-9_<>, ?|[\]]+|interface\s+\w+|type\s+\w+\s*=",
                text,
            ):
                return "typescript"
            return "javascript"
        if re.search(r"^\s*<\?php|^\s*namespace\s+\S+;|->", text, re.MULTILINE):
            return "php"
        if re.search(
            (
                r"^\s*class\s+\w+\s*<\s*ApplicationRecord"
                r"|^\s*module\s+\w+"
                r"|^\s*end\s*$"
            ),
            text,
            re.MULTILINE,
        ):
            return "ruby"
        if re.search(
            r"^\s*fun\s+\w+\s*\(|^\s*val\s+\w+\s*=|^\s*data\s+class\b",
            text,
            re.MULTILINE,
        ):
            return "kotlin"
        if re.search(r"^\s*interface\s+\w+|^\s*enum\s+\w+|=>\s*{", text, re.MULTILINE):
            return "typescript"
        return None


_LANGCHAIN_LANGUAGE_BY_NAME: dict[str, Language] = {
    "cpp": Language.CPP,
    "go": Language.GO,
    "java": Language.JAVA,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "php": Language.PHP,
    "python": Language.PYTHON,
    "rust": Language.RUST,
    "ruby": Language.RUBY,
    "kotlin": Language.KOTLIN,
}

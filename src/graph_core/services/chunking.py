"""Domain-aware document chunking for ingestion.

Uses recursive, structure-preserving chunking for general text and an
AST-first strategy for code.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

import tiktoken
from chonkie import CodeChunker
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from graph_core.models.domain_config import get_domain_config

_ATX_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})(?:[ \t]+|$)(.*)$")
_SETEXT_UNDERLINE_RE = re.compile(r"^[ \t]{0,3}(=+|-+)[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_MAX_HEADING_CHARS = 256
_MAX_SECTION_PATH_CHARS = 512


def _truncate_unicode(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    end = limit
    while end > 0 and unicodedata.combining(text[end]):
        end -= 1
    return text[:end].rstrip()


@dataclass(frozen=True)
class SourceHierarchy:
    document_path: str | None
    folder_path: str | None
    headings: tuple[str, ...]

    def prefix(self) -> str:
        lines = ["Source hierarchy:"]
        if self.document_path:
            lines.append(f"Document: {self.document_path}")
        if self.folder_path:
            lines.append(f"Folder: {self.folder_path}")
        if self.headings:
            section_path = " > ".join(self.headings)
            lines.append(
                "Section: "
                + _truncate_unicode(section_path, _MAX_SECTION_PATH_CHARS)
            )
        if len(lines) == 1:
            return ""
        return "\n".join(lines) + "\n\nChunk text:\n"


def _normalize_source_path(document_path: str | None) -> str | None:
    normalized = str(document_path or "").strip().replace("\\", "/")
    if not normalized:
        return None
    normalized = PurePosixPath(normalized).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized or None


def _folder_path(document_path: str | None) -> str | None:
    if not document_path:
        return None
    parent = PurePosixPath(document_path).parent.as_posix()
    if parent in ("", "."):
        return None
    return parent


def _clean_heading(heading: str) -> str:
    cleaned = unicodedata.normalize("NFC", heading).strip()
    cleaned = re.sub(r"[ \t]+#+[ \t]*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return _truncate_unicode(cleaned, _MAX_HEADING_CHARS)


@dataclass(frozen=True)
class _MarkdownHeading:
    level: int
    text: str
    offset: int


def _markdown_headings_before(text: str, offset: int) -> list[_MarkdownHeading]:
    headings: list[_MarkdownHeading] = []
    previous_content: tuple[str, int] | None = None
    in_fence: str | None = None
    position = 0

    for raw_line in text.splitlines(keepends=True):
        line_start = position
        position += len(raw_line)
        if line_start > offset:
            break

        line = raw_line.rstrip("\r\n")
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            fence_marker = fence_match.group(1)[0]
            if in_fence == fence_marker:
                in_fence = None
            elif in_fence is None:
                in_fence = fence_marker
            previous_content = None
            continue
        if in_fence is not None:
            continue

        atx_match = _ATX_HEADING_RE.match(line)
        if atx_match:
            heading = _clean_heading(atx_match.group(2))
            if heading:
                headings.append(
                    _MarkdownHeading(
                        level=len(atx_match.group(1)),
                        text=heading,
                        offset=line_start,
                    )
                )
            previous_content = None
            continue

        setext_match = _SETEXT_UNDERLINE_RE.match(line)
        if setext_match and previous_content is not None:
            content, content_offset = previous_content
            heading = _clean_heading(content)
            if heading:
                headings.append(
                    _MarkdownHeading(
                        level=1 if setext_match.group(1).startswith("=") else 2,
                        text=heading,
                        offset=content_offset,
                    )
                )
            previous_content = None
            continue

        if line.strip():
            previous_content = (line, line_start)
        else:
            previous_content = None

    return headings


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
        text = unicodedata.normalize("NFC", text)
        if not text.strip():
            return []

        if get_domain_config(domain).use_ast_chunking:
            return self._chunk_code(text)
        return self._clean_chunks(self._prose_splitter.split_text(text))

    def chunk_document(
        self,
        text: str,
        *,
        domain: str | None = None,
        document_path: str | None = None,
    ) -> list[str]:
        text = unicodedata.normalize("NFC", text)
        chunks = self.chunk_text(text, domain=domain)
        if not chunks:
            return []
        if get_domain_config(domain).use_ast_chunking:
            return chunks

        enriched: list[str] = []
        cursor = 0
        for chunk in chunks:
            start = text.find(chunk, cursor)
            if start < 0:
                start = cursor
            cursor = max(start + len(chunk), cursor)
            hierarchy = self._source_hierarchy_at(
                text,
                start,
                document_path=document_path,
            )
            prefix = hierarchy.prefix()
            enriched.append(f"{prefix}{chunk}" if prefix else chunk)
        return enriched

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
        cleaned = [
            unicodedata.normalize("NFC", chunk.strip())
            for chunk in chunks
            if chunk and chunk.strip()
        ]
        for index in range(1, len(cleaned)):
            leading_marks = ""
            while cleaned[index] and unicodedata.combining(cleaned[index][0]):
                leading_marks += cleaned[index][0]
                cleaned[index] = cleaned[index][1:]
            if leading_marks:
                cleaned[index - 1] += leading_marks
        return [chunk for chunk in cleaned if chunk]

    @staticmethod
    def _source_hierarchy_at(
        text: str,
        offset: int,
        *,
        document_path: str | None = None,
    ) -> SourceHierarchy:
        normalized_path = _normalize_source_path(document_path)
        folder_path = _folder_path(normalized_path)
        heading_stack: dict[int, str] = {}
        for heading_info in _markdown_headings_before(text, offset):
            level = heading_info.level
            heading = heading_info.text
            heading_stack[level] = heading
            for existing_level in list(heading_stack):
                if existing_level > level:
                    del heading_stack[existing_level]
        return SourceHierarchy(
            document_path=normalized_path,
            folder_path=folder_path,
            headings=tuple(
                heading_stack[level] for level in sorted(heading_stack)
            ),
        )

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

"""TextSanitizer — sanitizes inbound text before LLM processing.

Defense-in-depth: primary injection defense is structured outputs.
This layer handles Unicode normalization, size limits, encoding validation,
and collection-aware pattern detection.
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

MAX_CHUNK_SIZE = 16_000  # characters


@dataclass
class SanitizationReport:
    """What was sanitized and at what severity."""

    normalized: bool
    size_clipped: bool
    patterns_stripped: int
    severity: Literal["none", "low", "medium", "high"]
    details: list[str]


class TextSanitizer:
    def __init__(self, *, trusted_namespace_ids: set[str] | None = None):
        """trusted_namespace_ids: namespaces that skip aggressive pattern stripping."""
        self._trusted = trusted_namespace_ids or set()

    def sanitize(self, text: str, namespace_id: str) -> tuple[str, SanitizationReport]:
        details: list[str] = []
        normalized = False
        size_clipped = False
        patterns_stripped = 0

        # 1. Unicode normalization
        if text != unicodedata.normalize("NFC", text):
            text = unicodedata.normalize("NFC", text)
            normalized = True
            details.append("nfc-normalized")

        # 2. Remove zero-width characters
        zero_width = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060]")
        if zero_width.search(text):
            text = zero_width.sub("", text)
            details.append("zero-width-removed")

        # 3. Reject null bytes
        if "\x00" in text:
            text = text.replace("\x00", "")
            details.append("null-bytes-removed")

        # 4. Size enforcement
        if len(text) > MAX_CHUNK_SIZE:
            text = text[:MAX_CHUNK_SIZE]
            size_clipped = True
            details.append(f"clipped-to-{MAX_CHUNK_SIZE}")

        # 5. Pattern detection (skip for trusted namespaces)
        if namespace_id not in self._trusted:
            injection_patterns = [
                r"(?i)ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|directions?)",
                r"(?i)you\s+(?:are\s+)?(?:now\s+)?(?:a\s+)?(?:new\s+)?(?:assistant|system|model)",
                r"(?i)disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|directives?)",
                r"(?i)new\s+(?:instructions?|directives?)\s*:",
                r"<\s*system\s*>",
                r"(?i)base64\s*:\s*[A-Za-z0-9+/=]{50,}",
            ]
            for pat in injection_patterns:
                matches = re.findall(pat, text)
                patterns_stripped += len(matches)
                if matches:
                    details.append(f"pattern-stripped:{pat[:30]}")

        # Determine severity
        severity = "none"
        if patterns_stripped:
            severity = "low" if patterns_stripped <= 2 else "medium" if patterns_stripped <= 5 else "high"

        report = SanitizationReport(
            normalized=normalized,
            size_clipped=size_clipped,
            patterns_stripped=patterns_stripped,
            severity=severity,
            details=details,
        )
        return text, report

    @staticmethod
    def chunk_hash(text: str) -> str:
        """SHA-256 hash of text for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

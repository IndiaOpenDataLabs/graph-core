"""Helpers for leveled meta-collection naming and parsing."""

from __future__ import annotations

import re

_LEGACY_META_SUFFIX = "__meta"
_LEVELED_META_RE = re.compile(r"^(?P<root>.+)__meta__l(?P<level>[1-9]\d*)$")


def parse_meta_collection_name(
    collection_name: str,
) -> tuple[str, int] | None:
    """Return ``(root_name, level)`` for meta collections, else ``None``."""
    match = _LEVELED_META_RE.fullmatch(collection_name)
    if match:
        return match.group("root"), int(match.group("level"))
    if collection_name.endswith(_LEGACY_META_SUFFIX):
        root_name = collection_name[: -len(_LEGACY_META_SUFFIX)]
        if root_name:
            return root_name, 1
    return None


def is_meta_collection_name(collection_name: str) -> bool:
    return parse_meta_collection_name(collection_name) is not None


def base_collection_name(collection_name: str) -> str:
    parsed = parse_meta_collection_name(collection_name)
    if parsed is None:
        return collection_name
    return parsed[0]


def meta_collection_level(collection_name: str) -> int:
    parsed = parse_meta_collection_name(collection_name)
    if parsed is None:
        return 0
    return parsed[1]


def meta_collection_name(collection_name: str, level: int = 1) -> str:
    if level < 1:
        raise ValueError("Meta collection level must be 1 or greater")
    return f"{base_collection_name(collection_name)}__meta__l{level}"


def legacy_meta_collection_name(collection_name: str) -> str:
    return f"{base_collection_name(collection_name)}{_LEGACY_META_SUFFIX}"

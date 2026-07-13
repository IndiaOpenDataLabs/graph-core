"""Service layer - pure Python, no transport dependencies."""

from typing import Any

import graph_core.services.auth_service as auth_service

__all__ = ["auth_service", "GraphService", "PlatformService", "TextSanitizer"]


def __getattr__(name: str) -> Any:
    """Keep service exports without importing every service during migrations."""
    if name == "GraphService":
        from graph_core.services.graph import GraphService

        return GraphService
    if name == "PlatformService":
        from graph_core.services.platform import PlatformService

        return PlatformService
    if name == "TextSanitizer":
        from graph_core.services.sanitizer import TextSanitizer

        return TextSanitizer
    raise AttributeError(name)

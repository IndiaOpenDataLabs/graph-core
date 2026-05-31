"""Helpers for turning provider connectivity failures into API errors."""

from fastapi import HTTPException
from openai import APIConnectionError, AuthenticationError


def raise_provider_http_error(exc: Exception) -> None:
    """Raise a user-facing HTTP error for known provider failures."""
    if isinstance(exc, AuthenticationError):
        raise HTTPException(
            status_code=502,
            detail=(
                "Model provider authentication failed. Check the profile "
                "credential secret and base URL."
            ),
        ) from exc
    if isinstance(exc, APIConnectionError):
        raise HTTPException(
            status_code=502,
            detail=(
                "Model provider connection failed. If you configured a local "
                "OpenAI-compatible server and graph-core is running in Docker, "
                "use host.docker.internal instead of localhost in the profile "
                "or credential base URL."
            ),
        ) from exc

"""Helpers for provider base URL normalization."""

from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit

_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def normalize_provider_base_url(
    base_url: str | None,
    *,
    running_in_container: bool | None = None,
) -> str | None:
    """Rewrite host-local provider URLs for containerized app/worker processes.

    When graph-core runs in Docker, `localhost` points at the container itself,
    not the developer's host machine. Most local OpenAI-compatible servers run on
    the host, so map loopback URLs to `host.docker.internal`.
    """
    if not base_url:
        return base_url

    parts = urlsplit(base_url)
    hostname = parts.hostname
    if hostname not in _LOCAL_HOSTNAMES:
        return base_url

    if running_in_container is None:
        running_in_container = Path("/.dockerenv").exists()
    if not running_in_container:
        return base_url

    netloc = "host.docker.internal"
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"

    rewritten = SplitResult(
        scheme=parts.scheme,
        netloc=netloc,
        path=parts.path,
        query=parts.query,
        fragment=parts.fragment,
    )
    return urlunsplit(rewritten)

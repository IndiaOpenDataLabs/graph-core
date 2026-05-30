"""Persistent TUI configuration — stored at ~/.config/graph-core/config.json."""

import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "graph-core"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_MCP_URL = "http://localhost:8001/mcp/"


def config_exists() -> bool:
    """Return True if a persisted config file is present."""
    return CONFIG_FILE.is_file()


def load_config() -> dict:
    """Load persisted config from disk.

    Returns a dict with keys: mcp_url, api_key, is_admin, namespace_id, namespace_name.
    Falls back to defaults if the file is missing or malformed.
    """
    defaults = {
        "mcp_url": DEFAULT_MCP_URL,
        "api_key": "",
        "is_admin": False,
        "namespace_id": "",
        "namespace_name": "",
    }
    if not CONFIG_FILE.is_file():
        return defaults

    try:
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        # Merge with defaults so new keys are always present.
        merged = {**defaults, **data}
        merged["is_admin"] = bool(merged.get("is_admin", False))
        return merged
    except (json.JSONDecodeError, OSError):
        return defaults


def save_config(cfg: dict) -> None:
    """Persist config dict to disk."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump({
            "mcp_url": cfg.get("mcp_url", DEFAULT_MCP_URL),
            "api_key": cfg.get("api_key", ""),
            "is_admin": bool(cfg.get("is_admin", False)),
            "namespace_id": cfg.get("namespace_id", ""),
            "namespace_name": cfg.get("namespace_name", ""),
        }, f, indent=2)

    # Also populate os.environ so the current process picks up the values.
    os.environ["MCP_URL"] = cfg.get("mcp_url", DEFAULT_MCP_URL)
    api_key = cfg.get("api_key", "")
    if api_key:
        os.environ["GRAPH_CORE_API_KEY"] = api_key
        if cfg.get("is_admin"):
            os.environ["PLATFORM_ADMIN_KEY"] = api_key

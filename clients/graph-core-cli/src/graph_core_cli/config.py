"""Persistent TUI configuration — stored at ~/.config/graph-core/config.json."""

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "graph-core"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_MCP_URL = "http://localhost:8001/mcp/"


def config_exists() -> bool:
    """Return True if a persisted config file is present."""
    return CONFIG_FILE.is_file()


def load_config() -> dict:
    """Load persisted config from disk."""
    defaults = {
        "mcp_url": DEFAULT_MCP_URL,
        "api_key": "",
        "admin_api_key": "",
        "namespace_api_key": "",
        "active_api_key_kind": "admin",
        "is_admin": False,
        "namespace_id": "",
        "namespace_name": "",
    }
    if not CONFIG_FILE.is_file():
        return defaults
    try:
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        merged = {**defaults, **data}
        legacy_api_key = str(merged.get("api_key", "") or "")
        legacy_is_admin = bool(merged.get("is_admin", False))
        if legacy_api_key:
            if legacy_is_admin:
                merged["admin_api_key"] = merged.get("admin_api_key") or legacy_api_key
                merged["active_api_key_kind"] = (
                    merged.get("active_api_key_kind") or "admin"
                )
            else:
                merged["namespace_api_key"] = (
                    merged.get("namespace_api_key") or legacy_api_key
                )
                merged["active_api_key_kind"] = (
                    merged.get("active_api_key_kind") or "namespace"
                )
        return merged
    except (json.JSONDecodeError, OSError):
        return defaults


def save_config(cfg: dict) -> None:
    """Persist config dict to disk."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    active_kind = cfg.get("active_api_key_kind", "admin")
    admin_api_key = cfg.get("admin_api_key", "")
    namespace_api_key = cfg.get("namespace_api_key", "")
    active_api_key = namespace_api_key if active_kind == "namespace" else admin_api_key
    with open(CONFIG_FILE, "w") as f:
        json.dump({
            "mcp_url": cfg.get("mcp_url", DEFAULT_MCP_URL),
            "api_key": active_api_key,
            "admin_api_key": admin_api_key,
            "namespace_api_key": namespace_api_key,
            "active_api_key_kind": active_kind,
            "is_admin": active_kind == "admin",
            "namespace_id": cfg.get("namespace_id", ""),
            "namespace_name": cfg.get("namespace_name", ""),
        }, f, indent=2)

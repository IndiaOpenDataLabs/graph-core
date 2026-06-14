"""Persistent TUI configuration — stored at ~/.config/graph-core/config.json."""

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "graph-core"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_API_BASE_URL = "http://localhost:8001"
DEFAULT_ADMIN_MCP_URL = "http://localhost:8002/mcp/"
DEFAULT_USER_MCP_URL = "http://localhost:8003/mcp/"


def config_exists() -> bool:
    """Return True if a persisted config file is present."""
    return CONFIG_FILE.is_file()


def load_config() -> dict:
    """Load persisted config from disk."""
    defaults = {
        "admin_mcp_url": DEFAULT_ADMIN_MCP_URL,
        "user_mcp_url": DEFAULT_USER_MCP_URL,
        "admin_jwt": "",
        "namespace_token": "",
        "ui_mode": "admin",
        "namespace_id": "",
        "namespace_name": "",
    }
    if not CONFIG_FILE.is_file():
        return defaults
    try:
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        return {**defaults, **data}
    except (json.JSONDecodeError, OSError):
        return defaults


def save_config(cfg: dict) -> None:
    """Persist config dict to disk."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump({
            "admin_mcp_url": cfg.get("admin_mcp_url", DEFAULT_ADMIN_MCP_URL),
            "user_mcp_url": cfg.get("user_mcp_url", DEFAULT_USER_MCP_URL),
            "admin_jwt": cfg.get("admin_jwt", ""),
            "namespace_token": cfg.get("namespace_token", ""),
            "ui_mode": cfg.get("ui_mode", "admin"),
            "namespace_id": cfg.get("namespace_id", ""),
            "namespace_name": cfg.get("namespace_name", ""),
        }, f, indent=2)

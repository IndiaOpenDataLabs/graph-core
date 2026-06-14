"""Graph Core TUI — terminal client for the platform."""

from textual.app import App, ComposeResult
from textual.binding import Binding

from graph_core_cli.config import load_config, save_config
from graph_core_cli.mcp_client import AuthenticatedMCPClient


class GraphCoreTUI(App):
    """Terminal UI for the Graph Core platform."""

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
    ]

    CSS = """
    /* App-wide styles */
    """

    def on_mount(self) -> None:
        from graph_core_cli.screens import ConsoleScreen, SetupScreen

        persisted = load_config()
        self._config = {
            "mcp_url": persisted.get("mcp_url", "http://localhost:8001/mcp/"),
            "api_key": persisted.get("api_key", ""),
            "admin_jwt": persisted.get("admin_jwt", ""),
            "namespace_api_key": persisted.get("namespace_api_key", ""),
            "active_api_key_kind": persisted.get("active_api_key_kind", "admin"),
            "is_admin": bool(persisted.get("is_admin", False)),
            "namespace_id": persisted.get("namespace_id", ""),
            "namespace_name": persisted.get("namespace_name", ""),
        }
        if self._config.get("active_api_key_kind") == "admin":
            self._config["namespace_id"] = ""
            self._config["namespace_name"] = ""
            save_config(self._config)

        if self.active_api_key:
            self.push_screen(ConsoleScreen())
        else:
            self.push_screen(SetupScreen())

    def compose(self) -> ComposeResult:
        yield from ()

    @property
    def config(self) -> dict:
        if not hasattr(self, "_config"):
            self._config = {
                "mcp_url": "",
                "api_key": "",
                "admin_jwt": "",
                "namespace_api_key": "",
                "active_api_key_kind": "admin",
                "is_admin": False,
            }
        return self._config

    @config.setter
    def config(self, value: dict) -> None:
        self._config = value
        save_config(value)

    @property
    def mcp_client(self) -> AuthenticatedMCPClient:
        return self.mcp_client_for_key(self.active_api_key)

    @property
    def active_api_key(self) -> str:
        kind = self.config.get("active_api_key_kind", "admin")
        if kind == "namespace":
            return self.config.get("namespace_api_key", "")
        return self.config.get("admin_jwt", "") or self.config.get("api_key", "")

    @property
    def admin_jwt(self) -> str:
        return self.config.get("admin_jwt", "") or (
            self.config.get("api_key", "") if self.config.get("is_admin", False) else ""
        )

    @property
    def admin_api_key(self) -> str:
        return self.admin_jwt

    @property
    def namespace_api_key(self) -> str:
        return self.config.get("namespace_api_key", "") or (
            self.config.get("api_key", "")
            if not self.config.get("is_admin", True)
            else ""
        )

    def mcp_client_for_key(self, api_key: str) -> AuthenticatedMCPClient:
        mcp_url = self.config.get("mcp_url", "http://localhost:8001/mcp/")
        return AuthenticatedMCPClient(mcp_url, api_key)


async def main() -> None:
    app = GraphCoreTUI()
    app.run(mouse=False)

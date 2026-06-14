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
            "api_base_url": persisted.get("api_base_url", "http://localhost:8001"),
            "mcp_url": persisted.get("admin_mcp_url", "http://localhost:8002/mcp/"),
            "admin_mcp_url": persisted.get("admin_mcp_url", "http://localhost:8002/mcp/"),
            "user_mcp_url": persisted.get("user_mcp_url", "http://localhost:8003/mcp/"),
            "admin_jwt": persisted.get("admin_jwt", ""),
            "namespace_token": persisted.get("namespace_token", ""),
            "active_token_kind": persisted.get("active_token_kind", "admin"),
            "namespace_id": persisted.get("namespace_id", ""),
            "namespace_name": persisted.get("namespace_name", ""),
        }
        if self._config.get("active_token_kind") == "admin":
            self._config["namespace_id"] = ""
            self._config["namespace_name"] = ""
        save_config(self._config)

        if self.active_token:
            self.push_screen(ConsoleScreen())
        else:
            self.push_screen(SetupScreen())

    def compose(self) -> ComposeResult:
        yield from ()

    @property
    def config(self) -> dict:
        if not hasattr(self, "_config"):
            self._config = {
                "api_base_url": "",
                "mcp_url": "",
                "admin_mcp_url": "",
                "user_mcp_url": "",
                "admin_jwt": "",
                "namespace_token": "",
                "active_token_kind": "admin",
                "namespace_id": "",
                "namespace_name": "",
            }
        return self._config

    @config.setter
    def config(self, value: dict) -> None:
        self._config = value
        save_config(value)

    @property
    def mcp_client(self) -> AuthenticatedMCPClient:
        return self.mcp_client_for_token(
            self.active_token,
            kind=self.active_token_kind,
        )

    @property
    def active_token(self) -> str:
        kind = self.active_token_kind
        if kind == "namespace":
            return self.config.get("namespace_token", "")
        return self.config.get("admin_jwt", "")

    @property
    def active_token_kind(self) -> str:
        return self.config.get("active_token_kind", "admin")

    @property
    def admin_jwt(self) -> str:
        return self.config.get("admin_jwt", "")

    @property
    def admin_token(self) -> str:
        return self.admin_jwt

    @property
    def namespace_token(self) -> str:
        return self.config.get("namespace_token", "")

    def mcp_client_for_token(
        self,
        token: str,
        *,
        kind: str | None = None,
    ) -> AuthenticatedMCPClient:
        kind = kind or self.active_token_kind
        if kind == "namespace":
            mcp_url = self.config.get("user_mcp_url", "http://localhost:8003/mcp/")
        else:
            mcp_url = self.config.get("admin_mcp_url", self.config.get("mcp_url", "http://localhost:8002/mcp/"))
        return AuthenticatedMCPClient(mcp_url, token)


async def main() -> None:
    app = GraphCoreTUI()
    app.run(mouse=False)

"""Graph Core TUI — terminal interface for the platform."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from graph_core.cli.mcp_client import MCPClient


class GraphCoreTUI(App):
    """Terminal UI for the Graph Core platform."""

    BINDINGS = [
        Binding("c", "show_config", "Config", priority=True),
        Binding("q", "quit", "Quit", priority=True),
    ]

    CSS = """
    /* App-wide styles */
    """

    def on_mount(self) -> None:
        if not self.config.get("api_key"):
            from graph_core.cli.screens import ConfigScreen

            self.push_screen(ConfigScreen())

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    async def action_show_config(self) -> None:
        from graph_core.cli.screens import ConfigScreen

        self.push_screen(ConfigScreen())

    @property
    def config(self) -> dict:
        if not hasattr(self, "_config"):
            self._config = {"base_url": "", "api_key": "", "is_admin": False}
        return self._config

    @config.setter
    def config(self, value: dict) -> None:
        self._config = value

    @property
    def mcp_client(self) -> MCPClient:
        mcp_url = self.config.get("mcp_url", f"{self.config['base_url']}/mcp")
        return MCPClient(mcp_url)


async def main() -> None:
    app = GraphCoreTUI()
    app.run()

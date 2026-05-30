"""Graph Core TUI — terminal interface for the platform."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from graph_core.cli.config import load_config, save_config
from graph_core.cli.mcp_client import MCPClient


class GraphCoreTUI(App):
    """Terminal UI for the Graph Core platform."""

    BINDINGS = [
        Binding("h", "show_home", "Home", priority=True),
        Binding("c", "show_config", "Config", priority=True),
        Binding("n", "show_namespaces", "Namespaces", priority=True),
        Binding("l", "show_collections", "Collections", priority=True),
        Binding("shift+q", "show_query", "Query", priority=True),
        Binding("i", "show_ingest", "Ingest", priority=True),
        Binding("j", "show_jobs", "Jobs", priority=True),
        Binding("q", "quit", "Quit", priority=True),
    ]

    CSS = """
    /* App-wide styles */
    """

    def on_mount(self) -> None:
        from graph_core.cli.screens import HomeScreen, SetupScreen

        persisted = load_config()
        self._config = {
            "mcp_url": persisted.get("mcp_url", "http://localhost:8001/mcp/"),
            "api_key": persisted.get("api_key", ""),
            "is_admin": bool(persisted.get("is_admin", False)),
            "namespace_id": persisted.get("namespace_id", ""),
            "namespace_name": persisted.get("namespace_name", ""),
        }

        if persisted.get("api_key"):
            self.push_screen(HomeScreen())
        else:
            self.push_screen(SetupScreen())

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    async def action_show_home(self) -> None:
        from graph_core.cli.screens import HomeScreen

        self.push_screen(HomeScreen())

    async def action_show_config(self) -> None:
        from graph_core.cli.screens import HomeScreen

        self.push_screen(HomeScreen())

    async def action_show_namespaces(self) -> None:
        from graph_core.cli.screens import NamespacesScreen

        self.push_screen(NamespacesScreen())

    async def action_show_collections(self) -> None:
        from graph_core.cli.screens import CollectionsScreen

        self.push_screen(CollectionsScreen())

    async def action_show_query(self) -> None:
        from graph_core.cli.screens import QueryScreen

        self.push_screen(QueryScreen())

    async def action_show_ingest(self) -> None:
        from graph_core.cli.screens import IngestScreen

        self.push_screen(IngestScreen())

    async def action_show_jobs(self) -> None:
        from graph_core.cli.screens import JobsScreen

        self.push_screen(JobsScreen())

    @property
    def config(self) -> dict:
        if not hasattr(self, "_config"):
            self._config = {"mcp_url": "", "api_key": "", "is_admin": False}
        return self._config

    @config.setter
    def config(self, value: dict) -> None:
        self._config = value
        save_config(value)

    @property
    def mcp_client(self) -> MCPClient:
        mcp_url = self.config.get("mcp_url", "http://localhost:8001/mcp/")
        return MCPClient(mcp_url)


async def main() -> None:
    app = GraphCoreTUI()
    app.run()

"""TUI screens for Graph Core — communicates via MCP tools."""

import os
import re

from textual import on
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
    TextArea,
)

# -- Response parsers --------------------------------------------------------


def parse_namespaces(text: str) -> list[dict]:
    items = []
    for m in re.finditer(r"^  - ([^|]+)\| (.+)$", text, re.MULTILINE):
        parts = m.group(2).split()
        items.append({"id": m.group(1).strip(), "name": parts[0] if parts else ""})
    return items


def parse_collections(text: str) -> list[dict]:
    items = []
    for m in re.finditer(r"^  - ([^|]+)\| (.+?) \((.+?)\)$", text, re.MULTILINE):
        items.append({
            "id": m.group(1).strip(),
            "name": m.group(2).strip(),
            "strategy": m.group(3).strip(),
        })
    return items


def parse_key_value(text: str) -> dict:
    result = {}
    for m in re.finditer(r"^  ([\w_]+):\s*(.+)$", text, re.MULTILINE):
        result[m.group(1).strip()] = m.group(2).strip()
    return result


def extract_id(text: str) -> str:
    m = re.search(r"id:\s*([^\s\n]+)", text)
    return m.group(1) if m else ""


def extract_name(text: str) -> str:
    m = re.search(r"name:\s*([^\s\n]+)", text)
    return m.group(1) if m else ""


def extract_job_id(text: str) -> str:
    m = re.search(r"job_id:\s*([^\s\n]+)", text)
    return m.group(1) if m else ""


def extract_status(text: str) -> str:
    m = re.search(r"status:\s*([^\s\n]+)", text)
    return m.group(1) if m else ""


# -- Screens -----------------------------------------------------------------


class HomeScreen(Screen):
    """Dashboard home screen with inline config and navigation."""

    CSS = """
    HomeScreen {
        align: center middle;
    }

    #dashboard {
        width: 70;
        height: auto;
        max-height: 100%;
        border: round $accent;
        padding: 1 2;
    }

    #title {
        text-align: center;
        color: $accent;
        text-style: bold underline;
    }

    #info {
        margin-top: 1;
    }

    #config-section {
        margin-top: 1;
        padding-top: 1;
        border-top: solid $accent-darken-2;
    }

    #config-section Label {
        width: 100%;
    }

    #config-section Label.margin-top {
        margin-top: 1;
    }

    #config-section Input {
        width: 100%;
    }

    #home-connect {
        margin-top: 1;
        width: 100%;
    }

    #nav {
        margin-top: 1;
        height: auto;
    }

    #nav-title {
        margin-bottom: 1;
        text-style: bold;
    }

    #nav Button {
        width: 100%;
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> None:
        mcp_url = os.getenv("MCP_URL", "http://localhost:8001/mcp/")
        yield Container(
            Label("Graph Core Terminal", id="title"),
            Label("", id="info"),
            Container(
                Label("Connection Setup:"),
                Label("MCP URL:"),
                Input(
                    placeholder="http://localhost:8001/mcp/",
                    id="home-mcp-url",
                    value=mcp_url,
                ),
                Label("Auth mode:", classes="margin-top"),
                RadioSet(
                    RadioButton("namespace", id="home-mode-namespace"),
                    RadioButton("admin", id="home-mode-admin"),
                    id="home-auth-mode",
                ),
                Label("Namespace API Key:", id="key-label", classes="margin-top"),
                Input(
                    placeholder="ns_key_xxxx",
                    id="home-api-key",
                    password=True,
                    value=os.getenv("GRAPH_CORE_API_KEY", ""),
                ),
                id="config-section",
            ),
            Button("Connect", id="home-connect", variant="primary"),
            Container(
                Label("Actions:", id="nav-title"),
                Button(
                    "📋 Namespaces - Manage namespaces (admin only)",
                    id="nav-namespaces",
                ),
                Button("📁 Collections - Manage collections", id="nav-collections"),
                Button("🔍 Query - Query a collection", id="nav-query"),
                Button("📥 Ingest - Add data to a collection", id="nav-ingest"),
                Button("📊 Jobs - View ingestion job status", id="nav-jobs"),
                id="nav",
            ),
            id="dashboard",
        )

    async def on_mount(self) -> None:
        ns_button = self.query_one("#home-mode-namespace", RadioButton)
        ns_button.value = True
        self._refresh_view()

    @on(RadioSet.Changed, "#home-auth-mode")
    def on_auth_mode_changed(self, event: RadioSet.Changed) -> None:
        key_input = self.query_one("#home-api-key", Input)
        key_label = self.query_one("#key-label", Label)
        is_admin = event.pressed.id == "home-mode-admin"
        if is_admin:
            key_label.update("Platform Admin Key:")
            key_input.placeholder = "e.g. graph-core-admin-key-dev"
            key_input.value = os.getenv("PLATFORM_ADMIN_KEY", "")
        else:
            key_label.update("Namespace API Key:")
            key_input.placeholder = "ns_key_xxxx"
            key_input.value = os.getenv("GRAPH_CORE_API_KEY", "")

    def _refresh_view(self) -> None:
        cfg = self.app.config
        info = self.query_one("#info", Label)
        config_section = self.query_one("#config-section", Container)
        connect_button = self.query_one("#home-connect", Button)
        nav_section = self.query_one("#nav", Container)

        if not cfg.get("api_key"):
            config_section.display = True
            connect_button.display = True
            nav_section.display = False
            info.update("Status: Not connected")
        else:
            config_section.display = False
            connect_button.display = False
            nav_section.display = True
            parts = [
                "Status: Connected",
                f"MCP: {cfg.get('mcp_url', 'http://localhost:8001/mcp/')}",
                f"Mode: {'Admin' if cfg['is_admin'] else 'Namespace'}",
            ]
            if cfg.get("namespace_name"):
                ns_name = cfg["namespace_name"]
                ns_id = cfg["namespace_id"]
                parts.append(f"Namespace: {ns_name} ({ns_id})")
            else:
                parts.append("Namespace: (not selected)")
            info.update("\n".join(parts))

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id == "home-connect":
            self._save_and_connect()
            return

        screen_map = {
            "nav-namespaces": NamespacesScreen,
            "nav-collections": CollectionsScreen,
            "nav-query": QueryScreen,
            "nav-ingest": IngestScreen,
            "nav-jobs": JobsScreen,
        }
        screen_cls = screen_map.get(event.button.id)
        if screen_cls:
            self.app.push_screen(screen_cls())

    def _save_and_connect(self) -> None:
        mcp_url = self.query_one("#home-mcp-url", Input).value.strip()
        api_key = self.query_one("#home-api-key", Input).value.strip()
        admin_button = self.query_one("#home-mode-admin", RadioButton)
        is_admin = admin_button.value

        if not api_key:
            self.notify("API key is required", severity="error")
            return

        if not mcp_url:
            mcp_url = "http://localhost:8001/mcp/"

        self.app.config = {
            "mcp_url": mcp_url,
            "api_key": api_key,
            "is_admin": is_admin,
            "namespace_id": "",
            "namespace_name": "",
        }

        os.environ["MCP_URL"] = mcp_url
        os.environ["GRAPH_CORE_API_KEY"] = api_key
        if is_admin:
            os.environ["PLATFORM_ADMIN_KEY"] = api_key

        self.notify("Connected!", severity="information", timeout=5)
        self._refresh_view()


class NamespacesScreen(Screen):
    """Manage namespaces."""

    CSS = """
    NamespacesScreen {
        layout: grid;
        grid-size: 1;
        grid-rows: auto 1fr auto;
    }

    #header {
        padding: 1;
        background: $boost;
        color: $accent;
    }

    #namespace-table {
        height: 1fr;
    }

    #actions {
        padding: 1;
        dock: bottom;
    }

    #create-form {
        display: none;
        padding: 1;
        background: $surface;
        border: round $accent;
    }

    #create-form.visible {
        display: block;
    }

    Input {
        width: 100%;
    }
    """

    BINDINGS = [
        ("a", "create_namespace", "Create"),
        ("r", "refresh", "Refresh"),
        ("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> None:
        yield Label("Namespaces  |  a=Create  r=Refresh  esc=Back", id="header")
        yield DataTable(id="namespace-table")
        yield Container(
            Input(placeholder="Namespace name", id="ns-name-input"),
            Button("Create", id="ns-create-btn", variant="primary"),
            Button("Cancel", id="ns-cancel-btn"),
            id="create-form",
        )

    async def on_mount(self) -> None:
        self.run_worker(self._load_namespaces(), exclusive=True, group="load")

    async def action_create_namespace(self) -> None:
        form = self.query_one("#create-form", Container)
        form.add_class("visible")
        self.query_one("#ns-name-input", Input).focus()

    async def action_refresh(self) -> None:
        self.run_worker(self._load_namespaces(), exclusive=True, group="load")

    async def _load_namespaces(self) -> None:
        if not self.app.config.get("is_admin"):
            self.notify("Admin key required to list namespaces", severity="warning")
            return

        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                text = await client.call("list_namespaces")
                namespaces = parse_namespaces(text)
            finally:
                await client.disconnect()

            table = self.query_one("#namespace-table", DataTable)
            table.clear()
            table.add_columns("ID", "Name")
            for ns in namespaces:
                table.add_row(ns["id"][:8] if ns["id"] else "", ns["name"])
            if not namespaces:
                self.notify("No namespaces found", severity="information")
        except Exception as e:
            error_msg = str(e)
            if "405" in error_msg or "Method Not Allowed" in error_msg:
                self.notify(
                    "MCP server not available. Run 'make server' first.",
                    severity="error",
                    timeout=10,
                )
            elif "Connection refused" in error_msg or "connect" in error_msg.lower():
                self.notify(
                    "Cannot connect to server. Run 'make server' first.",
                    severity="error",
                    timeout=10,
                )
            else:
                self.notify(f"Failed to load namespaces: {error_msg}", severity="error")

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id == "ns-create-btn":
            self.run_worker(self._create_namespace(), exclusive=True, group="action")
        elif event.button.id == "ns-cancel-btn":
            self.query_one("#create-form", Container).remove_class("visible")

    async def _create_namespace(self) -> None:
        name = self.query_one("#ns-name-input", Input).value.strip()
        if not name:
            self.notify("Name is required", severity="error")
            return

        self.notify(f"Creating namespace '{name}'...", severity="information")
        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                text = await client.call("create_namespace", {"name": name})
                ns_id = extract_id(text)
                ns_name = extract_name(text)
            finally:
                await client.disconnect()

            self.notify(f"Created namespace: {ns_name or name}", severity="information")
            self.query_one("#ns-name-input", Input).value = ""
            self.app.config["namespace_id"] = ns_id
            self.app.config["namespace_name"] = ns_name or name
            self.query_one("#create-form", Container).remove_class("visible")
            self.run_worker(self._load_namespaces(), exclusive=True, group="load")
        except Exception as e:
            error_msg = str(e)
            if "405" in error_msg or "Method Not Allowed" in error_msg:
                self.notify(
                    "MCP server not available. Run 'make server' first.",
                    severity="error",
                    timeout=10,
                )
            elif "Connection refused" in error_msg or "connect" in error_msg.lower():
                self.notify(
                    "Cannot connect to server. Run 'make server' first.",
                    severity="error",
                    timeout=10,
                )
            else:
                self.notify(
                    f"Failed to create namespace: {error_msg}",
                    severity="error",
                )


class CollectionsScreen(Screen):
    """Manage collections in current namespace."""

    CSS = """
    CollectionsScreen {
        layout: grid;
        grid-size: 1;
        grid-rows: auto 1fr auto;
    }

    #header {
        padding: 1;
        background: $boost;
        color: $accent;
    }

    #collection-table {
        height: 1fr;
    }

    #create-form {
        display: none;
        padding: 1;
        background: $surface;
        border: round $accent;
    }

    #create-form.visible {
        display: block;
    }

    Input {
        width: 60%;
    }

    Select {
        width: 20%;
    }
    """

    BINDINGS = [
        ("a", "create_collection", "Create"),
        ("r", "refresh", "Refresh"),
        ("escape", "app.pop_screen", "Back"),
    ]

    STRATEGIES = [
        ("vector", "Vector"),
        ("light_rag", "Light RAG"),
        ("custom_graph_rag", "Graph RAG"),
    ]

    def compose(self) -> None:
        yield Label("Collections  |  a=Create  r=Refresh  esc=Back", id="header")
        yield DataTable(id="collection-table")
        yield Container(
            Input(placeholder="Collection name", id="col-name-input"),
            Select(self.STRATEGIES, allow_blank=True, id="col-strategy"),
            Button("Create", id="col-create-btn", variant="primary"),
            Button("Cancel", id="col-cancel-btn"),
            id="create-form",
        )

    async def on_mount(self) -> None:
        self.run_worker(self._load_collections(), exclusive=True, group="load")

    async def action_create_collection(self) -> None:
        form = self.query_one("#create-form", Container)
        form.add_class("visible")
        self.query_one("#col-name-input", Input).focus()

    async def action_refresh(self) -> None:
        self.run_worker(self._load_collections(), exclusive=True, group="load")

    async def _load_collections(self) -> None:
        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                text = await client.call("list_collections")
                collections = parse_collections(text)
            finally:
                await client.disconnect()

            table = self.query_one("#collection-table", DataTable)
            table.clear()
            table.add_columns("ID", "Name", "Strategy")
            for col in collections:
                cid = col["id"][:8] if col["id"] else ""
                table.add_row(cid, col["name"], col["strategy"])
        except Exception as e:
            self.notify(str(e), severity="error")

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id == "col-create-btn":
            self.run_worker(self._create_collection(), exclusive=True, group="action")
        elif event.button.id == "col-cancel-btn":
            self.query_one("#create-form", Container).remove_class("visible")

    async def _create_collection(self) -> None:
        name = self.query_one("#col-name-input", Input).value.strip()
        strategy_select = self.query_one("#col-strategy", Select)
        strategy_raw = strategy_select.value
        if strategy_raw == strategy_select.NULL or not strategy_raw:
            strategy = "vector"
        elif isinstance(strategy_raw, tuple):
            strategy = strategy_raw[0]
        else:
            strategy = strategy_raw
        if not name:
            self.notify("Name is required", severity="error")
            return

        self.notify(f"Creating collection '{name}'...", severity="information")
        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                args = {"name": name, "strategy": strategy}
                text = await client.call("create_collection", args)
                col_name = extract_name(text)
            finally:
                await client.disconnect()

            self.notify(
                f"Created collection: {col_name or name}",
                severity="information",
            )
            self.query_one("#col-name-input", Input).value = ""
            self.query_one("#create-form", Container).remove_class("visible")
            self.run_worker(self._load_collections(), exclusive=True, group="load")
        except Exception as e:
            self.notify(f"Failed to create collection: {e}", severity="error")


class QueryScreen(Screen):
    """Query a collection via MCP."""

    CSS = """
    QueryScreen {
        layout: grid;
        grid-rows: auto auto 1fr auto;
    }

    #header {
        padding: 1;
        background: $boost;
        color: $accent;
    }

    #controls {
        padding: 1;
    }

    #query-text {
        height: 5;
        border: round $accent;
    }

    #results {
        border: round $accent;
        padding: 1;
        overflow: auto;
    }

    Select {
        width: 40%;
    }

    Input {
        width: 100%;
    }
    """

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> None:
        yield Label("Query Collection  |  esc=Back", id="header")
        yield Container(
            Label("Collection: "),
            Select([], value=None, allow_blank=True, id="collection-select"),
            Label("  Mode: "),
            Select(
                [
                    ("", "default"),
                    ("local", "local"),
                    ("global", "global"),
                    ("hybrid", "hybrid"),
                    ("naive", "naive"),
                    ("mix", "mix"),
                ],
                value=("", "default"),
                id="query-mode",
            ),
            Button("Query", id="query-btn", variant="primary"),
            id="controls",
        )
        yield TextArea("", language="plaintext", id="query-text")
        yield RichLog(id="results", wrap=True)

    async def on_mount(self) -> None:
        self.run_worker(self._load_collections(), exclusive=True, group="load")

    async def _load_collections(self) -> None:
        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                text = await client.call("list_collections")
                collections = parse_collections(text)
            finally:
                await client.disconnect()

            options = [("", "(select collection)")] + [
                (c["id"], f"{c['name']} ({c['strategy']})") for c in collections
            ]
            self.query_one("#collection-select", Select).options = options
            if collections:
                sel = self.query_one("#collection-select", Select)
                sel.value = collections[0]["id"]
        except Exception as e:
            self.notify(str(e), severity="error")

    @on(Button.Pressed)
    def handle_query(self, event: Button.Pressed) -> None:
        if event.button.id == "query-btn":
            self.run_worker(self._run_query(), exclusive=True, group="action")

    async def _run_query(self) -> None:
        collection_id = self.query_one("#collection-select", Select).value
        mode = self.query_one("#query-mode", Select).value[0]
        query_text = self.query_one("#query-text", TextArea)
        results = self.query_one("#results", RichLog)

        if not collection_id:
            self.notify("Select a collection", severity="warning")
            return

        question = query_text.text
        if not question.strip():
            self.notify("Enter a question", severity="warning")
            return

        results.write(f"Querying: {question}\n")
        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                args = {"collection_id": collection_id, "question": question}
                if mode:
                    args["mode"] = mode
                text = await client.call("query_collection", args)
            finally:
                await client.disconnect()

            results.write("\n" + text + "\n\n")
        except Exception as e:
            results.write(f"\nError: {e}\n")
            self.notify(str(e), severity="error")


class IngestScreen(Screen):
    """Ingest text or files into a collection via MCP."""

    CSS = """
    IngestScreen {
        layout: grid;
        grid-rows: auto auto 1fr auto;
    }

    #header {
        padding: 1;
        background: $boost;
        color: $accent;
    }

    #controls {
        padding: 1;
    }

    #text-area {
        height: 1fr;
        border: round $accent;
    }

    #status {
        padding: 1;
        dock: bottom;
    }

    Select {
        width: 30%;
    }

    Input {
        width: 30%;
    }
    """

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> None:
        yield Label("Ingest  |  esc=Back", id="header")
        yield Container(
            Label("Collection: "),
            Select([], value=None, allow_blank=True, id="collection-select"),
            Label("  Method: "),
            Select(
                [("doc", "Document (async)"), ("chunk", "Chunk (sync)")],
                value="doc",
                id="ingest-method",
            ),
            Input(placeholder="File path (optional)", id="file-path"),
            Button("Ingest", id="ingest-btn", variant="primary"),
            id="controls",
        )
        yield TextArea(
            "Paste text here or provide a file path above.",
            language="plaintext",
            id="text-area",
        )
        yield Label("Ready.", id="status")

    async def on_mount(self) -> None:
        self.run_worker(self._load_collections(), exclusive=True, group="load")

    async def _load_collections(self) -> None:
        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                text = await client.call("list_collections")
                collections = parse_collections(text)
            finally:
                await client.disconnect()

            options = [("", "(select)")] + [(c["id"], c["name"]) for c in collections]
            self.query_one("#collection-select", Select).options = options
            if collections:
                sel = self.query_one("#collection-select", Select)
                sel.value = collections[0]["id"]
        except Exception as e:
            self.notify(str(e), severity="error")

    @on(Button.Pressed)
    def handle_ingest(self, event: Button.Pressed) -> None:
        if event.button.id == "ingest-btn":
            self.run_worker(self._do_ingest(), exclusive=True, group="action")

    async def _do_ingest(self) -> None:
        collection_id = self.query_one("#collection-select", Select).value
        method = self.query_one("#ingest-method", Select).value
        status = self.query_one("#status", Label)

        if not collection_id:
            self.notify("Select a collection", severity="warning")
            return

        file_path = self.query_one("#file-path", Input).value.strip()
        if file_path and os.path.isfile(file_path):
            with open(file_path) as f:
                text = f.read()
        else:
            text = self.query_one("#text-area", TextArea).text

        if not text.strip():
            self.notify("No text to ingest", severity="warning")
            return

        status.update("Ingesting...")
        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                if method == "doc":
                    tool_name = "ingest_document"
                else:
                    tool_name = "ingest_chunk"
                args = {"collection_id": collection_id, "text": text}
                mcptext = await client.call(tool_name, args)
            finally:
                await client.disconnect()

            job_id = extract_job_id(mcptext)
            mc_status = extract_status(mcptext)
            if job_id:
                status.update(f"Job started: {job_id} (status: {mc_status})")
                self.notify(f"Ingestion job: {job_id}")
            else:
                status.update(f"Chunk ingested: {mcptext[:80]}...")
        except Exception as e:
            status.update(f"Error: {e}")
            self.notify(str(e), severity="error")


class JobsScreen(Screen):
    """Track ingestion jobs via MCP."""

    CSS = """
    JobsScreen {
        layout: grid;
        grid-rows: auto 1fr auto;
    }

    #header {
        padding: 1;
        background: $boost;
        color: $accent;
    }

    #job-table {
        height: 1fr;
    }

    #status-bar {
        padding: 1;
        dock: bottom;
    }

    Input {
        width: 40%;
    }
    """

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> None:
        yield Label("Jobs  |  r=Refresh  esc=Back", id="header")
        yield DataTable(id="job-table")
        yield Container(
            Label("Job ID: "),
            Input(placeholder="Enter job UUID to check", id="job-id-input"),
            Button("Check", id="check-job-btn", variant="primary"),
            id="status-bar",
        )

    async def action_refresh(self) -> None:
        pass

    @on(Button.Pressed)
    def handle_check(self, event: Button.Pressed) -> None:
        if event.button.id == "check-job-btn":
            self.run_worker(self._check_job(), exclusive=True, group="action")

    async def _check_job(self) -> None:
        job_id = self.query_one("#job-id-input", Input).value.strip()
        if not job_id:
            self.notify("Enter a job ID", severity="warning")
            return

        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                text = await client.call("get_job_status", {"job_id": job_id})
            finally:
                await client.disconnect()

            job_data = parse_key_value(text)
            table = self.query_one("#job-table", DataTable)
            table.clear()
            table.add_columns("Field", "Value")
            for key in ("type", "status", "progress", "chunks", "error"):
                table.add_row(key, job_data.get(key, "-"))

            first_line = text.split("\n")[0]
            m = re.search(r"Job:\s*(.+)", first_line)
            if m:
                table.add_row("id", m.group(1).strip())
        except Exception as e:
            self.notify(str(e), severity="error")

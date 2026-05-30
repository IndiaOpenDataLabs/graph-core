"""TUI screens for Graph Core."""

import os

from textual import on
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    RadioSet,
    RichLog,
    Select,
    TextArea,
)

from graph_core.client import GraphCoreAPIError, GraphCoreClient


class ConfigScreen(Screen):
    """Initial configuration screen."""

    CSS = """
    ConfigScreen {
        align: center middle;
    }

    #config-box {
        width: 60;
        border: round $accent;
        padding: 2;
        background: $boost;
    }

    #config-box Label {
        width: 100%;
    }

    Input {
        width: 100%;
    }

    Horizontal {
        dock: bottom;
        margin: 1 0;
    }

    Button {
        width: 10;
    }

    #title {
        text-align: center;
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def compose(self) -> None:
        yield Container(
            Label("Graph Core Configuration", id="title"),
            Label("Base URL:"),
            Input(
                placeholder="http://localhost:8000",
                id="base-url",
                value=os.getenv("GRAPH_CORE_URL", "http://localhost:8000"),
            ),
            Label("API Key (namespace or admin):", margin=(1, 0, 0, 0)),
            Input(
                placeholder="ns_key_xxxx or admin key",
                id="api-key",
                password=True,
                value=os.getenv("GRAPH_CORE_API_KEY", ""),
            ),
            Label("Auth mode:", margin=(1, 0, 0, 0)),
            RadioSet(
                RadioSet.Radio("namespace", id="mode-namespace"),
                RadioSet.Radio("admin", id="mode-admin"),
                id="auth-mode",
            ),
            Horizontal(
                Button("Connect", id="connect", variant="primary"),
                Button("Skip (connect later)", id="skip"),
            ),
            id="config-box",
        )

    def on_mount(self) -> None:
        self.query_one("#auth-mode").value = "namespace"

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        key_input = self.query_one("#api-key", Input)
        if event.value == "admin":
            key_input.placeholder = "admin key"
            if not key_input.value:
                key_input.value = os.getenv("PLATFORM_ADMIN_KEY", "")
        else:
            key_input.placeholder = "ns_key_xxxx"
            if not key_input.value:
                key_input.value = os.getenv("GRAPH_CORE_API_KEY", "")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connect":
            self._save_and_connect()
        elif event.button.id == "skip":
            self.app.pop_screen()

    def _save_and_connect(self) -> None:
        base_url = self.query_one("#base-url", Input).value.strip()
        api_key = self.query_one("#api-key", Input).value.strip()
        is_admin = self.query_one("#auth-mode", RadioSet).value == "admin"

        if not api_key:
            self.notify("API key is required", severity="error")
            return

        self.app.config = {
            "base_url": base_url or "http://localhost:8000",
            "api_key": api_key,
            "is_admin": is_admin,
            "namespace_id": "",
            "namespace_name": "",
        }

        os.environ["GRAPH_CORE_URL"] = self.app.config["base_url"]
        os.environ["GRAPH_CORE_API_KEY"] = api_key
        if is_admin:
            os.environ["PLATFORM_ADMIN_KEY"] = api_key

        self.app.pop_screen()
        self.app.notify("Connected!")


class HomeScreen(Screen):
    """Dashboard home screen."""

    CSS = """
    HomeScreen {
        align: center middle;
    }

    #dashboard {
        width: 70;
        height: auto;
        border: round $accent;
        padding: 2;
    }

    #title {
        text-align: center;
        color: $accent;
        text-style: bold underline;
    }

    #info {
        margin-top: 1;
    }
    """

    def compose(self) -> None:
        yield Container(
            Label("Graph Core Terminal", id="title"),
            Label("", id="info"),
            id="dashboard",
        )

    async def on_mount(self) -> None:
        cfg = self.app.config
        info = self.query_one("#info", Label)

        if not cfg.get("api_key"):
            info.update("Not configured. Press c to set up connection.")
            return

        parts = [
            f"Server: {cfg['base_url']}",
            f"Mode: {'Admin' if cfg['is_admin'] else 'Namespace'}",
        ]
        if cfg.get("namespace_name"):
            parts.append(f"Namespace: {cfg['namespace_name']} ({cfg['namespace_id']})")
        else:
            parts.append("Namespace: (not selected)")

        info.update("\n".join(parts))


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
        ("escape", "pop_screen", "Back"),
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
        await self._load_namespaces()

    async def action_create_namespace(self) -> None:
        form = self.query_one("#create-form", Container)
        form.add_class("visible")
        self.query_one("#ns-name-input", Input).focus()

    async def action_refresh(self) -> None:
        await self._load_namespaces()

    async def _load_namespaces(self) -> None:
        try:
            client = GraphCoreClient(
                base_url=self.app.config["base_url"],
                api_key=self.app.config["is_admin"]
                and self.app.config.get("api_key")
                or self.app.config.get("api_key"),
                is_admin=self.app.config.get("is_admin", False),
            )
            if not self.app.config.get("is_admin"):
                self.notify("Admin key required to list namespaces", severity="warning")
                return

            namespaces = await client.list_namespaces()
            table = self.query_one("#namespace-table", DataTable)
            table.clear()
            table.add_columns("ID", "Name", "Key Prefix", "Created")
            for ns in namespaces:
                table.add_row(
                    str(ns["id"])[:8],
                    ns["name"],
                    ns.get("api_key_prefix") or "-",
                    str(ns.get("created_at", "-"))[:19]
                    if ns.get("created_at")
                    else "-",
                )
            await client.close()
        except GraphCoreAPIError as e:
            self.notify(str(e), severity="error")

    @on(Button.Pressed)
    async def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id == "ns-create-btn":
            await self._create_namespace()
        elif event.button.id == "ns-cancel-btn":
            self.query_one("#create-form", Container).remove_class("visible")

    async def _create_namespace(self) -> None:
        name = self.query_one("#ns-name-input", Input).value.strip()
        if not name:
            self.notify("Name is required", severity="error")
            return

        try:
            client = GraphCoreClient(
                base_url=self.app.config["base_url"],
                api_key=self.app.config["api_key"],
                is_admin=True,
            )
            result = await client.create_namespace(name)
            await client.close()
            self.notify(f"Created namespace: {result['name']}")
            self.query_one("#ns-name-input", Input).value = ""
            self.app.config["namespace_id"] = result["id"]
            self.app.config["namespace_name"] = result["name"]
            self.query_one("#create-form", Container).remove_class("visible")
            await self._load_namespaces()
        except GraphCoreAPIError as e:
            self.notify(str(e), severity="error")


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
        ("escape", "pop_screen", "Back"),
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
            Select(self.STRATEGIES, value=("vector", "Vector"), id="col-strategy"),
            Button("Create", id="col-create-btn", variant="primary"),
            Button("Cancel", id="col-cancel-btn"),
            id="create-form",
        )

    async def on_mount(self) -> None:
        await self._load_collections()

    async def action_create_collection(self) -> None:
        form = self.query_one("#create-form", Container)
        form.add_class("visible")
        self.query_one("#col-name-input", Input).focus()

    async def action_refresh(self) -> None:
        await self._load_collections()

    async def _load_collections(self) -> None:
        try:
            client = GraphCoreClient(
                base_url=self.app.config["base_url"],
                api_key=self.app.config["api_key"],
            )
            collections = await client.list_collections()
            table = self.query_one("#collection-table", DataTable)
            table.clear()
            table.add_columns("ID", "Name", "Strategy")
            for col in collections:
                table.add_row(str(col["id"])[:8], col["name"], col["strategy"])
            await client.close()
        except GraphCoreAPIError as e:
            self.notify(str(e), severity="error")

    @on(Button.Pressed)
    async def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id == "col-create-btn":
            await self._create_collection()
        elif event.button.id == "col-cancel-btn":
            self.query_one("#create-form", Container).remove_class("visible")

    async def _create_collection(self) -> None:
        name = self.query_one("#col-name-input", Input).value.strip()
        strategy = self.query_one("#col-strategy", Select).value[0]
        if not name:
            self.notify("Name is required", severity="error")
            return

        try:
            client = GraphCoreClient(
                base_url=self.app.config["base_url"],
                api_key=self.app.config["api_key"],
            )
            result = await client.create_collection(name=name, strategy=strategy)
            await client.close()
            self.notify(f"Created collection: {result['name']}")
            self.query_one("#col-name-input", Input).value = ""
            self.query_one("#create-form", Container).remove_class("visible")
            await self._load_collections()
        except GraphCoreAPIError as e:
            self.notify(str(e), severity="error")


class QueryScreen(Screen):
    """Query a collection."""

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
        ("escape", "pop_screen", "Back"),
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
        await self._load_collections()

    async def _load_collections(self) -> None:
        try:
            client = GraphCoreClient(
                base_url=self.app.config["base_url"],
                api_key=self.app.config["api_key"],
            )
            collections = await client.list_collections()
            options = [("", "(select collection)")] + [
                (c["id"], f"{c['name']} ({c['strategy']})") for c in collections
            ]
            self.query_one("#collection-select", Select).options = options
            if collections:
                self.query_one("#collection-select", Select).value = collections[0][
                    "id"
                ]
            await client.close()
        except GraphCoreAPIError as e:
            self.notify(str(e), severity="error")

    @on(Button.Pressed)
    async def handle_query(self, event: Button.Pressed) -> None:
        if event.button.id == "query-btn":
            await self._run_query()

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
            client = GraphCoreClient(
                base_url=self.app.config["base_url"],
                api_key=self.app.config["api_key"],
            )
            result = await client.query_collection(
                collection_id, question, mode=mode or None
            )
            await client.close()
            results.write("\n" + str(result) + "\n\n")
        except GraphCoreAPIError as e:
            results.write(f"\nError: {e}\n")
            self.notify(str(e), severity="error")


class IngestScreen(Screen):
    """Ingest text or files into a collection."""

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
        ("escape", "pop_screen", "Back"),
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
        await self._load_collections()

    async def _load_collections(self) -> None:
        try:
            client = GraphCoreClient(
                base_url=self.app.config["base_url"],
                api_key=self.app.config["api_key"],
            )
            collections = await client.list_collections()
            options = [("", "(select)")] + [(c["id"], c["name"]) for c in collections]
            self.query_one("#collection-select", Select).options = options
            if collections:
                self.query_one("#collection-select", Select).value = collections[0][
                    "id"
                ]
            await client.close()
        except GraphCoreAPIError as e:
            self.notify(str(e), severity="error")

    @on(Button.Pressed)
    async def handle_ingest(self, event: Button.Pressed) -> None:
        if event.button.id == "ingest-btn":
            await self._do_ingest()

    async def _do_ingest(self) -> None:
        collection_id = self.query_one("#collection-select", Select).value
        method = self.query_one("#ingest-method", Select).value
        status = self.query_one("#status", Label)

        if not collection_id:
            self.notify("Select a collection", severity="warning")
            return

        # Get text from file or textarea
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
            client = GraphCoreClient(
                base_url=self.app.config["base_url"],
                api_key=self.app.config["api_key"],
            )
            if method == "doc":
                result = await client.ingest_document(collection_id, text)
                status.update(
                    f"Job started: {result['job_id']} (status: {result['status']})"
                )
                self.notify(f"Ingestion job: {result['job_id']}")
            else:
                result = await client.ingest_chunk(collection_id, text)
                status.update(
                    f"Chunk ingested (hash: {result.get('chunk_hash', '?')}, "
                    f"entities: {result.get('entity_count', 0)}, "
                    f"rels: {result.get('relationship_count', 0)})"
                )
            await client.close()
        except GraphCoreAPIError as e:
            status.update(f"Error: {e}")
            self.notify(str(e), severity="error")


class JobsScreen(Screen):
    """Track ingestion jobs."""

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
        ("escape", "pop_screen", "Back"),
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
        pass  # No auto list endpoint; user enters job ID

    @on(Button.Pressed)
    async def handle_check(self, event: Button.Pressed) -> None:
        if event.button.id == "check-job-btn":
            await self._check_job()

    async def _check_job(self) -> None:
        job_id = self.query_one("#job-id-input", Input).value.strip()
        if not job_id:
            self.notify("Enter a job ID", severity="warning")
            return

        try:
            client = GraphCoreClient(
                base_url=self.app.config["base_url"],
                api_key=self.app.config["api_key"],
            )
            job = await client.get_job(job_id)
            await client.close()

            table = self.query_one("#job-table", DataTable)
            table.clear()
            table.add_columns("Field", "Value")
            for key in (
                "id",
                "job_type",
                "status",
                "progress_percent",
                "chunks_completed",
                "chunks_total",
                "error",
            ):
                table.add_row(key, str(job.get(key, "-")))
            for key in ("created_at", "started_at", "completed_at"):
                val = job.get(key)
                table.add_row(key, str(val)[:19] if val else "-")
        except GraphCoreAPIError as e:
            self.notify(str(e), severity="error")

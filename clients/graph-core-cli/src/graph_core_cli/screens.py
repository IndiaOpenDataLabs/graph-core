"""TUI screens for Graph Core — communicates via MCP tools."""

import os
import re

from textual import events, on
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
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


def parse_profiles(text: str, kind: str) -> list[dict]:
    items = []
    for m in re.finditer(
        r"^  - ([^|]+)\| ([^|]+)\| ([^|]+)\| (.+)$",
        text,
        re.MULTILINE,
    ):
        label = m.group(2).strip()
        items.append({
            "kind": kind,
            "profile_id": m.group(1).strip(),
            "label": "" if label == "-" else label,
            "provider": m.group(3).strip(),
            "model": m.group(4).strip(),
        })
    return items


def extract_id(text: str) -> str:
    m = re.search(r"id:\s*([^\s\n]+)", text)
    return m.group(1) if m else ""


def extract_name(text: str) -> str:
    m = re.search(r"name:\s*([^\s\n]+)", text)
    return m.group(1) if m else ""


def extract_api_key(text: str) -> str:
    m = re.search(r"api_key:\s*([^\s\n]+)", text)
    return m.group(1) if m else ""


def extract_job_id(text: str) -> str:
    m = re.search(r"job_id:\s*([^\s\n]+)", text)
    return m.group(1) if m else ""


def extract_status(text: str) -> str:
    m = re.search(r"status:\s*([^\s\n]+)", text)
    return m.group(1) if m else ""


# -- Screens -----------------------------------------------------------------


class SetupScreen(Screen):
    """One-time setup: capture MCP URL and admin key, then persist."""

    CSS = """
    SetupScreen {
        align: center middle;
    }

    #setup-card {
        width: 65;
        height: auto;
        border: round $accent;
        padding: 1 2;
    }

    #setup-title {
        text-align: center;
        color: $accent;
        text-style: bold underline;
    }

    #setup-desc {
        text-align: center;
        margin-top: 1;
        color: $text;
    }

    Label {
        width: 100%;
    }

    Input {
        width: 100%;
    }

    #setup-connect {
        margin-top: 1;
        width: 100%;
    }

    #setup-error {
        text-align: center;
        color: $error;
        margin-top: 1;
        visibility: hidden;
    }

    #setup-error.visible {
        visibility: visible;
    }
    """

    BINDINGS = [
        ("escape", "app.quit", "Quit"),
    ]

    def compose(self) -> None:
        mcp_url = os.getenv("MCP_URL", "http://localhost:8001/mcp/")
        yield Container(
            Label("Graph Core — First-Time Setup", id="setup-title"),
            Label(
                "Configure your connection. This is saved for future sessions.",
                id="setup-desc",
            ),
            Label("", id="spacer"),
            Label("MCP URL:"),
            Input(
                placeholder="http://localhost:8001/mcp/",
                id="setup-mcp-url",
                value=mcp_url,
            ),
            Label("API Key:", classes="margin-top"),
            Input(
                placeholder="Admin key or namespace key",
                id="setup-api-key",
                password=True,
                value=(
                    os.getenv("PLATFORM_ADMIN_KEY", "")
                    or os.getenv("GRAPH_CORE_API_KEY", "")
                ),
            ),
            Button("Save & Connect", id="setup-connect", variant="primary"),
            Label("", id="setup-error"),
            id="setup-card",
        )

    def on_mount(self) -> None:
        self.query_one("#setup-api-key", Input).focus()

    @on(Button.Pressed)
    def handle_setup(self, event: Button.Pressed) -> None:
        if event.button.id == "setup-connect":
            self._save_and_done()

    def _save_and_done(self) -> None:
        mcp_url = self.query_one("#setup-mcp-url", Input).value.strip()
        api_key = self.query_one("#setup-api-key", Input).value.strip()

        if not api_key:
            err = self.query_one("#setup-error", Label)
            err.update("API key is required.")
            err.add_class("visible")
            return

        if not mcp_url:
            mcp_url = "http://localhost:8001/mcp/"

        is_namespace_key = api_key.startswith("ns_key_")

        self.app.config = {
            "mcp_url": mcp_url,
            "api_key": api_key,
            "admin_api_key": "" if is_namespace_key else api_key,
            "namespace_api_key": api_key if is_namespace_key else "",
            "active_api_key_kind": "namespace" if is_namespace_key else "admin",
            "is_admin": not is_namespace_key,
            "namespace_id": "",
            "namespace_name": "",
        }

        self.notify("Configuration saved!", severity="information", timeout=3)
        from graph_core_cli.screens import HomeScreen
        self.app.push_screen(HomeScreen())


class HomeScreen(Screen):
    """Dashboard home screen with navigation."""

    CSS = """
    HomeScreen {
        align: center middle;
    }

    #dashboard {
        width: 60;
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

    #reconfig {
        margin-top: 1;
        padding-top: 1;
        border-top: solid $accent-darken-2;
        display: none;
    }

    #reconfig.visible {
        display: block;
    }

    #reconfig Input {
        width: 100%;
    }

    #reconfig Label {
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
        yield Container(
            Label("Graph Core Terminal", id="title"),
            Label("", id="info"),
            Container(
                Button("Reconfigure", id="home-reconfigure"),
                id="reconfig",
            ),
            Container(
                Label("Actions:", id="nav-title"),
                Button(
                    "Namespaces - Manage namespaces (admin only)",
                    id="nav-namespaces",
                ),
                Button(
                    "Profiles - Manage embedding and LLM profiles",
                    id="nav-profiles",
                ),
                Button("Collections - Manage collections", id="nav-collections"),
                Button("Query - Query a collection", id="nav-query"),
                Button("Ingest - Add data to a collection", id="nav-ingest"),
                Button("Jobs - View ingestion job status", id="nav-jobs"),
                id="nav",
            ),
            id="dashboard",
        )

    def on_mount(self) -> None:
        self._refresh_view()

    @on(events.ScreenResume)
    def refresh_on_resume(self) -> None:
        self._refresh_view()

    def _refresh_view(self) -> None:
        cfg = self.app.config
        info = self.query_one("#info", Label)
        parts = [
            "Status: Connected",
            f"MCP: {cfg.get('mcp_url', 'http://localhost:8001/mcp/')}",
            (
                "Mode: Namespace"
                if cfg.get("active_api_key_kind") == "namespace"
                else "Mode: Admin"
            ),
        ]
        if cfg.get("namespace_name"):
            parts.append(f"Namespace: {cfg['namespace_name']} ({cfg['namespace_id']})")
        else:
            parts.append("Namespace: (not selected)")
        info.update("\n".join(parts))

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id == "home-reconfigure":
            self.app.push_screen(SetupScreen())
            return

        screen_map = {
            "nav-namespaces": NamespacesScreen,
            "nav-profiles": ProfilesScreen,
            "nav-collections": CollectionsScreen,
            "nav-query": QueryScreen,
            "nav-ingest": IngestScreen,
            "nav-jobs": JobsScreen,
        }
        screen_cls = screen_map.get(event.button.id)
        if screen_cls:
            self.app.push_screen(screen_cls())


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

    #embedding-profile-fields.hidden {
        display: none;
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
        if not self.app.admin_api_key:
            self.notify("Admin key required to list namespaces", severity="warning")
            return

        try:
            client = self.app.mcp_client_for_key(self.app.admin_api_key)
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
                    "MCP server not available. Run 'make docker-up' first.",
                    severity="error",
                    timeout=10,
                )
            elif "Connection refused" in error_msg or "connect" in error_msg.lower():
                self.notify(
                    "Cannot connect to server. Run 'make docker-up' first.",
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
            client = self.app.mcp_client_for_key(self.app.admin_api_key)
            await client.connect()
            try:
                text = await client.call("create_namespace", {"name": name})
                ns_id = extract_id(text)
                ns_name = extract_name(text)
                ns_api_key = extract_api_key(text)
            finally:
                await client.disconnect()

            self.notify(f"Created namespace: {ns_name or name}", severity="information")
            self.query_one("#ns-name-input", Input).value = ""
            active_key_kind = (
                "namespace"
                if ns_api_key
                else self.app.config.get("active_api_key_kind", "admin")
            )
            is_admin = False if ns_api_key else self.app.config.get("is_admin", False)
            self.app.config = {
                **self.app.config,
                "namespace_id": ns_id,
                "namespace_name": ns_name or name,
                "namespace_api_key": ns_api_key or self.app.namespace_api_key,
                "active_api_key_kind": active_key_kind,
                "is_admin": is_admin,
            }
            self.query_one("#create-form", Container).remove_class("visible")
            self.run_worker(self._load_namespaces(), exclusive=True, group="load")
        except Exception as e:
            error_msg = str(e)
            if "405" in error_msg or "Method Not Allowed" in error_msg:
                self.notify(
                    "MCP server not available. Run 'make docker-up' first.",
                    severity="error",
                    timeout=10,
                )
            elif "Connection refused" in error_msg or "connect" in error_msg.lower():
                self.notify(
                    "Cannot connect to server. Run 'make docker-up' first.",
                    severity="error",
                    timeout=10,
                )
            else:
                self.notify(
                    f"Failed to create namespace: {error_msg}",
                    severity="error",
                )


class ProfilesScreen(Screen):
    """Manage embedding and LLM profiles."""

    CSS = """
    ProfilesScreen {
        layout: grid;
        grid-size: 1;
        grid-rows: auto 1fr auto;
    }

    #header {
        padding: 1;
        background: $boost;
        color: $accent;
    }

    #profiles-table {
        height: 1fr;
    }

    #create-form {
        display: none;
        padding: 1;
        background: $surface;
        border: round $accent;
        height: auto;
    }

    #create-form.visible {
        display: block;
    }

    #create-form Input {
        width: 100%;
        margin-bottom: 1;
    }

    #create-form Select {
        width: 100%;
        margin-bottom: 1;
    }
    """

    PROFILE_KINDS = [
        ("Embedding", "embedding"),
        ("LLM", "llm"),
    ]
    DISTANCE_METRICS = [
        ("Cosine", "cosine"),
        ("L2", "l2"),
        ("Inner Product", "ip"),
    ]

    BINDINGS = [
        ("a", "create_profile", "Create"),
        ("r", "refresh", "Refresh"),
        ("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> None:
        yield Label("Profiles  |  a=Create  r=Refresh  esc=Back", id="header")
        yield DataTable(id="profiles-table")
        yield Container(
            Select(self.PROFILE_KINDS, value="embedding", id="profile-kind"),
            Input(placeholder="Label", id="profile-label-input"),
            Input(placeholder="Provider (e.g. openai)", id="profile-provider-input"),
            Input(placeholder="Model", id="profile-model-input"),
            Input(
                placeholder="Secret / API key",
                id="profile-secret-input",
                password=True,
            ),
            Input(
                placeholder=(
                    "Base URL (optional, use host.docker.internal for local servers)"
                ),
                id="profile-base-url-input",
            ),
            Container(
                Input(
                    placeholder="Dimensions (embedding only)",
                    id="profile-dimensions-input",
                ),
                Select(
                    self.DISTANCE_METRICS,
                    value="cosine",
                    id="profile-distance-metric",
                ),
                id="embedding-profile-fields",
            ),
            Button("Create", id="profile-create-btn", variant="primary"),
            Button("Cancel", id="profile-cancel-btn"),
            id="create-form",
        )

    async def on_mount(self) -> None:
        self._sync_profile_form()
        self.run_worker(self._load_profiles(), exclusive=True, group="load")

    @on(Select.Changed, "#profile-kind")
    def handle_profile_kind_changed(self, _: Select.Changed) -> None:
        self._sync_profile_form()

    async def action_create_profile(self) -> None:
        form = self.query_one("#create-form", Container)
        form.add_class("visible")
        self._sync_profile_form()
        self.query_one("#profile-provider-input", Input).focus()

    async def action_refresh(self) -> None:
        self.run_worker(self._load_profiles(), exclusive=True, group="load")

    async def _load_profiles(self) -> None:
        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                embedding_text = await client.call("list_embedding_profiles")
                llm_text = await client.call("list_llm_profiles")
            finally:
                await client.disconnect()

            profiles = parse_profiles(embedding_text, "embedding") + parse_profiles(
                llm_text, "llm"
            )
            table = self.query_one("#profiles-table", DataTable)
            table.clear()
            table.add_columns("Kind", "ID", "Label", "Provider", "Model")
            for profile in profiles:
                label = profile["label"] or "-"
                table.add_row(
                    profile["kind"],
                    profile["profile_id"][:8],
                    label,
                    profile["provider"],
                    profile["model"],
                )
            if not profiles:
                self.notify("No profiles found", severity="information")
        except Exception as e:
            self.notify(f"Failed to load profiles: {e}", severity="error")

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id == "profile-create-btn":
            self.run_worker(self._create_profile(), exclusive=True, group="action")
        elif event.button.id == "profile-cancel-btn":
            self.query_one("#create-form", Container).remove_class("visible")

    def _sync_profile_form(self) -> None:
        kind = self.query_one("#profile-kind", Select).value or "embedding"
        embedding_fields = self.query_one("#embedding-profile-fields", Container)
        if kind == "embedding":
            embedding_fields.remove_class("hidden")
            return

        self.query_one("#profile-dimensions-input", Input).value = ""
        self.query_one("#profile-distance-metric", Select).value = "cosine"
        embedding_fields.add_class("hidden")

    async def _create_profile(self) -> None:
        kind = self.query_one("#profile-kind", Select).value or "embedding"
        label = self.query_one("#profile-label-input", Input).value.strip()
        provider = self.query_one("#profile-provider-input", Input).value.strip()
        model = self.query_one("#profile-model-input", Input).value.strip()
        secret = self.query_one("#profile-secret-input", Input).value.strip()
        base_url = self.query_one("#profile-base-url-input", Input).value.strip()
        dimensions_text = self.query_one(
            "#profile-dimensions-input",
            Input,
        ).value.strip()
        distance_metric = self.query_one("#profile-distance-metric", Select).value

        if not provider or not model:
            self.notify("Provider and model are required", severity="error")
            return

        if "localhost" in base_url or "127.0.0.1" in base_url:
            self.notify(
                "Local model servers should use host.docker.internal, not localhost.",
                severity="warning",
            )

        dimensions = None
        if kind == "embedding" and dimensions_text:
            try:
                dimensions = int(dimensions_text)
            except ValueError:
                self.notify("Dimensions must be an integer", severity="error")
                return

        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                if kind == "embedding":
                    args = {
                        "provider": provider,
                        "model": model,
                        "secret": secret,
                    }
                    if label:
                        args["label"] = label
                    if base_url:
                        args["base_url"] = base_url
                    if dimensions is not None:
                        args["dimensions"] = dimensions
                    if distance_metric:
                        args["distance_metric"] = distance_metric
                    result = await client.call("create_embedding_profile", args)
                else:
                    args = {
                        "provider": provider,
                        "model": model,
                        "secret": secret,
                    }
                    if label:
                        args["label"] = label
                    if base_url:
                        args["base_url"] = base_url
                    result = await client.call("create_llm_profile", args)
            finally:
                await client.disconnect()

            created_name = extract_name(result) or label or model
            self.notify(f"Created profile: {created_name}", severity="information")
            for widget_id in (
                "#profile-label-input",
                "#profile-provider-input",
                "#profile-model-input",
                "#profile-secret-input",
                "#profile-base-url-input",
                "#profile-dimensions-input",
            ):
                self.query_one(widget_id, Input).value = ""
            self.query_one("#create-form", Container).remove_class("visible")
            self.run_worker(self._load_profiles(), exclusive=True, group="load")
        except Exception as e:
            self.notify(f"Failed to create profile: {e}", severity="error")


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
        width: 100%;
    }

    Select {
        width: 100%;
    }
    """

    BINDINGS = [
        ("a", "create_collection", "Create"),
        ("r", "refresh", "Refresh"),
        ("escape", "app.pop_screen", "Back"),
    ]

    STRATEGIES = [
        ("Vector", "vector"),
        ("Light RAG", "light_rag"),
        ("Graph RAG", "custom_graph_rag"),
    ]
    QUERY_MODES = [
        ("Use default", ""),
        ("local", "local"),
        ("global", "global"),
        ("hybrid", "hybrid"),
        ("naive", "naive"),
        ("mix", "mix"),
    ]

    def compose(self) -> None:
        yield Label("Collections  |  a=Create  r=Refresh  esc=Back", id="header")
        yield DataTable(id="collection-table")
        yield Container(
            Label("Collection Name:"),
            Input(placeholder="Collection name", id="col-name-input"),
            Label("Strategy:"),
            Select(self.STRATEGIES, value="vector", id="col-strategy"),
            Label("Embedding Profile:"),
            Select(
                [("(select embedding profile)", "")],
                allow_blank=True,
                id="col-embedding-profile",
            ),
            Label("LLM Profile (optional):"),
            Select(
                [("(no llm profile)", "")],
                allow_blank=True,
                id="col-llm-profile",
            ),
            Label("Default Query Mode (optional):"),
            Select(self.QUERY_MODES, value="", id="col-query-mode"),
            Button("Create", id="col-create-btn", variant="primary"),
            Button("Cancel", id="col-cancel-btn"),
            id="create-form",
        )

    async def on_mount(self) -> None:
        self.run_worker(self._load_profile_options(), exclusive=True, group="profiles")
        self.run_worker(self._load_collections(), exclusive=True, group="load")

    async def action_create_collection(self) -> None:
        form = self.query_one("#create-form", Container)
        form.add_class("visible")
        self.query_one("#col-name-input", Input).focus()

    async def action_refresh(self) -> None:
        self.run_worker(self._load_profile_options(), exclusive=True, group="profiles")
        self.run_worker(self._load_collections(), exclusive=True, group="load")

    async def _load_profile_options(self) -> None:
        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                embedding_text = await client.call("list_embedding_profiles")
                llm_text = await client.call("list_llm_profiles")
            finally:
                await client.disconnect()

            embedding_profiles = parse_profiles(embedding_text, "embedding")
            llm_profiles = parse_profiles(llm_text, "llm")
            embedding_select = self.query_one("#col-embedding-profile", Select)
            llm_select = self.query_one("#col-llm-profile", Select)
            embedding_select.set_options([
                (
                    profile["label"] or profile["model"],
                    profile["profile_id"],
                )
                for profile in embedding_profiles
            ])
            llm_select.set_options(
                [("(no llm profile)", "")]
                + [
                    (
                        profile["label"] or profile["model"],
                        profile["profile_id"],
                    )
                    for profile in llm_profiles
                ]
            )
            if embedding_profiles:
                embedding_select.value = embedding_profiles[0]["profile_id"]
            else:
                self.notify(
                    "Create an embedding profile before creating a collection.",
                    severity="warning",
                )
            llm_select.value = ""
        except Exception as e:
            self.notify(f"Failed to load profiles: {e}", severity="error")

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
        strategy = strategy_select.value or "vector"
        embedding_profile_id = self.query_one("#col-embedding-profile", Select).value
        llm_profile_id = self.query_one("#col-llm-profile", Select).value
        default_query_mode = self.query_one("#col-query-mode", Select).value
        if not name:
            self.notify("Name is required", severity="error")
            return
        if not embedding_profile_id:
            self.notify("Embedding profile is required", severity="error")
            return

        self.notify(f"Creating collection '{name}'...", severity="information")
        try:
            client = self.app.mcp_client
            await client.connect()
            try:
                args = {
                    "name": name,
                    "strategy": strategy,
                    "embedding_profile_id": embedding_profile_id,
                }
                if llm_profile_id:
                    args["llm_profile_id"] = llm_profile_id
                if default_query_mode:
                    args["default_query_mode"] = default_query_mode
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
            Select(
                [("(select collection)", "")],
                allow_blank=True,
                id="collection-select",
            ),
            Label("  Mode: "),
            Select(
                [
                    ("default", ""),
                    ("local", "local"),
                    ("global", "global"),
                    ("hybrid", "hybrid"),
                    ("naive", "naive"),
                    ("mix", "mix"),
                ],
                value="",
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

            options = [
                (f"{c['name']} ({c['strategy']})", c["id"]) for c in collections
            ]
            sel = self.query_one("#collection-select", Select)
            sel.set_options(options)
            if collections:
                sel.value = collections[0]["id"]
        except Exception as e:
            self.notify(str(e), severity="error")

    @on(Button.Pressed)
    def handle_query(self, event: Button.Pressed) -> None:
        if event.button.id == "query-btn":
            self.run_worker(self._run_query(), exclusive=True, group="action")

    async def _run_query(self) -> None:
        collection_id = self.query_one("#collection-select", Select).value
        mode = self.query_one("#query-mode", Select).value
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
        grid-rows: auto auto 12 auto auto;
    }

    #header {
        padding: 1;
        background: $boost;
        color: $accent;
    }

    #controls {
        padding: 1;
        layout: vertical;
        height: auto;
    }

    #controls-top {
        height: auto;
        margin-bottom: 1;
    }

    #controls-bottom {
        height: auto;
    }

    #text-area {
        width: 100%;
        height: 100%;
        border: none;
    }

    #text-panel {
        height: 12;
        border: round $accent;
        padding: 0 1;
    }

    #text-panel Label {
        margin-top: 0;
    }

    #actions {
        padding: 1;
        align: center middle;
    }

    #actions Button {
        width: 24;
    }

    #status {
        padding: 1;
        dock: bottom;
    }

    Select {
        width: 40%;
    }

    Input {
        width: 1fr;
    }
    """

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> None:
        yield Label("Ingest  |  esc=Back", id="header")
        yield Container(
            Container(
                Label("Collection: "),
                Select([("(select)", "")], allow_blank=True, id="collection-select"),
                Label("  Method: "),
                Select(
                    [("Document (async)", "doc"), ("Chunk (sync)", "chunk")],
                    value="doc",
                    id="ingest-method",
                ),
                id="controls-top",
            ),
            Container(
                Label("File path (optional): "),
                Input(placeholder="/path/to/file.txt", id="file-path"),
                id="controls-bottom",
            ),
            id="controls",
        )
        yield Container(
            Label("Enter text to ingest:"),
            TextArea(
                "Paste text here or provide a file path above.",
                language="plaintext",
                id="text-area",
            ),
            id="text-panel",
        )
        yield Container(
            Button("Ingest", id="ingest-btn", variant="primary"),
            id="actions",
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

            options = [
                (f"{c['name']} ({c['strategy']})", c["id"]) for c in collections
            ]
            sel = self.query_one("#collection-select", Select)
            sel.set_options(options)
            if collections:
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

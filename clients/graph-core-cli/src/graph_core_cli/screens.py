"""Slash-command console screens for Graph Core."""

import asyncio
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from textual import events, on
from textual.containers import Container
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static, TextArea


def parse_namespaces(text: str) -> list[dict]:
    items = []
    for match in re.finditer(r"^  - ([^|]+)\| (.+)$", text, re.MULTILINE):
        parts = match.group(2).split()
        items.append({
            "id": match.group(1).strip(),
            "name": parts[0] if parts else "",
        })
    return items


def parse_collections(text: str) -> list[dict]:
    items = []
    for match in re.finditer(r"^  - ([^|]+)\| (.+?) \((.+?)\)$", text, re.MULTILINE):
        items.append({
            "id": match.group(1).strip(),
            "name": match.group(2).strip(),
            "strategy": match.group(3).strip(),
        })
    return items


def parse_profiles(text: str, kind: str) -> list[dict]:
    items = []
    for match in re.finditer(
        r"^  - ([^|]+)\| ([^|]+)\| ([^|]+)\| ([^|]+)\| max_concurrent_calls=(.+)$",
        text,
        re.MULTILINE,
    ):
        label = match.group(2).strip()
        limit = match.group(5).strip()
        items.append({
            "kind": kind,
            "profile_id": match.group(1).strip(),
            "label": "" if label == "-" else label,
            "provider": match.group(3).strip(),
            "model": match.group(4).strip(),
            "max_concurrent_calls": None if limit == "-" else int(limit),
        })
    return items


def extract_id(text: str) -> str:
    match = re.search(r"id:\s*([^\s\n]+)", text)
    return match.group(1) if match else ""


def extract_name(text: str) -> str:
    match = re.search(r"name:\s*([^\s\n]+)", text)
    return match.group(1) if match else ""


def extract_api_key(text: str) -> str:
    match = re.search(r"api_key:\s*([^\s\n]+)", text)
    return match.group(1) if match else ""


def extract_job_id(text: str) -> str:
    match = re.search(r"job_id:\s*([^\s\n]+)", text)
    return match.group(1) if match else ""


def parse_jobs(text: str) -> list[dict]:
    items = []
    for match in re.finditer(
        r"^  - ([^|]+)\| ([^|]+)\| ([^|]+)\| ([^%\n]+)%(?: \| chunks (\d+)/(\d+))?$",
        text,
        re.MULTILINE,
    ):
        items.append({
            "id": match.group(1).strip(),
            "type": match.group(2).strip(),
            "status": match.group(3).strip(),
            "progress_percent": int(match.group(4).strip()),
            "chunks_completed": int(match.group(5)) if match.group(5) else None,
            "chunks_total": int(match.group(6)) if match.group(6) else None,
        })
    return items


def parse_flag_args(tokens: list[str]) -> tuple[list[str], dict[str, str | bool]]:
    positional: list[str] = []
    flags: dict[str, str | bool] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                flags[key] = tokens[index + 1]
                index += 2
            else:
                flags[key] = True
                index += 1
            continue
        positional.append(token)
        index += 1
    return positional, flags


class SetupScreen(Screen):
    """First-run setup for MCP URL and API key."""

    CSS = """
    SetupScreen {
        align: center middle;
    }

    #setup-card {
        width: 72;
        height: auto;
        border: round $accent;
        padding: 1 2;
    }

    #setup-title {
        color: $accent;
        text-style: bold;
    }

    #setup-hint {
        color: $text-muted;
        margin-bottom: 1;
    }

    Input {
        width: 100%;
        margin-bottom: 1;
    }

    #setup-status {
        color: $error;
        height: auto;
    }
    """

    def compose(self):
        mcp_url = os.getenv("MCP_URL", "http://localhost:8001/mcp/")
        api_key = os.getenv("PLATFORM_ADMIN_KEY", "") or os.getenv(
            "GRAPH_CORE_API_KEY",
            "",
        )
        yield Container(
            Label("Graph Core CLI Setup", id="setup-title"),
            Label(
                "Enter your MCP URL and API key. After this, use slash commands.",
                id="setup-hint",
            ),
            Label("MCP URL"),
            Input(
                value=mcp_url,
                placeholder="http://localhost:8001/mcp/",
                id="mcp-url",
            ),
            Label("API Key"),
            Input(
                value=api_key,
                placeholder="Admin key or namespace key",
                password=True,
                id="api-key",
            ),
            Label("", id="setup-status"),
            id="setup-card",
        )

    def on_mount(self) -> None:
        self.query_one("#api-key", Input).focus()

    @on(Input.Submitted)
    def save_config_and_continue(self) -> None:
        mcp_url = self.query_one("#mcp-url", Input).value.strip()
        api_key = self.query_one("#api-key", Input).value.strip()
        status = self.query_one("#setup-status", Label)

        if not api_key:
            status.update("API key is required.")
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
        self.app.push_screen(ConsoleScreen())


class ConfirmScreen(Screen):
    """Simple yes/no confirmation modal."""

    CSS = """
    ConfirmScreen {
        align: center middle;
    }

    #confirm-card {
        width: 72;
        height: auto;
        border: round $accent;
        padding: 1 2;
    }

    #confirm-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    #confirm-message {
        margin-bottom: 1;
    }

    #confirm-actions {
        height: auto;
    }

    #confirm-actions Button {
        margin-right: 1;
    }
    """

    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self._title = title
        self._message = message

    def compose(self):
        yield Container(
            Label(self._title, id="confirm-title"),
            Label(self._message, id="confirm-message"),
            Container(
                Button("Confirm", id="confirm-ok", variant="error"),
                Button("Cancel", id="confirm-cancel"),
                id="confirm-actions",
            ),
            id="confirm-card",
        )

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-ok":
            self.dismiss(True)
            return
        self.dismiss(False)


class ProfileCreateScreen(Screen):
    """Guided profile creation form."""

    CSS = """
    ProfileCreateScreen {
        align: center middle;
    }

    #profile-card {
        width: 84;
        height: auto;
        border: round $accent;
        padding: 1 2;
    }

    #profile-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    #profile-card Label {
        margin-top: 1;
    }

    #profile-card Input, #profile-card Select {
        width: 100%;
    }

    #profile-embedding-fields {
        height: auto;
    }

    #profile-embedding-fields.hidden {
        display: none;
    }

    #profile-actions {
        height: auto;
        margin-top: 1;
    }

    #profile-actions Button {
        margin-right: 1;
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

    def compose(self):
        yield Container(
            Label("Create Profile", id="profile-title"),
            Label("Kind"),
            Select(self.PROFILE_KINDS, value="embedding", id="profile-kind"),
            Label("Provider"),
            Input(placeholder="openai", id="profile-provider"),
            Label("Model"),
            Input(placeholder="text-embedding-3-large", id="profile-model"),
            Label("Secret"),
            Input(placeholder="API key", password=True, id="profile-secret"),
            Label("Base URL"),
            Input(
                placeholder="http://host.docker.internal:8002/v1",
                id="profile-base-url",
            ),
            Container(
                Label("Dimensions"),
                Input(placeholder="4096", id="profile-dimensions"),
                Label("Distance Metric"),
                Select(
                    self.DISTANCE_METRICS,
                    value="cosine",
                    id="profile-distance-metric",
                ),
                id="profile-embedding-fields",
            ),
            Label("Max Concurrent Calls (optional)"),
            Input(placeholder="1", id="profile-max-concurrent-calls"),
            Label("Label (optional)"),
            Input(placeholder="local-embed", id="profile-label"),
            Container(
                Button("Create", id="profile-submit", variant="primary"),
                Button("Cancel", id="profile-cancel"),
                id="profile-actions",
            ),
            id="profile-card",
        )

    def on_mount(self) -> None:
        self.query_one("#profile-provider", Input).focus()
        self._update_profile_kind_fields("embedding")

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id == "profile-submit":
            self.run_worker(self._submit(), exclusive=True, group="submit")
            return
        self.dismiss(None)

    @on(Select.Changed, "#profile-kind")
    def handle_profile_kind_changed(self, event: Select.Changed) -> None:
        self._update_profile_kind_fields(str(event.value or "embedding"))

    async def _submit(self) -> None:
        kind = self.query_one("#profile-kind", Select).value or "embedding"
        provider = self.query_one("#profile-provider", Input).value.strip()
        model = self.query_one("#profile-model", Input).value.strip()
        secret = self.query_one("#profile-secret", Input).value.strip()
        base_url = self.query_one("#profile-base-url", Input).value.strip()
        label = self.query_one("#profile-label", Input).value.strip()
        dimensions = self.query_one("#profile-dimensions", Input).value.strip()
        max_concurrent_calls = self.query_one(
            "#profile-max-concurrent-calls",
            Input,
        ).value.strip()
        distance_metric = self.query_one(
            "#profile-distance-metric",
            Select,
        ).value

        if not provider or not model or not secret:
            self.notify("Provider, model, and secret are required.", severity="error")
            return

        args: dict[str, str | int] = {
            "provider": provider,
            "model": model,
            "secret": secret,
        }
        if label:
            args["label"] = label
        if base_url:
            args["base_url"] = base_url
        if max_concurrent_calls:
            try:
                args["max_concurrent_calls"] = int(max_concurrent_calls)
            except ValueError:
                self.notify(
                    "Max concurrent calls must be an integer.",
                    severity="error",
                )
                return

        tool_name = "create_llm_profile"
        if kind == "embedding":
            if not dimensions:
                self.notify(
                    "Dimensions are required for embedding profiles.",
                    severity="error",
                )
                return
            try:
                args["dimensions"] = int(dimensions)
            except ValueError:
                self.notify("Dimensions must be an integer.", severity="error")
                return
            if distance_metric:
                args["distance_metric"] = str(distance_metric)
            tool_name = "create_embedding_profile"

        client = self.app.mcp_client_for_key(self.app.active_api_key)
        await client.connect()
        try:
            result = await client.call(tool_name, args)
        finally:
            await client.disconnect()
        self.dismiss(result)

    def _update_profile_kind_fields(self, kind: str) -> None:
        embedding_fields = self.query_one("#profile-embedding-fields", Container)
        dimensions = self.query_one("#profile-dimensions", Input)
        distance_metric = self.query_one("#profile-distance-metric", Select)
        if kind == "embedding":
            embedding_fields.remove_class("hidden")
            return
        embedding_fields.add_class("hidden")
        dimensions.value = ""
        distance_metric.value = "cosine"


class CollectionFormScreen(Screen):
    """Guided collection create/edit form."""

    CSS = """
    CollectionFormScreen {
        align: center middle;
    }

    #collection-card {
        width: 84;
        height: auto;
        border: round $accent;
        padding: 1 2;
    }

    #collection-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    #collection-card Label {
        margin-top: 1;
    }

    #collection-card Input, #collection-card Select {
        width: 100%;
    }

    #collection-actions {
        height: auto;
        margin-top: 1;
    }

    #collection-actions Button {
        margin-right: 1;
    }
    """

    STRATEGIES = [
        ("vector", "vector"),
        ("light_rag", "light_rag"),
        ("custom_graph_rag", "custom_graph_rag"),
    ]
    QUERY_MODES = [
        ("(leave unchanged / none)", ""),
        ("local", "local"),
        ("global", "global"),
        ("hybrid", "hybrid"),
        ("naive", "naive"),
        ("mix", "mix"),
    ]

    def __init__(self, mode: str, collection: dict | None = None) -> None:
        super().__init__()
        self._mode = mode
        self._collection = collection or {}

    def compose(self):
        title = "Create Collection" if self._mode == "create" else "Edit Collection"
        yield Container(
            Label(title, id="collection-title"),
            Label("Name"),
            Input(
                value=self._collection.get("name", ""),
                placeholder="coll1",
                id="collection-name",
            ),
            Label("Strategy"),
            Select(
                self.STRATEGIES,
                value=self._collection.get("strategy", "vector"),
                id="collection-strategy",
            ),
            Label("Embedding Profile"),
            Select(
                [("(select embedding profile)", "")],
                allow_blank=True,
                id="collection-embedding-profile",
            ),
            Label("LLM Profile (optional)"),
            Select(
                [("(none)", "")],
                allow_blank=True,
                id="collection-llm-profile",
            ),
            Label("Default Query Mode"),
            Select(
                self.QUERY_MODES,
                value="",
                allow_blank=True,
                id="collection-query-mode",
            ),
            Container(
                Button(
                    "Save" if self._mode == "edit" else "Create",
                    id="collection-submit",
                    variant="primary",
                ),
                Button("Cancel", id="collection-cancel"),
                id="collection-actions",
            ),
            id="collection-card",
        )

    def on_mount(self) -> None:
        self.query_one("#collection-name", Input).focus()
        self.run_worker(self._load_profiles(), exclusive=True, group="profiles")

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id == "collection-submit":
            self.run_worker(self._submit(), exclusive=True, group="submit")
            return
        self.dismiss(None)

    async def _load_profiles(self) -> None:
        client = self.app.mcp_client_for_key(self.app.active_api_key)
        await client.connect()
        try:
            embedding_text = await client.call("list_embedding_profiles")
            llm_text = await client.call("list_llm_profiles")
        finally:
            await client.disconnect()

        embedding_profiles = parse_profiles(embedding_text, "embedding")
        llm_profiles = parse_profiles(llm_text, "llm")
        embedding_select = self.query_one("#collection-embedding-profile", Select)
        llm_select = self.query_one("#collection-llm-profile", Select)

        embedding_select.set_options([
            (
                profile["label"] or profile["model"],
                profile["profile_id"],
            )
            for profile in embedding_profiles
        ])
        llm_select.set_options(
            [("(none)", "")]
            + [
                (
                    profile["label"] or profile["model"],
                    profile["profile_id"],
                )
                for profile in llm_profiles
            ]
        )

        if self._mode == "create" and embedding_profiles:
            embedding_select.value = embedding_profiles[0]["profile_id"]

    async def _submit(self) -> None:
        name = self.query_one("#collection-name", Input).value.strip()
        strategy = self.query_one("#collection-strategy", Select).value or "vector"
        embedding_profile_id = self.query_one(
            "#collection-embedding-profile",
            Select,
        ).value
        llm_profile_id = self.query_one("#collection-llm-profile", Select).value
        default_query_mode = self.query_one(
            "#collection-query-mode",
            Select,
        ).value

        if not name:
            self.notify("Collection name is required.", severity="error")
            return

        client = self.app.mcp_client_for_key(self.app.active_api_key)
        await client.connect()
        try:
            if self._mode == "create":
                if not embedding_profile_id:
                    self.notify("Embedding profile is required.", severity="error")
                    return
                args: dict[str, str] = {
                    "name": name,
                    "strategy": str(strategy),
                    "embedding_profile_id": str(embedding_profile_id),
                }
                if llm_profile_id:
                    args["llm_profile_id"] = str(llm_profile_id)
                if default_query_mode:
                    args["default_query_mode"] = str(default_query_mode)
                result = await client.call("create_collection", args)
            else:
                args = {
                    "collection_id": str(self._collection["id"]),
                    "name": name,
                    "strategy": str(strategy),
                }
                if embedding_profile_id:
                    args["embedding_profile_id"] = str(embedding_profile_id)
                if llm_profile_id:
                    args["llm_profile_id"] = str(llm_profile_id)
                if default_query_mode:
                    args["default_query_mode"] = str(default_query_mode)
                result = await client.call("update_collection", args)
        finally:
            await client.disconnect()

        self.dismiss(result)


class ConsoleScreen(Screen):
    """Single-screen slash-command console."""

    CSS = """
    ConsoleScreen {
        layout: grid;
        grid-size: 1;
        grid-rows: auto auto 1fr auto;
    }

    #title {
        padding: 1;
        background: $boost;
        color: $accent;
        text-style: bold;
    }

    #context {
        padding: 0 1 1 1;
        color: $text-muted;
    }

    #output {
        border: round $accent;
        margin: 0 1;
        padding: 0 1;
        scrollbar-size-vertical: 1;
    }

    #command-panel {
        margin: 0 1 1 1;
        border: round $accent;
        height: 8;
        padding: 0 1;
    }

    #command-label {
        padding: 0;
        color: $accent;
        text-style: bold;
    }

    #command {
        width: 100%;
        height: 3;
        border: none;
    }

    #suggestions {
        height: 3;
        color: $text-muted;
    }
    """

    COMMAND_HELP = {
        "/help": "Show command help.",
        "/status": "Show current auth and namespace context.",
        "/copy": "Copy selected output, or the full output if nothing is selected.",
        "/clear": "Clear console output.",
        "/quit": "Exit the CLI.",
        "/exit": "Exit the CLI.",
        "/config show": "Show saved MCP URL and key mode.",
        "/config set-url URL": "Update MCP URL.",
        "/auth set-key KEY [--kind admin|namespace|auto]": "Save and switch API key.",
        "/auth use admin|namespace": "Switch active saved key.",
        "/namespace list": "List namespaces with the admin key.",
        "/namespace create NAME": "Create namespace and switch to its namespace key.",
        "/namespace current": "Show current namespace for active namespace key.",
        "/namespace rotate-key ID_OR_NAME": "Rotate namespace key and switch to it.",
        "/profile list [embedding|llm]": "List profiles.",
        (
            "/profile create embedding|llm --provider P --model M --secret S "
            "[--label L] [--base-url U] [--dimensions N] "
            "[--distance-metric cosine] [--max-concurrent-calls N]"
        ): "Create a profile.",
        "/collection list": "List collections in the active namespace.",
        (
            "/collection create NAME --strategy vector|light_rag|custom_graph_rag "
            "--embedding-profile ID_OR_LABEL [--llm-profile ID_OR_LABEL] "
            "[--default-query-mode local|global|hybrid|naive|mix]"
        ): "Create a collection.",
        (
            "/collection edit COLLECTION [--name NAME] "
            "[--strategy vector|light_rag|custom_graph_rag] "
            "[--embedding-profile ID_OR_LABEL] [--llm-profile ID_OR_LABEL] "
            "[--clear-llm-profile] [--default-query-mode MODE] "
            "[--clear-default-query-mode]"
        ): "Update a collection.",
        "/collection delete COLLECTION": "Delete a collection.",
        "/ingest chunk COLLECTION \"text\"": "Ingest a single chunk.",
        "/ingest file COLLECTION /path/to/file.txt": "Ingest a file asynchronously.",
        "/query COLLECTION \"question\" [--mode MODE]": "Query a collection.",
        "/jobs list [--limit N]": "List recent jobs.",
        "/jobs show JOB_ID": "Show job status.",
        "/jobs watch JOB_ID": "Poll a job until it finishes.",
    }
    COMMAND_INSERT_TEXT = {
        "/help": "/help ",
        "/status": "/status",
        "/copy": "/copy",
        "/clear": "/clear",
        "/quit": "/quit",
        "/exit": "/exit",
        "/config show": "/config show",
        "/config set-url URL": "/config set-url ",
        "/auth set-key KEY [--kind admin|namespace|auto]": "/auth set-key ",
        "/auth use admin|namespace": "/auth use ",
        "/namespace list": "/namespace list",
        "/namespace create NAME": "/namespace create ",
        "/namespace current": "/namespace current",
        "/namespace rotate-key ID_OR_NAME": "/namespace rotate-key ",
        "/profile list [embedding|llm]": "/profile list ",
        (
            "/profile create embedding|llm --provider P --model M --secret S "
            "[--label L] [--base-url U] [--dimensions N] "
            "[--distance-metric cosine] [--max-concurrent-calls N]"
        ): "/profile create",
        "/collection list": "/collection list",
        (
            "/collection create NAME --strategy vector|light_rag|custom_graph_rag "
            "--embedding-profile ID_OR_LABEL [--llm-profile ID_OR_LABEL] "
            "[--default-query-mode local|global|hybrid|naive|mix]"
        ): "/collection create",
        (
            "/collection edit COLLECTION [--name NAME] "
            "[--strategy vector|light_rag|custom_graph_rag] "
            "[--embedding-profile ID_OR_LABEL] [--llm-profile ID_OR_LABEL] "
            "[--clear-llm-profile] [--default-query-mode MODE] "
            "[--clear-default-query-mode]"
        ): "/collection edit ",
        "/collection delete COLLECTION": "/collection delete <collection>",
        "/ingest chunk COLLECTION \"text\"": "/ingest chunk <collection> \"<text>\"",
        (
            "/ingest file COLLECTION /path/to/file.txt"
        ): "/ingest file <collection> @<path>",
        (
            "/query COLLECTION \"question\" [--mode MODE]"
        ): "/query <collection> \"<question>\"",
        "/jobs list [--limit N]": "/jobs list",
        "/jobs show JOB_ID": "/jobs show <job_id>",
        "/jobs watch JOB_ID": "/jobs watch <job_id>",
    }
    STRATEGIES = ["vector", "light_rag", "custom_graph_rag"]
    QUERY_MODES = ["local", "global", "hybrid", "naive", "mix"]

    def __init__(self) -> None:
        super().__init__()
        self._history: list[str] = []
        self._history_index = 0
        self._namespace_verified = False
        self._suggestions: list[tuple[str, str]] = []
        self._suggestion_index = 0
        self._file_cache: list[str] = []
        self._last_job_id = ""
        self._output_buffer = ""
        self._query_started_at: float | None = None
        self._query_progress_task: asyncio.Task | None = None

    def compose(self):
        yield Label(
            "Graph Core CLI  |  Slash commands only  |  q=Quit  |  /copy or y=Copy",
            id="title",
        )
        yield Label("", id="context")
        yield TextArea(
            "",
            id="output",
            read_only=True,
            show_line_numbers=False,
            show_cursor=False,
            soft_wrap=True,
            highlight_cursor_line=False,
        )
        yield Container(
            Label("Command", id="command-label"),
            Input(placeholder="/help", id="command"),
            Static("", id="suggestions"),
            id="command-panel",
        )

    def on_mount(self) -> None:
        self._refresh_context()
        self._write("Use /help to see available commands.")
        self.call_after_refresh(self._focus_command)
        self._file_cache = self._collect_files()

    def on_screen_resume(self) -> None:
        self.call_after_refresh(self._focus_command)

    def on_key(self, event: events.Key) -> None:
        output = self.query_one("#output", TextArea)
        if event.key in {"ctrl+shift+c", "ctrl+y"}:
            self._copy_output_selection()
            event.prevent_default()
            event.stop()
            return
        if self.focused is output and event.key == "y":
            self._copy_output_selection()
            event.prevent_default()
            event.stop()
            return
        command_input = self.query_one("#command", Input)
        if event.is_printable and self.focused is not command_input:
            command_input.focus()
            command_input.insert_text_at_cursor(event.character)
            event.prevent_default()
            return
        if self.focused is not command_input:
            return
        if event.key == "ctrl+c":
            command_input.value = ""
            command_input.cursor_position = 0
            self._update_suggestions("")
            event.prevent_default()
            event.stop()
        elif event.key == "tab":
            self._accept_suggestion()
            event.prevent_default()
            event.stop()
        elif event.key == "up":
            if self._should_navigate_suggestions(command_input.value):
                self._move_suggestion(-1)
                event.prevent_default()
                event.stop()
                return
            self._history_move(-1)
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            if self._should_navigate_suggestions(command_input.value):
                self._move_suggestion(1)
                event.prevent_default()
                event.stop()
                return
            self._history_move(1)
            event.prevent_default()
            event.stop()
        elif event.key == "ctrl+p":
            self._move_suggestion(-1)
            event.prevent_default()
            event.stop()
        elif event.key == "ctrl+n":
            self._move_suggestion(1)
            event.prevent_default()
            event.stop()

    @on(Input.Submitted, "#command")
    def handle_command_submit(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        event.input.value = ""
        self._focus_command()
        if not raw:
            return

        self._history.append(raw)
        self._history_index = len(self._history)
        self._write(f"> {raw}")
        self.run_worker(self._execute_command(raw), exclusive=True, group="command")

    @on(Input.Changed, "#command")
    def handle_command_changed(self, event: Input.Changed) -> None:
        self._update_suggestions(event.value)

    async def _execute_command(self, raw: str) -> None:
        try:
            if not raw.startswith("/"):
                self._write_error("Commands must start with '/'. Try /help.")
                return

            parts = shlex.split(raw)
            command = parts[0]
            if command == "/help":
                self._show_help(parts[1:] if len(parts) > 1 else [])
                return
            if command in {"/quit", "/exit"}:
                self.app.exit()
                return
            if command == "/copy":
                self._copy_output_selection()
                return
            if command == "/clear":
                self._clear_output()
                return
            if command == "/status":
                await self._command_status()
                return
            if command == "/config":
                await self._command_config(parts[1:])
                return
            if command == "/auth":
                await self._command_auth(parts[1:])
                return
            if command == "/namespace":
                await self._command_namespace(parts[1:])
                return
            if command == "/profile":
                await self._command_profile(parts[1:])
                return
            if command == "/collection":
                await self._command_collection(parts[1:])
                return
            if command == "/ingest":
                await self._command_ingest(parts[1:])
                return
            if command == "/query":
                await self._command_query(parts[1:])
                return
            if command == "/jobs":
                await self._command_jobs(parts[1:])
                return

            self._write_error(f"Unknown command: {command}")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._write_error(str(exc))
        finally:
            self._refresh_context()

    async def _command_status(self) -> None:
        await self._hydrate_namespace_context()
        cfg = self.app.config
        namespace = "(admin context)"
        if cfg.get("active_api_key_kind") == "namespace":
            namespace = (
                cfg.get("namespace_name")
                if self._namespace_verified
                else "(unverified namespace key)"
            ) or "(not selected)"
        lines = [
            f"MCP URL: {cfg.get('mcp_url', '')}",
            f"Active key: {cfg.get('active_api_key_kind', 'admin')}",
            f"Namespace: {namespace}",
        ]
        if (
            cfg.get("active_api_key_kind") == "namespace"
            and self._namespace_verified
            and cfg.get("namespace_id")
        ):
            lines.append(f"Namespace ID: {cfg['namespace_id']}")
        self._write("\n".join(lines))

    async def _command_config(self, args: list[str]) -> None:
        if not args or args[0] == "show":
            await self._command_status()
            return
        if args[0] == "set-url" and len(args) >= 2:
            cfg = dict(self.app.config)
            cfg["mcp_url"] = args[1]
            self.app.config = cfg
            self._write(f"Updated MCP URL to {args[1]}")
            return
        raise ValueError("Usage: /config show | /config set-url URL")

    async def _command_auth(self, args: list[str]) -> None:
        if not args:
            raise ValueError("Usage: /auth set-key KEY [--kind ...] | /auth use ...")
        if args[0] == "set-key":
            if len(args) < 2:
                raise ValueError(
                    "Usage: /auth set-key KEY [--kind admin|namespace|auto]"
                )
            positional, flags = parse_flag_args(args[1:])
            key = positional[0]
            kind = str(flags.get("kind", "auto"))
            await self._set_key(key, kind)
            return
        if args[0] == "use" and len(args) >= 2:
            target = args[1]
            cfg = dict(self.app.config)
            if target == "admin":
                if not self.app.admin_api_key:
                    raise ValueError("No saved admin key.")
                cfg["active_api_key_kind"] = "admin"
            elif target == "namespace":
                if not self.app.namespace_api_key:
                    raise ValueError("No saved namespace key.")
                cfg["active_api_key_kind"] = "namespace"
            else:
                raise ValueError("Usage: /auth use admin|namespace")
            self.app.config = cfg
            self._write(f"Switched active key to {target}.")
            return
        raise ValueError("Usage: /auth set-key KEY [--kind ...] | /auth use ...")

    async def _command_namespace(self, args: list[str]) -> None:
        if not args:
            raise ValueError("Usage: /namespace list|create|current|rotate-key ...")
        action = args[0]
        if action == "list":
            self._write(await self._call("list_namespaces", admin=True))
            return
        if action == "create" and len(args) >= 2:
            name = " ".join(args[1:])
            text = await self._call("create_namespace", {"name": name}, admin=True)
            namespace_id = extract_id(text)
            namespace_name = extract_name(text) or name
            api_key = extract_api_key(text)
            cfg = dict(self.app.config)
            cfg["namespace_id"] = namespace_id
            cfg["namespace_name"] = namespace_name
            cfg["namespace_api_key"] = api_key
            cfg["active_api_key_kind"] = "namespace"
            self.app.config = cfg
            self._write(text)
            return
        if action == "current":
            text = await self._call("get_current_namespace")
            match = re.search(r"Namespace:\s*([^\s]+)\s*\|\s*(.+)$", text)
            if match:
                cfg = dict(self.app.config)
                cfg["namespace_id"] = match.group(1).strip()
                cfg["namespace_name"] = match.group(2).strip()
                self.app.config = cfg
            self._write(text)
            return
        if action == "rotate-key" and len(args) >= 2:
            target = " ".join(args[1:])
            namespaces = parse_namespaces(
                await self._call("list_namespaces", admin=True)
            )
            namespace = self._resolve_entity(
                target,
                namespaces,
                id_key="id",
                text_keys=["name"],
            )
            text = await self._call(
                "rotate_namespace_key",
                {"namespace_id": namespace["id"]},
                admin=True,
            )
            api_key = extract_api_key(text)
            cfg = dict(self.app.config)
            cfg["namespace_id"] = namespace["id"]
            cfg["namespace_name"] = namespace["name"]
            cfg["namespace_api_key"] = api_key
            cfg["active_api_key_kind"] = "namespace"
            self.app.config = cfg
            self._write(f"Switched to namespace {namespace['name']}.\n{text}")
            return
        raise ValueError("Usage: /namespace list|create|current|rotate-key ...")

    async def _command_profile(self, args: list[str]) -> None:
        if not args:
            raise ValueError(
                "Usage: /profile list [embedding|llm] | /profile create ..."
            )
        action = args[0]
        if action == "list":
            kind = args[1] if len(args) > 1 else "all"
            if kind == "embedding":
                self._write(await self._call("list_embedding_profiles"))
                return
            if kind == "llm":
                self._write(await self._call("list_llm_profiles"))
                return
            if kind == "all":
                embedding = await self._call("list_embedding_profiles")
                llm = await self._call("list_llm_profiles")
                self._write(f"{embedding}\n\n{llm}")
                return
            raise ValueError("Usage: /profile list [embedding|llm]")
        if action == "create" and len(args) == 1:
            self.app.push_screen(
                ProfileCreateScreen(),
                self._handle_modal_result,
            )
            return
        if action == "create" and len(args) >= 2:
            kind = args[1]
            positional, flags = parse_flag_args(args[2:])
            del positional
            provider = self._require_flag(flags, "provider")
            model = self._require_flag(flags, "model")
            secret = self._require_flag(flags, "secret")
            call_args = {
                "provider": provider,
                "model": model,
                "secret": secret,
            }
            self._copy_optional_flag(flags, call_args, "label")
            self._copy_optional_flag(flags, call_args, "base_url")
            if "max_concurrent_calls" in flags:
                call_args["max_concurrent_calls"] = int(
                    str(flags["max_concurrent_calls"])
                )
            if kind == "embedding":
                self._copy_optional_flag(flags, call_args, "distance_metric")
                if "dimensions" in flags:
                    call_args["dimensions"] = int(str(flags["dimensions"]))
                self._write(await self._call("create_embedding_profile", call_args))
                return
            if kind == "llm":
                self._write(await self._call("create_llm_profile", call_args))
                return
        raise ValueError(
            "Usage: /profile list [embedding|llm] | /profile create embedding|llm ..."
        )

    async def _command_collection(self, args: list[str]) -> None:
        if not args:
            raise ValueError(
                "Usage: /collection list | /collection create|edit|delete ..."
            )
        action = args[0]
        if action == "list":
            self._write(await self._call("list_collections"))
            return
        if action == "create" and len(args) == 1:
            self.app.push_screen(
                CollectionFormScreen("create"),
                self._handle_modal_result,
            )
            return
        if action == "create" and len(args) >= 2:
            name = args[1]
            _, flags = parse_flag_args(args[2:])
            strategy = str(flags.get("strategy", "vector"))
            embed_target = self._require_flag(flags, "embedding_profile")
            embedding_profiles = await self._list_profiles("embedding")
            embedding_profile = self._resolve_entity(
                embed_target,
                embedding_profiles,
                id_key="profile_id",
                text_keys=["label", "model"],
            )
            call_args = {
                "name": name,
                "strategy": strategy,
                "embedding_profile_id": embedding_profile["profile_id"],
            }
            llm_target = flags.get("llm_profile")
            if isinstance(llm_target, str) and llm_target:
                llm_profiles = await self._list_profiles("llm")
                llm_profile = self._resolve_entity(
                    llm_target,
                    llm_profiles,
                    id_key="profile_id",
                    text_keys=["label", "model"],
                )
                call_args["llm_profile_id"] = llm_profile["profile_id"]
            if isinstance(flags.get("default_query_mode"), str):
                call_args["default_query_mode"] = str(flags["default_query_mode"])
            self._write(await self._call("create_collection", call_args))
            return
        if action == "edit" and len(args) == 2:
            collection = await self._resolve_collection(args[1])
            self.app.push_screen(
                CollectionFormScreen("edit", collection),
                self._handle_modal_result,
            )
            return
        if action == "edit" and len(args) >= 2:
            target = args[1]
            collection = await self._resolve_collection(target)
            _, flags = parse_flag_args(args[2:])
            call_args: dict[str, str | bool] = {"collection_id": collection["id"]}
            if isinstance(flags.get("name"), str):
                call_args["name"] = str(flags["name"])
            if isinstance(flags.get("strategy"), str):
                call_args["strategy"] = str(flags["strategy"])
            if isinstance(flags.get("embedding_profile"), str):
                embedding_profiles = await self._list_profiles("embedding")
                embedding_profile = self._resolve_entity(
                    str(flags["embedding_profile"]),
                    embedding_profiles,
                    id_key="profile_id",
                    text_keys=["label", "model"],
                )
                call_args["embedding_profile_id"] = embedding_profile["profile_id"]
            if flags.get("clear_llm_profile") is True:
                call_args["clear_llm_profile"] = True
            elif isinstance(flags.get("llm_profile"), str):
                llm_profiles = await self._list_profiles("llm")
                llm_profile = self._resolve_entity(
                    str(flags["llm_profile"]),
                    llm_profiles,
                    id_key="profile_id",
                    text_keys=["label", "model"],
                )
                call_args["llm_profile_id"] = llm_profile["profile_id"]
            if flags.get("clear_default_query_mode") is True:
                call_args["clear_default_query_mode"] = True
            elif isinstance(flags.get("default_query_mode"), str):
                call_args["default_query_mode"] = str(flags["default_query_mode"])
            self._write(await self._call("update_collection", call_args))
            return
        if action == "delete" and len(args) >= 2:
            collection = await self._resolve_collection(args[1])
            self.app.push_screen(
                ConfirmScreen(
                    "Delete Collection",
                    f"Delete collection '{collection['name']}'?",
                ),
                lambda confirmed: self.run_worker(
                    self._handle_collection_delete_confirm(confirmed, collection),
                    exclusive=True,
                    group="delete",
                ),
            )
            return
        raise ValueError(
            "Usage: /collection list | /collection create|edit|delete ..."
        )

    async def _command_ingest(self, args: list[str]) -> None:
        if len(args) < 3:
            raise ValueError(
                "Usage: /ingest chunk COLLECTION \"text\" | "
                "/ingest file COLLECTION PATH"
            )
        action = args[0]
        collection = await self._resolve_collection(args[1])
        payload = self._normalize_file_reference(" ".join(args[2:]))
        if action == "chunk":
            self._write(
                await self._call(
                    "ingest_chunk",
                    {"collection_id": collection["id"], "text": payload},
                )
            )
            return
        if action == "file":
            file_path = Path(self._normalize_file_reference(payload)).expanduser()
            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Unable to read file: {file_path} ({exc})") from exc
            result = await self._call(
                "ingest_document",
                {"collection_id": collection["id"], "text": content},
            )
            job_id = extract_job_id(result)
            if job_id:
                self._last_job_id = job_id
            self._write(result)
            return
        raise ValueError(
            "Usage: /ingest chunk COLLECTION \"text\" | "
            "/ingest file COLLECTION PATH"
        )

    async def _command_query(self, args: list[str]) -> None:
        if len(args) < 2:
            raise ValueError("Usage: /query COLLECTION \"question\" [--mode MODE]")
        positional, flags = parse_flag_args(args)
        if len(positional) < 2:
            raise ValueError("Usage: /query COLLECTION \"question\" [--mode MODE]")
        collection = await self._resolve_collection(positional[0])
        question = " ".join(positional[1:])
        call_args = {"collection_id": collection["id"], "question": question}
        if isinstance(flags.get("mode"), str) and flags["mode"]:
            call_args["mode"] = str(flags["mode"])
        self._start_query_progress(collection["name"])
        try:
            result = await self._call("query_collection", call_args)
        finally:
            self._stop_query_progress()
        self._write(result)

    async def _command_jobs(self, args: list[str]) -> None:
        if not args or args[0] == "list":
            limit = 20
            if len(args) > 1:
                _, flags = parse_flag_args(args[1:])
                if isinstance(flags.get("limit"), str):
                    limit = int(str(flags["limit"]))
            self._write(await self._call("list_jobs", {"limit": limit}))
            return
        if len(args) >= 2 and args[0] == "show":
            job_id = self._resolve_job_id(args[1])
            self._write(await self._call("get_job_status", {"job_id": job_id}))
            return
        if len(args) >= 2 and args[0] == "watch":
            job_id = self._resolve_job_id(args[1])
            await self._watch_job(job_id)
            return
        raise ValueError(
            "Usage: /jobs list [--limit N] | "
            "/jobs show JOB_ID | /jobs watch JOB_ID"
        )

    async def _call(
        self,
        tool_name: str,
        arguments: dict | None = None,
        *,
        admin: bool = False,
    ) -> str:
        api_key = self.app.admin_api_key if admin else self.app.active_api_key
        if not api_key:
            raise ValueError("No API key available for this command.")
        client = self.app.mcp_client_for_key(api_key)
        await client.connect()
        result = ""
        try:
            result = await client.call(tool_name, arguments or {})
        finally:
            await client.disconnect()
        return result

    async def _list_profiles(self, kind: str) -> list[dict]:
        if kind == "embedding":
            text = await self._call("list_embedding_profiles")
            return parse_profiles(text, "embedding")
        return parse_profiles(await self._call("list_llm_profiles"), "llm")

    async def _hydrate_namespace_context(self) -> None:
        cfg = dict(self.app.config)
        if cfg.get("active_api_key_kind") != "namespace":
            self._namespace_verified = False
            if cfg.get("namespace_id") or cfg.get("namespace_name"):
                cfg["namespace_id"] = ""
                cfg["namespace_name"] = ""
                self.app.config = cfg
            self._refresh_context()
            return

        try:
            text = await self._call("get_current_namespace")
        except Exception:
            self._namespace_verified = False
            if cfg.get("namespace_id") or cfg.get("namespace_name"):
                cfg["namespace_id"] = ""
                cfg["namespace_name"] = ""
                self.app.config = cfg
                self._write_error(
                    "Saved namespace context is stale; cleared local namespace state."
                )
            self._refresh_context()
            return

        match = re.search(r"Namespace:\s*([^\s]+)\s*\|\s*(.+)$", text)
        if match:
            self._namespace_verified = True
            cfg["namespace_id"] = match.group(1).strip()
            cfg["namespace_name"] = match.group(2).strip()
            self.app.config = cfg
        self._refresh_context()

    async def _resolve_collection(self, target: str) -> dict:
        collections = parse_collections(await self._call("list_collections"))
        return self._resolve_entity(
            target,
            collections,
            id_key="id",
            text_keys=["name"],
        )

    async def _set_key(self, key: str, kind: str) -> None:
        effective_kind = kind
        if effective_kind == "auto":
            effective_kind = "namespace" if key.startswith("ns_key_") else "admin"
        if effective_kind not in {"admin", "namespace"}:
            raise ValueError("Kind must be admin, namespace, or auto.")
        cfg = dict(self.app.config)
        if effective_kind == "admin":
            cfg["admin_api_key"] = key
            cfg["namespace_id"] = ""
            cfg["namespace_name"] = ""
            self._namespace_verified = False
        else:
            cfg["namespace_api_key"] = key
            self._namespace_verified = False
        cfg["active_api_key_kind"] = effective_kind
        self.app.config = cfg
        self._write(f"Saved {effective_kind} key and switched to it.")

    def _resolve_entity(
        self,
        target: str,
        items: list[dict],
        *,
        id_key: str,
        text_keys: list[str],
    ) -> dict:
        lower_target = target.lower()

        for item in items:
            if str(item[id_key]) == target:
                return item
        prefix_matches = [
            item for item in items if str(item[id_key]).startswith(target)
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]

        exact_matches = []
        for item in items:
            for key in text_keys:
                value = str(item.get(key, "")).strip()
                if value and value.lower() == lower_target:
                    exact_matches.append(item)
                    break
        if len(exact_matches) == 1:
            return exact_matches[0]

        contains_matches = []
        for item in items:
            for key in text_keys:
                value = str(item.get(key, "")).strip()
                if value and lower_target in value.lower():
                    contains_matches.append(item)
                    break
        if len(contains_matches) == 1:
            return contains_matches[0]

        if not items:
            raise ValueError("No matching items found.")
        raise ValueError(f"Could not uniquely resolve '{target}'.")

    def _show_help(self, args: list[str]) -> None:
        if not args:
            lines = [
                f"{command}\n  {text}"
                for command, text in self.COMMAND_HELP.items()
            ]
            self._write("\n".join(lines))
            return
        key = "/" + " ".join(args)
        for command, text in self.COMMAND_HELP.items():
            if command.startswith(key):
                self._write(f"{command}\n  {text}")
                return
        self._write_error(f"No help for {key}")

    def _require_flag(self, flags: dict[str, str | bool], key: str) -> str:
        value = flags.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"--{key.replace('_', '-')} is required")
        return value

    def _copy_optional_flag(
        self,
        flags: dict[str, str | bool],
        target: dict,
        key: str,
    ) -> None:
        value = flags.get(key)
        if isinstance(value, str) and value:
            target[key] = value

    def _history_move(self, delta: int) -> None:
        if not self._history:
            return
        self._history_index = max(
            0,
            min(len(self._history), self._history_index + delta),
        )
        input_widget = self.query_one("#command", Input)
        if self._history_index == len(self._history):
            input_widget.value = ""
            input_widget.cursor_position = 0
            self._focus_command()
            self._update_suggestions(input_widget.value)
            return
        input_widget.value = self._history[self._history_index]
        input_widget.cursor_position = len(input_widget.value)
        self._focus_command()
        self._update_suggestions(input_widget.value)

    def _refresh_context(self) -> None:
        cfg = self.app.config
        key_kind = cfg.get("active_api_key_kind", "admin")
        if key_kind == "namespace":
            namespace = (
                cfg.get("namespace_name")
                if self._namespace_verified
                else "(unverified namespace key)"
            ) or "(not selected)"
        else:
            namespace = "(admin context)"
        query_suffix = ""
        if self._query_started_at is not None:
            elapsed = int(asyncio.get_running_loop().time() - self._query_started_at)
            query_suffix = f"  query=running {elapsed}s"
        try:
            self.query_one("#context", Label).update(
                f"mcp={cfg.get('mcp_url', '')}  key={key_kind}  namespace={namespace}"
                f"{query_suffix}"
            )
        except NoMatches:
            return

    def _write(self, text: str) -> None:
        self._append_output(text)

    def _write_error(self, text: str) -> None:
        self._append_output(f"Error: {text}")

    def _append_output(self, text: str) -> None:
        if self._output_buffer:
            self._output_buffer = f"{self._output_buffer}\n{text}"
        else:
            self._output_buffer = text
        try:
            output = self.query_one("#output", TextArea)
        except NoMatches:
            return
        output.load_text(self._output_buffer)
        lines = self._output_buffer.split("\n")
        output.move_cursor((len(lines) - 1, len(lines[-1])), center=False)
        self.call_after_refresh(output.scroll_end, animate=False)

    def _clear_output(self) -> None:
        self._output_buffer = ""
        try:
            output = self.query_one("#output", TextArea)
        except NoMatches:
            return
        output.load_text("")
        self.call_after_refresh(output.scroll_end, animate=False)

    def _copy_output_selection(self) -> None:
        output = self.query_one("#output", TextArea)
        text = output.selected_text or self._output_buffer
        if not text:
            self.notify("Nothing to copy.", severity="warning")
            return
        self._copy_text_to_clipboard(text)
        self.notify("Copied output to clipboard.")

    def _copy_text_to_clipboard(self, text: str) -> None:
        if sys.platform == "darwin":
            subprocess.run(
                ["pbcopy"],
                input=text,
                text=True,
                check=True,
            )
            return

        if sys.platform.startswith("win"):
            subprocess.run(
                ["clip"],
                input=text,
                text=True,
                check=True,
            )
            return

        if shutil.which("wl-copy"):
            subprocess.run(
                ["wl-copy"],
                input=text,
                text=True,
                check=True,
            )
            return

        if shutil.which("xclip"):
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text,
                text=True,
                check=True,
            )
            return

        if shutil.which("xsel"):
            subprocess.run(
                ["xsel", "--clipboard", "--input"],
                input=text,
                text=True,
                check=True,
            )
            return

        self.app.copy_to_clipboard(text)

    def _start_query_progress(self, collection_name: str) -> None:
        self._stop_query_progress(write_completion=False)
        self._query_started_at = asyncio.get_running_loop().time()
        self._write(f"Query running against {collection_name}...")
        self._refresh_context()
        self._query_progress_task = asyncio.create_task(self._query_progress_loop())

    def _stop_query_progress(self, *, write_completion: bool = True) -> None:
        if self._query_progress_task is not None:
            self._query_progress_task.cancel()
            self._query_progress_task = None
        if self._query_started_at is None:
            return
        elapsed = asyncio.get_running_loop().time() - self._query_started_at
        self._query_started_at = None
        self._refresh_context()
        if write_completion:
            self._write(f"Query completed in {elapsed:.1f}s.")

    async def _query_progress_loop(self) -> None:
        try:
            while self._query_started_at is not None:
                await asyncio.sleep(1)
                self._refresh_context()
        except asyncio.CancelledError:
            return

    def _resolve_job_id(self, token: str) -> str:
        if token == "last":
            if not self._last_job_id:
                raise ValueError("No recent job recorded.")
            return self._last_job_id
        return token

    async def _watch_job(self, job_id: str) -> None:
        self._write(f"Watching job {job_id}...")
        last_snapshot = ""
        for _ in range(120):
            status_text = await self._call("get_job_status", {"job_id": job_id})
            if status_text != last_snapshot:
                self._write(status_text)
                last_snapshot = status_text
            if (
                "status: completed" in status_text
                or "status: failed" in status_text
                or "status: cancelled" in status_text
            ):
                return
            await asyncio.sleep(1)
        self._write("Stopped watching after 120 seconds.")

    def _handle_modal_result(self, result: str | None) -> None:
        if result:
            self._write(result)

    async def _handle_collection_delete_confirm(
        self,
        confirmed: bool | None,
        collection: dict,
    ) -> None:
        if not confirmed:
            return
        self._write(
            await self._call(
                "delete_collection",
                {"collection_id": collection["id"]},
            )
        )

    def _focus_command(self) -> None:
        self.query_one("#command", Input).focus()

    def _should_navigate_suggestions(self, value: str) -> bool:
        return bool(value.strip()) and bool(self._suggestions)

    def _set_command_input(
        self,
        value: str,
        *,
        preserve_placeholder: bool = True,
    ) -> None:
        input_widget = self.query_one("#command", Input)
        input_widget.value = value
        cursor_position = len(value)
        if preserve_placeholder:
            placeholder_index = value.find("<")
            if placeholder_index >= 0:
                cursor_position = placeholder_index
        input_widget.cursor_position = cursor_position
        self._focus_command()
        self._update_suggestions(input_widget.value)

    def _update_suggestions(self, value: str) -> None:
        suggestions: list[tuple[str, str]] = []
        if value.startswith("/") and " " not in value:
            prefix = value
            suggestions = [
                (command, description)
                for command, description in self.COMMAND_HELP.items()
                if command.startswith(prefix)
            ]
        elif "--strategy " in value:
            prefix = value.rsplit("--strategy ", 1)[1].strip()
            suggestions = [
                (strategy, "strategy")
                for strategy in self.STRATEGIES
                if strategy.startswith(prefix)
            ]
        elif "--default-query-mode " in value:
            prefix = value.rsplit("--default-query-mode ", 1)[1].strip()
            suggestions = [
                (mode, "query mode")
                for mode in self.QUERY_MODES
                if mode.startswith(prefix)
            ]
        else:
            file_token = self._extract_file_token(value)
            if file_token is not None:
                needle = file_token[1:].lower()
                matches = [
                    path for path in self._file_cache
                    if needle in path.lower()
                ][:8]
                suggestions = [
                    (f"@{path}", "file")
                    for path in matches
                ]

        self._suggestions = suggestions
        self._suggestion_index = 0
        self._render_suggestions()

    def _render_suggestions(self) -> None:
        widget = self.query_one("#suggestions", Static)
        if not self._suggestions:
            widget.update("")
            return
        lines = []
        for index, (value, description) in enumerate(self._suggestions[:5]):
            prefix = "> " if index == self._suggestion_index else "  "
            lines.append(f"{prefix}{value}  {description}")
        widget.update("\n".join(lines))

    def _move_suggestion(self, delta: int) -> None:
        if not self._suggestions:
            return
        self._suggestion_index = (
            self._suggestion_index + delta
        ) % len(self._suggestions)
        self._render_suggestions()

    def _accept_suggestion(self) -> None:
        if not self._suggestions:
            return
        value, _ = self._suggestions[self._suggestion_index]
        input_widget = self.query_one("#command", Input)
        current = input_widget.value
        if current.startswith("/") and " " not in current:
            inserted = self.COMMAND_INSERT_TEXT.get(value, value)
            self._set_command_input(inserted)
            return

        if "--strategy " in current:
            before, _, _ = current.rpartition("--strategy ")
            new_value = f"{before}--strategy {value}"
            input_widget.value = new_value
            input_widget.cursor_position = len(new_value)
            self._focus_command()
            self._update_suggestions(input_widget.value)
            return

        if "--default-query-mode " in current:
            before, _, _ = current.rpartition("--default-query-mode ")
            new_value = f"{before}--default-query-mode {value}"
            input_widget.value = new_value
            input_widget.cursor_position = len(new_value)
            self._focus_command()
            self._update_suggestions(input_widget.value)
            return

        file_token = self._extract_file_token(current)
        if file_token is None:
            return
        start = current.rfind(file_token)
        if start < 0:
            return
        new_value = f"{current[:start]}{value}"
        self._set_command_input(new_value, preserve_placeholder=False)

    def _extract_file_token(self, value: str) -> str | None:
        match = re.search(r"(^|\s)(@[^\s]*)$", value)
        if not match:
            return None
        token = match.group(2)
        return token if token.startswith("@") else None

    def _normalize_file_reference(self, value: str) -> str:
        return value[1:] if value.startswith("@") else value

    def _collect_files(self) -> list[str]:
        root = Path.cwd()
        excluded = {".git", ".venv", "node_modules", "__pycache__"}
        results: list[str] = []
        for current_root, dirs, files in os.walk(root):
            dirs[:] = [
                directory
                for directory in dirs
                if directory not in excluded
                and not directory.startswith(".mypy_cache")
            ]
            for filename in files:
                if filename.endswith((".pyc", ".pyo")):
                    continue
                full_path = Path(current_root) / filename
                try:
                    results.append(str(full_path.relative_to(root)))
                except ValueError:
                    continue
                if len(results) >= 2000:
                    return results
        return results

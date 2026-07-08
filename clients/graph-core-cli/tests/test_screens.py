from __future__ import annotations

import pytest
from unittest.mock import AsyncMock
from types import SimpleNamespace

from graph_core_cli.screens import ConsoleScreen


@pytest.mark.asyncio
async def test_jobs_cancel_dispatches_to_mcp_cancel(monkeypatch):
    screen = ConsoleScreen()
    screen._resolve_job_id = lambda token: token  # type: ignore[method-assign]
    screen._call = AsyncMock(return_value="Cancelled job: 123")  # type: ignore[method-assign]
    screen._write = lambda text: setattr(screen, "_last_response_text", text)  # type: ignore[method-assign]

    await screen._command_jobs(["cancel", "123"])

    screen._call.assert_awaited_once_with("cancel_job", {"job_id": "123"})
    assert screen._last_response_text == "Cancelled job: 123"


def test_jobs_command_help_includes_cancel(monkeypatch):
    monkeypatch.setattr(
        ConsoleScreen,
        "app",
        property(lambda self: SimpleNamespace(ui_mode="user")),
    )
    screen = ConsoleScreen()

    assert "/jobs cancel JOB_ID" in screen._command_help()
    assert screen._command_insert_text()["/jobs cancel JOB_ID"] == "/jobs cancel <job_id>"

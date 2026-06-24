"""Edge-case tests for notification backends."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from gpt_trader.monitoring.alert_types import Alert, AlertSeverity
from gpt_trader.monitoring.notifications.backends import (
    ConsoleNotificationBackend,
    FileNotificationBackend,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["", "\x00\x01"])
async def test_console_backend_handles_empty_or_nonprintable_messages(message: str, capsys) -> None:
    backend = ConsoleNotificationBackend(use_colors=False)
    alert = Alert(
        severity=AlertSeverity.WARNING,
        title="Edge Message",
        message=message,
        source="tests",
    )

    result = await backend.send(alert)

    assert result is True
    captured = capsys.readouterr()
    assert "Edge Message" in captured.out


@pytest.mark.asyncio
async def test_file_backend_handles_json_serialization_failure(tmp_path) -> None:
    file_path = tmp_path / "alerts.jsonl"
    file_path.write_text("seed\n")

    backend = FileNotificationBackend(file_path=str(file_path))
    alert = Alert(
        severity=AlertSeverity.ERROR,
        title="Serialization Failure",
        message="metadata contains non-serializable object",
        metadata={"bad": Mock()},
    )

    result = await backend.send(alert)

    assert result is False
    assert file_path.read_text() == "seed\n"


@pytest.mark.asyncio
async def test_file_backend_nested_path_creates_parents_and_writes_jsonl(tmp_path) -> None:
    nested_path = tmp_path / "nested" / "alerts" / "out.jsonl"
    backend = FileNotificationBackend(file_path=str(nested_path))
    alert = Alert(
        severity=AlertSeverity.ERROR,
        title="Nested Path",
        message="should create parent directories",
    )

    result = await backend.send(alert)

    assert result is True
    assert nested_path.parent.is_dir()
    assert nested_path.exists()
    lines = nested_path.read_text().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["title"] == "Nested Path"
    assert payload["message"] == "should create parent directories"


@pytest.mark.asyncio
async def test_file_backend_test_connection_creates_missing_parents(tmp_path) -> None:
    nested_path = tmp_path / "connection" / "alerts" / "out.jsonl"
    backend = FileNotificationBackend(file_path=str(nested_path))

    result = await backend.test_connection()

    assert result is True
    assert nested_path.parent.is_dir()
    assert nested_path.exists()

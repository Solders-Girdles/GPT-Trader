"""Unit tests for notification backends."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from gpt_trader.monitoring.alert_types import Alert, AlertSeverity
from gpt_trader.monitoring.notifications.backends import (
    ConsoleNotificationBackend,
    FileNotificationBackend,
)


class TestConsoleNotificationBackend:
    """Tests for ConsoleNotificationBackend."""

    def test_name_property(self) -> None:
        backend = ConsoleNotificationBackend()
        assert backend.name == "console"

    def test_is_enabled_property(self) -> None:
        backend = ConsoleNotificationBackend(enabled=True)
        assert backend.is_enabled is True

        backend = ConsoleNotificationBackend(enabled=False)
        assert backend.is_enabled is False

    @pytest.mark.asyncio
    async def test_send_prints_to_console(self, capsys) -> None:
        backend = ConsoleNotificationBackend(use_colors=False)
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test message",
        )

        result = await backend.send(alert)

        assert result is True
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "Test Alert" in captured.out
        assert "This is a test message" in captured.out

    @pytest.mark.asyncio
    async def test_send_filters_by_severity(self, capsys) -> None:
        backend = ConsoleNotificationBackend(min_severity=AlertSeverity.ERROR)
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="Low Severity Alert",
            message="Should be filtered",
        )

        result = await backend.send(alert)

        assert result is True  # Filtered but not failed
        captured = capsys.readouterr()
        assert captured.out == ""  # Nothing printed

    @pytest.mark.asyncio
    async def test_test_connection_always_true(self) -> None:
        backend = ConsoleNotificationBackend()
        result = await backend.test_connection()
        assert result is True


class TestFileNotificationBackend:
    """Tests for FileNotificationBackend."""

    def test_name_property(self) -> None:
        backend = FileNotificationBackend(file_path="/tmp/alerts.jsonl")
        assert backend.name == "file"

    @pytest.mark.asyncio
    async def test_send_writes_to_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            file_path = f.name

        try:
            backend = FileNotificationBackend(file_path=file_path)
            alert = Alert(
                severity=AlertSeverity.ERROR,
                title="File Test",
                message="Testing file output",
            )

            result = await backend.send(alert)

            assert result is True
            content = Path(file_path).read_text()
            assert "File Test" in content
            assert "Testing file output" in content
        finally:
            Path(file_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_test_connection_checks_writability(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            file_path = f.name

        try:
            backend = FileNotificationBackend(file_path=file_path)
            result = await backend.test_connection()
            assert result is True
        finally:
            Path(file_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_test_connection_fails_for_invalid_path(self) -> None:
        backend = FileNotificationBackend(file_path="/nonexistent/path/alerts.jsonl")
        result = await backend.test_connection()
        assert result is False

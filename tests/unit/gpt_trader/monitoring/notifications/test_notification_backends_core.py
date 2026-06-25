"""Unit tests for notification backends."""

from __future__ import annotations

import os
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from gpt_trader.monitoring.alert_types import Alert, AlertSeverity
from gpt_trader.monitoring.notifications import backends as backends_module
from gpt_trader.monitoring.notifications.backends import (
    ConsoleNotificationBackend,
    FileNotificationBackend,
)
from gpt_trader.utilities import console_logging as console_logging_module
from gpt_trader.utilities.console_logging import ConsoleLogger


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
    async def test_send_writes_to_project_console_sink(self, capsys) -> None:
        output_stream = StringIO()
        output_sink = ConsoleLogger(output_stream=output_stream)
        backend = ConsoleNotificationBackend(use_colors=False, output_sink=output_sink)
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test message",
        )

        result = await backend.send(alert)

        assert result is True
        captured = capsys.readouterr()
        assert captured.out == ""
        output = output_stream.getvalue()
        assert "WARNING" in output
        assert "Test Alert" in output
        assert "This is a test message" in output

    @pytest.mark.asyncio
    async def test_send_uses_default_console_sink_for_cli_fallback(self, capsys) -> None:
        console_logging_module._console_logger = None
        try:
            backend = ConsoleNotificationBackend(use_colors=False)
            alert = Alert(
                severity=AlertSeverity.WARNING,
                title="Fallback Alert",
                message="This is a fallback message",
            )

            result = await backend.send(alert)

            assert result is True
            captured = capsys.readouterr()
            assert "WARNING" in captured.out
            assert "Fallback Alert" in captured.out
            assert "This is a fallback message" in captured.out
        finally:
            console_logging_module._console_logger = None

    @pytest.mark.asyncio
    async def test_send_default_sink_does_not_reuse_cached_stdout(self, capsys) -> None:
        stale_stream = StringIO()
        console_logging_module._console_logger = ConsoleLogger(output_stream=stale_stream)
        try:
            backend = ConsoleNotificationBackend(use_colors=False)
            alert = Alert(
                severity=AlertSeverity.WARNING,
                title="Fresh Stdout Alert",
                message="This should use the active stdout stream",
            )

            result = await backend.send(alert)

            assert result is True
            captured = capsys.readouterr()
            assert "Fresh Stdout Alert" in captured.out
            assert stale_stream.getvalue() == ""
        finally:
            console_logging_module._console_logger = None

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
            f.write("seed\n")

        try:
            backend = FileNotificationBackend(file_path=file_path)
            result = await backend.test_connection()
            assert result is True
            assert Path(file_path).read_text() == "seed\n"
        finally:
            Path(file_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_test_connection_checks_existing_target_directly(self, tmp_path: Path) -> None:
        alert_path = tmp_path / "alerts.jsonl"
        alert_path.mkdir()
        backend = FileNotificationBackend(file_path=str(alert_path))

        result = await backend.test_connection()

        assert result is False
        assert list(tmp_path.iterdir()) == [alert_path]

    @pytest.mark.asyncio
    async def test_test_connection_rejects_trailing_separator_file_path(
        self, tmp_path: Path
    ) -> None:
        alert_path = tmp_path / "alerts.jsonl"
        alert_path.write_text("seed\n")
        backend = FileNotificationBackend(file_path=f"{alert_path}/")

        result = await backend.test_connection()

        assert result is False
        assert alert_path.read_text() == "seed\n"

    @pytest.mark.asyncio
    async def test_test_connection_checks_broken_symlink_target_parent(
        self, tmp_path: Path
    ) -> None:
        target_path = tmp_path / "missing-target-parent" / "alerts.jsonl"
        alert_path = tmp_path / "alerts-link.jsonl"
        alert_path.symlink_to(target_path)
        backend = FileNotificationBackend(file_path=str(alert_path))

        result = await backend.test_connection()

        assert result is False
        assert alert_path.is_symlink()
        assert not target_path.parent.exists()

    @pytest.mark.asyncio
    async def test_test_connection_checks_missing_symlink_target_without_creating_it(
        self, tmp_path: Path
    ) -> None:
        target_parent = tmp_path / "target-parent"
        target_parent.mkdir()
        target_path = target_parent / "alerts.jsonl"
        alert_path = tmp_path / "alerts-link.jsonl"
        alert_path.symlink_to(target_path)
        backend = FileNotificationBackend(file_path=str(alert_path))

        result = await backend.test_connection()

        assert result is True
        assert alert_path.is_symlink()
        assert not target_path.exists()
        assert list(target_parent.iterdir()) == []

    @pytest.mark.asyncio
    async def test_test_connection_creates_parent_without_alert_file(self, tmp_path: Path) -> None:
        alert_path = tmp_path / "nested" / "alerts.jsonl"
        backend = FileNotificationBackend(file_path=str(alert_path))

        result = await backend.test_connection()

        assert result is True
        assert alert_path.parent.is_dir()
        assert not alert_path.exists()
        assert list(alert_path.parent.iterdir()) == []

    @pytest.mark.asyncio
    async def test_test_connection_handles_long_missing_file_name_without_alert_file(
        self, tmp_path: Path
    ) -> None:
        name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
        suffix = ".jsonl"
        alert_path = tmp_path / f"{'a' * (name_max - len(suffix))}{suffix}"
        backend = FileNotificationBackend(file_path=str(alert_path))

        result = await backend.test_connection()

        assert result is True
        assert not alert_path.exists()
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_test_connection_offloads_file_check_to_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[Any, tuple[Any, ...]]] = []

        async def fake_to_thread(func, /, *args, **kwargs):
            calls.append((func, args))
            return func(*args, **kwargs)

        monkeypatch.setattr(backends_module.asyncio, "to_thread", fake_to_thread)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            file_path = f.name

        try:
            backend = FileNotificationBackend(file_path=file_path)
            result = await backend.test_connection()

            assert result is True
            assert calls == [(backend._check_file_writable, ())]
        finally:
            Path(file_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_test_connection_fails_for_invalid_path(self, tmp_path) -> None:
        blocking_file = tmp_path / "not-a-directory"
        blocking_file.write_text("seed\n")
        invalid_path = blocking_file / "alerts.jsonl"
        backend = FileNotificationBackend(file_path=str(invalid_path))

        result = await backend.test_connection()

        assert result is False
        assert blocking_file.read_text() == "seed\n"

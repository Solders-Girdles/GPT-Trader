"""Unit tests for status reporter lifecycle methods."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from gpt_trader.monitoring.status_reporter import StatusReporter


class TestStatusReporterStart:
    """Tests for StatusReporter start method."""

    @pytest.mark.asyncio
    async def test_start_when_disabled(self) -> None:
        reporter = StatusReporter(enabled=False)
        task = await reporter.start()
        assert task is None
        assert reporter._running is False

    @pytest.mark.asyncio
    async def test_start_creates_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                file_write_interval=1,
            )
            task = await reporter.start()

            try:
                assert task is not None
                assert reporter._running is True
                assert reporter._start_time > 0
            finally:
                await reporter.stop()

    @pytest.mark.asyncio
    async def test_start_writes_initial_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                bot_id="test-bot",
            )
            await reporter.start()

            try:
                assert status_file.exists()
                with open(status_file) as f:
                    data = json.load(f)
                assert data["bot_id"] == "test-bot"
                assert "timestamp" in data
            finally:
                await reporter.stop()

    @pytest.mark.asyncio
    async def test_start_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "subdir" / "status.json"
            reporter = StatusReporter(status_file=str(status_file))

            await reporter.start()

            try:
                assert status_file.parent.exists()
                assert status_file.exists()
            finally:
                await reporter.stop()


class TestStatusReporterStop:
    """Tests for StatusReporter stop method."""

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self) -> None:
        reporter = StatusReporter()
        await reporter.stop()
        assert reporter._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                file_write_interval=10,
            )
            await reporter.start()
            await reporter.stop()

            assert reporter._running is False
            assert reporter._task is None

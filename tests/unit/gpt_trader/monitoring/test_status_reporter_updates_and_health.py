"""Unit tests for status reporter updates and health assessment."""

from __future__ import annotations

import json
import tempfile
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest

from gpt_trader.monitoring.metrics_collector import reset_all
from gpt_trader.monitoring.status_reporter import StatusReporter


class TestStatusReporterUpdates:
    """Tests for StatusReporter update methods."""

    def test_record_cycle(self) -> None:
        reporter = StatusReporter()
        assert reporter._cycle_count == 0

        reporter.record_cycle()
        assert reporter._cycle_count == 1

        reporter.record_cycle()
        assert reporter._cycle_count == 2

    def test_record_error(self) -> None:
        reporter = StatusReporter()
        assert reporter._errors_count == 0
        assert reporter._last_error is None

        reporter.record_error("Test error")
        assert reporter._errors_count == 1
        assert reporter._last_error == "Test error"
        assert reporter._last_error_time is not None

    def test_update_price(self) -> None:
        reporter = StatusReporter()
        assert len(reporter._last_prices) == 0

        reporter.update_price("BTC-USD", Decimal("50000.00"))
        assert reporter._last_prices["BTC-USD"] == Decimal("50000.00")
        assert reporter._last_price_update is not None

    def test_update_positions(self) -> None:
        reporter = StatusReporter()
        assert len(reporter._positions) == 0

        positions = {
            "BTC-PERP": {"quantity": Decimal("1.5"), "unrealized_pnl": Decimal("100")},
            "ETH-PERP": {"quantity": Decimal("10"), "unrealized_pnl": Decimal("-50")},
        }
        reporter.update_positions(positions)

        assert len(reporter._positions) == 2
        assert "BTC-PERP" in reporter._positions

    def test_set_heartbeat_service(self) -> None:
        reporter = StatusReporter()
        mock_heartbeat = Mock()
        mock_heartbeat.get_status.return_value = {"enabled": True, "running": True}
        mock_heartbeat.is_healthy = True

        reporter.set_heartbeat_service(mock_heartbeat)
        assert reporter._heartbeat_service is mock_heartbeat


class TestStatusReporterHealth:
    """Tests for StatusReporter health assessment."""

    @pytest.mark.asyncio
    async def test_healthy_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(status_file=str(status_file))

            reset_all()
            await reporter.start()
            reporter.update_price("BTC-USD", Decimal("50000"))

            try:
                status = reporter.get_status()
                assert status.healthy is True
                assert status.health_issues == []
            finally:
                await reporter.stop()

    def test_unhealthy_recent_error(self) -> None:
        reporter = StatusReporter()
        reporter._running = True
        reporter._start_time = time.time()

        reporter.record_error("Something went wrong")
        reporter.update_price("BTC-USD", Decimal("50000"))

        status = reporter.get_status()
        assert status.healthy is False
        assert any("Recent error" in issue for issue in status.health_issues)

    def test_unhealthy_stale_prices(self) -> None:
        reporter = StatusReporter()
        reporter._running = True
        reporter._start_time = time.time()

        reporter._last_prices["BTC-USD"] = Decimal("50000")
        reporter._last_price_update = time.time() - 300  # 5 minutes ago

        status = reporter.get_status()
        assert status.healthy is False
        assert any("Stale prices" in issue for issue in status.health_issues)


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
                data = json.loads(status_file.read_text())
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

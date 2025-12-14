"""Unit tests for the status reporter."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest

from gpt_trader.monitoring.status_reporter import (
    BotStatus,
    EngineStatus,
    MarketStatus,
    PositionStatus,
    StatusReporter,
)


class TestBotStatusDataclass:
    """Tests for BotStatus dataclass."""

    def test_default_values(self) -> None:
        status = BotStatus()
        assert status.bot_id == ""
        assert status.timestamp > 0
        assert status.timestamp_iso.endswith("Z")
        assert status.healthy is True
        assert status.health_issues == []

    def test_engine_status_defaults(self) -> None:
        status = EngineStatus()
        assert status.running is False
        assert status.cycle_count == 0
        assert status.errors_count == 0

    def test_market_status_defaults(self) -> None:
        status = MarketStatus()
        assert status.symbols == []
        assert status.last_prices == {}

    def test_position_status_defaults(self) -> None:
        status = PositionStatus()
        assert status.count == 0
        assert status.symbols == []
        assert status.total_unrealized_pnl == Decimal("0")


class TestStatusReporterInit:
    """Tests for StatusReporter initialization."""

    def test_default_values(self) -> None:
        reporter = StatusReporter()
        assert reporter.status_file == "status.json"
        assert reporter.update_interval == 10
        assert reporter.bot_id == ""
        assert reporter.enabled is True
        assert reporter._running is False

    def test_custom_values(self) -> None:
        reporter = StatusReporter(
            status_file="/tmp/custom_status.json",
            update_interval=30,
            bot_id="test-bot",
            enabled=False,
        )
        assert reporter.status_file == "/tmp/custom_status.json"
        assert reporter.update_interval == 30
        assert reporter.bot_id == "test-bot"
        assert reporter.enabled is False


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
                update_interval=1,
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
        await reporter.stop()  # Should not raise
        assert reporter._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                update_interval=10,
            )
            await reporter.start()
            await reporter.stop()

            assert reporter._running is False
            assert reporter._task is None


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

        # Set old price update
        reporter._last_prices["BTC-USD"] = Decimal("50000")
        reporter._last_price_update = time.time() - 300  # 5 minutes ago

        status = reporter.get_status()
        assert status.healthy is False
        assert any("Stale prices" in issue for issue in status.health_issues)


class TestStatusReporterFileOutput:
    """Tests for StatusReporter file output."""

    @pytest.mark.asyncio
    async def test_writes_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                bot_id="test-bot",
                observer_interval=0.05,
                file_write_interval=0.05,
            )

            await reporter.start()
            reporter.update_price("BTC-USD", Decimal("50000.123"))
            reporter.record_cycle()

            # Wait for file write cycle
            await asyncio.sleep(0.15)

            try:
                with open(status_file) as f:
                    data = json.load(f)

                assert data["bot_id"] == "test-bot"
                assert data["engine"]["cycle_count"] == 1
                assert "BTC-USD" in data["market"]["last_prices"]
            finally:
                await reporter.stop()

    @pytest.mark.asyncio
    async def test_handles_decimal_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                observer_interval=0.05,
                file_write_interval=0.05,
            )

            await reporter.start()
            reporter.update_price("BTC-USD", Decimal("50000.12345678"))
            reporter.update_positions({"BTC-PERP": {"unrealized_pnl": Decimal("123.456")}})

            # Wait for file write cycle
            await asyncio.sleep(0.15)

            try:
                with open(status_file) as f:
                    data = json.load(f)

                # Should be serialized as strings
                assert data["market"]["last_prices"]["BTC-USD"] == "50000.12345678"
            finally:
                await reporter.stop()

    @pytest.mark.asyncio
    async def test_atomic_write(self) -> None:
        """Test that writes are atomic (no partial files)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                observer_interval=0.02,
                file_write_interval=0.02,
            )

            await reporter.start()

            try:
                # Rapid updates - each cycle triggers observer + file write
                for i in range(10):
                    reporter.record_cycle()
                    await asyncio.sleep(0.03)

                # File should always be valid JSON
                with open(status_file) as f:
                    data = json.load(f)
                assert "timestamp" in data
            finally:
                await reporter.stop()


class TestStatusReporterIntegration:
    """Integration tests for StatusReporter."""

    @pytest.mark.asyncio
    async def test_full_status_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                bot_id="integration-test",
                observer_interval=0.05,
                file_write_interval=0.05,
            )

            # Start
            await reporter.start()

            # Simulate activity
            reporter.update_price("BTC-USD", Decimal("50000"))
            reporter.update_price("ETH-USD", Decimal("3000"))
            reporter.record_cycle()
            reporter.update_positions(
                {"BTC-PERP": {"quantity": Decimal("1"), "unrealized_pnl": Decimal("50")}}
            )

            # Wait for file write cycle
            await asyncio.sleep(0.15)

            # Verify status
            with open(status_file) as f:
                data = json.load(f)

            assert data["bot_id"] == "integration-test"
            assert data["engine"]["running"] is True
            assert data["engine"]["cycle_count"] == 1
            assert len(data["market"]["symbols"]) == 2
            assert data["positions"]["count"] == 1

            # Stop
            await reporter.stop()

            # Final status written
            with open(status_file) as f:
                final_data = json.load(f)

            assert final_data["engine"]["running"] is False

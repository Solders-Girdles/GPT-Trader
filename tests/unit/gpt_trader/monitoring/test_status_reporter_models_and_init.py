"""Unit tests for status reporter models and initialization."""

from __future__ import annotations

from decimal import Decimal

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
        assert reporter.file_write_interval == 60.0
        assert reporter.bot_id == ""
        assert reporter.enabled is True
        assert reporter._running is False

    def test_custom_values(self) -> None:
        reporter = StatusReporter(
            status_file="/tmp/custom_status.json",
            file_write_interval=30,
            bot_id="test-bot",
            enabled=False,
        )
        assert reporter.status_file == "/tmp/custom_status.json"
        assert reporter.file_write_interval == 30
        assert reporter.bot_id == "test-bot"
        assert reporter.enabled is False

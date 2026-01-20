"""Unit tests for status reporter models and initialization."""

from __future__ import annotations

import time
from decimal import Decimal

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


class TestStatusReporterMetrics:
    """Tests for metrics recording in StatusReporter."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        from gpt_trader.monitoring.metrics_collector import reset_all

        reset_all()
        yield
        reset_all()

    def test_update_equity_records_gauge(self) -> None:
        """update_equity records the equity gauge metric."""
        from gpt_trader.monitoring.metrics_collector import get_metrics_collector

        reporter = StatusReporter()
        reporter.update_equity(Decimal("10500.50"))

        collector = get_metrics_collector()
        assert "gpt_trader_equity_dollars" in collector.gauges
        assert collector.gauges["gpt_trader_equity_dollars"] == 10500.50

    def test_update_ws_health_records_gap_gauge(self) -> None:
        """update_ws_health records the WS gap count gauge metric."""
        from gpt_trader.monitoring.metrics_collector import get_metrics_collector

        reporter = StatusReporter()
        reporter.update_ws_health(
            {
                "connected": True,
                "last_message_ts": time.time(),
                "last_heartbeat_ts": time.time(),
                "gap_count": 5,
                "reconnect_count": 2,
            }
        )

        collector = get_metrics_collector()
        assert "gpt_trader_ws_gap_count" in collector.gauges
        assert collector.gauges["gpt_trader_ws_gap_count"] == 5.0

    def test_equity_gauge_updates_on_change(self) -> None:
        """equity gauge updates when equity changes."""
        from gpt_trader.monitoring.metrics_collector import get_metrics_collector

        reporter = StatusReporter()
        reporter.update_equity(Decimal("10000.00"))
        reporter.update_equity(Decimal("10500.00"))

        collector = get_metrics_collector()
        assert collector.gauges["gpt_trader_equity_dollars"] == 10500.00


class TestStatusReporterStrategyPerformance:
    """Tests for StatusReporter.update_strategy_performance method."""

    def test_update_strategy_performance_sets_performance(self) -> None:
        """update_strategy_performance sets strategy.performance."""
        reporter = StatusReporter()

        perf_data = {
            "win_rate": 0.58,
            "profit_factor": 1.65,
            "total_trades": 45,
        }

        reporter.update_strategy_performance(performance=perf_data)

        status = reporter.get_status()
        assert status.strategy.performance is not None
        assert status.strategy.performance["win_rate"] == 0.58
        assert status.strategy.performance["total_trades"] == 45

    def test_update_strategy_performance_sets_backtest(self) -> None:
        """update_strategy_performance sets strategy.backtest_performance."""
        reporter = StatusReporter()

        backtest_data = {
            "win_rate": 0.56,
            "profit_factor": 1.42,
            "total_trades": 120,
        }

        reporter.update_strategy_performance(backtest=backtest_data)

        status = reporter.get_status()
        assert status.strategy.backtest_performance is not None
        assert status.strategy.backtest_performance["win_rate"] == 0.56
        assert status.strategy.backtest_performance["total_trades"] == 120

    def test_update_strategy_performance_sets_both(self) -> None:
        """update_strategy_performance can set both at once."""
        reporter = StatusReporter()

        perf_data = {"win_rate": 0.58}
        backtest_data = {"win_rate": 0.56}

        reporter.update_strategy_performance(performance=perf_data, backtest=backtest_data)

        status = reporter.get_status()
        assert status.strategy.performance["win_rate"] == 0.58
        assert status.strategy.backtest_performance["win_rate"] == 0.56

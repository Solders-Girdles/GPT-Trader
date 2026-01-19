"""Unit tests for status reporter strategy performance updates."""

from __future__ import annotations

from gpt_trader.monitoring.status_reporter import StatusReporter


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

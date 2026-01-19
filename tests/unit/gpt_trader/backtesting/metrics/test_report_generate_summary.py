"""Tests for `BacktestReporter.generate_summary`."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

from tests.unit.gpt_trader.backtesting.metrics.report_test_utils import (  # naming: allow
    create_mock_broker,
    create_mock_risk_metrics,
    create_mock_trade_stats,
)

from gpt_trader.backtesting.metrics.report import BacktestReporter
from gpt_trader.backtesting.metrics.risk import RiskMetrics


class TestBacktestReporterGenerateSummary:
    """Tests for generate_summary method."""

    def test_generate_summary_returns_string(self) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = create_mock_trade_stats()
        mock_risk = create_mock_risk_metrics()

        with (
            patch(
                "gpt_trader.backtesting.metrics.report.calculate_trade_statistics",
                return_value=mock_stats,
            ),
            patch(
                "gpt_trader.backtesting.metrics.report.calculate_risk_metrics",
                return_value=mock_risk,
            ),
        ):
            summary = reporter.generate_summary()

            assert isinstance(summary, str)
            assert len(summary) > 0

    def test_generate_summary_includes_performance_section(self) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = create_mock_trade_stats()
        mock_risk = create_mock_risk_metrics()

        with (
            patch(
                "gpt_trader.backtesting.metrics.report.calculate_trade_statistics",
                return_value=mock_stats,
            ),
            patch(
                "gpt_trader.backtesting.metrics.report.calculate_risk_metrics",
                return_value=mock_risk,
            ),
        ):
            summary = reporter.generate_summary()

            assert "PERFORMANCE" in summary
            assert "Initial Equity" in summary
            assert "Final Equity" in summary
            assert "Total Return" in summary

    def test_generate_summary_includes_risk_metrics_section(self) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = create_mock_trade_stats()
        mock_risk = create_mock_risk_metrics()

        with (
            patch(
                "gpt_trader.backtesting.metrics.report.calculate_trade_statistics",
                return_value=mock_stats,
            ),
            patch(
                "gpt_trader.backtesting.metrics.report.calculate_risk_metrics",
                return_value=mock_risk,
            ),
        ):
            summary = reporter.generate_summary()

            assert "RISK METRICS" in summary
            assert "Max Drawdown" in summary
            assert "Sharpe Ratio" in summary
            assert "Sortino Ratio" in summary

    def test_generate_summary_includes_trade_statistics_section(self) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = create_mock_trade_stats()
        mock_risk = create_mock_risk_metrics()

        with (
            patch(
                "gpt_trader.backtesting.metrics.report.calculate_trade_statistics",
                return_value=mock_stats,
            ),
            patch(
                "gpt_trader.backtesting.metrics.report.calculate_risk_metrics",
                return_value=mock_risk,
            ),
        ):
            summary = reporter.generate_summary()

            assert "TRADE STATISTICS" in summary
            assert "Total Trades" in summary
            assert "Win Rate" in summary
            assert "Profit Factor" in summary

    def test_generate_summary_includes_costs_section(self) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = create_mock_trade_stats()
        mock_risk = create_mock_risk_metrics()

        with (
            patch(
                "gpt_trader.backtesting.metrics.report.calculate_trade_statistics",
                return_value=mock_stats,
            ),
            patch(
                "gpt_trader.backtesting.metrics.report.calculate_risk_metrics",
                return_value=mock_risk,
            ),
        ):
            summary = reporter.generate_summary()

            assert "COSTS" in summary
            assert "Total Fees" in summary
            assert "Avg Slippage" in summary
            assert "Funding PnL" in summary

    def test_generate_summary_handles_none_sharpe(self) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = create_mock_trade_stats()
        mock_risk = RiskMetrics(
            max_drawdown_pct=Decimal("15"),
            max_drawdown_usd=Decimal("15000"),
            avg_drawdown_pct=Decimal("5"),
            drawdown_duration_days=5,
            total_return_pct=Decimal("0"),
            annualized_return_pct=Decimal("0"),
            daily_return_avg=Decimal("0"),
            daily_return_std=Decimal("0"),
            sharpe_ratio=None,  # No Sharpe ratio
            sortino_ratio=None,  # No Sortino ratio
            calmar_ratio=None,
            volatility_annualized=Decimal("0"),
            downside_volatility=Decimal("0"),
            max_leverage_used=Decimal("0"),
            avg_leverage_used=Decimal("0"),
            time_in_market_pct=Decimal("0"),
            var_95_daily=Decimal("0"),
            var_99_daily=Decimal("0"),
        )

        with (
            patch(
                "gpt_trader.backtesting.metrics.report.calculate_trade_statistics",
                return_value=mock_stats,
            ),
            patch(
                "gpt_trader.backtesting.metrics.report.calculate_risk_metrics",
                return_value=mock_risk,
            ),
        ):
            summary = reporter.generate_summary()

            assert "Sharpe Ratio:       N/A" in summary
            assert "Sortino Ratio:      N/A" in summary

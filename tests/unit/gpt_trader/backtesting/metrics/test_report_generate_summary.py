"""Tests for `BacktestReporter.generate_summary`."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from tests.unit.gpt_trader.backtesting.metrics.report_test_utils import (  # naming: allow
    create_mock_broker,
    create_mock_risk_metrics,
    create_mock_trade_stats,
)

import gpt_trader.backtesting.metrics.report as report_module
from gpt_trader.backtesting.metrics.report import BacktestReporter
from gpt_trader.backtesting.metrics.risk import RiskMetrics


@dataclass(slots=True)
class SummaryStubs:
    trade_stats: object
    risk_metrics: RiskMetrics
    calculate_trade_statistics: MagicMock
    calculate_risk_metrics: MagicMock


@pytest.fixture
def reporter() -> BacktestReporter:
    return BacktestReporter(create_mock_broker())


@pytest.fixture
def summary_stubs(monkeypatch: pytest.MonkeyPatch) -> SummaryStubs:
    trade_stats = create_mock_trade_stats()
    risk_metrics = create_mock_risk_metrics()

    calculate_trade_statistics = MagicMock(
        name="calculate_trade_statistics",
        return_value=trade_stats,
    )
    calculate_risk_metrics = MagicMock(
        name="calculate_risk_metrics",
        return_value=risk_metrics,
    )

    monkeypatch.setattr(report_module, "calculate_trade_statistics", calculate_trade_statistics)
    monkeypatch.setattr(report_module, "calculate_risk_metrics", calculate_risk_metrics)

    return SummaryStubs(
        trade_stats=trade_stats,
        risk_metrics=risk_metrics,
        calculate_trade_statistics=calculate_trade_statistics,
        calculate_risk_metrics=calculate_risk_metrics,
    )


class TestBacktestReporterGenerateSummary:
    """Tests for generate_summary method."""

    def test_generate_summary_returns_string(
        self, reporter: BacktestReporter, summary_stubs: SummaryStubs
    ) -> None:
        summary = reporter.generate_summary()

        assert isinstance(summary, str)
        assert summary

    def test_generate_summary_includes_performance_section(
        self, reporter: BacktestReporter, summary_stubs: SummaryStubs
    ) -> None:
        summary = reporter.generate_summary()

        assert "PERFORMANCE" in summary
        assert "Initial Equity" in summary
        assert "Final Equity" in summary
        assert "Total Return" in summary

    def test_generate_summary_includes_risk_metrics_section(
        self, reporter: BacktestReporter, summary_stubs: SummaryStubs
    ) -> None:
        summary = reporter.generate_summary()

        assert "RISK METRICS" in summary
        assert "Max Drawdown" in summary
        assert "Sharpe Ratio" in summary
        assert "Sortino Ratio" in summary

    def test_generate_summary_includes_trade_statistics_section(
        self, reporter: BacktestReporter, summary_stubs: SummaryStubs
    ) -> None:
        summary = reporter.generate_summary()

        assert "TRADE STATISTICS" in summary
        assert "Total Trades" in summary
        assert "Win Rate" in summary
        assert "Profit Factor" in summary

    def test_generate_summary_includes_costs_section(
        self, reporter: BacktestReporter, summary_stubs: SummaryStubs
    ) -> None:
        summary = reporter.generate_summary()

        assert "COSTS" in summary
        assert "Total Fees" in summary
        assert "Avg Slippage" in summary
        assert "Funding PnL" in summary

    def test_generate_summary_handles_none_sharpe(
        self, reporter: BacktestReporter, summary_stubs: SummaryStubs
    ) -> None:
        summary_stubs.calculate_risk_metrics.return_value = RiskMetrics(
            max_drawdown_pct=Decimal("15"),
            max_drawdown_usd=Decimal("15000"),
            avg_drawdown_pct=Decimal("5"),
            drawdown_duration_days=5,
            total_return_pct=Decimal("0"),
            annualized_return_pct=Decimal("0"),
            daily_return_avg=Decimal("0"),
            daily_return_std=Decimal("0"),
            sharpe_ratio=None,
            sortino_ratio=None,
            calmar_ratio=None,
            volatility_annualized=Decimal("0"),
            downside_volatility=Decimal("0"),
            max_leverage_used=Decimal("0"),
            avg_leverage_used=Decimal("0"),
            time_in_market_pct=Decimal("0"),
            var_95_daily=Decimal("0"),
            var_99_daily=Decimal("0"),
        )

        summary = reporter.generate_summary()

        assert "Sharpe Ratio:       N/A" in summary
        assert "Sortino Ratio:      N/A" in summary

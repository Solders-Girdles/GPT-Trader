"""Tests for `BacktestReporter.generate_csv_row`."""

from __future__ import annotations

from decimal import Decimal

import pytest
from tests.unit.gpt_trader.backtesting.metrics.report_test_utils import (  # naming: allow
    create_mock_broker,
    create_mock_risk_metrics,
    create_mock_trade_stats,
)

import gpt_trader.backtesting.metrics.report as report_module
from gpt_trader.backtesting.metrics.report import BacktestReporter
from gpt_trader.backtesting.metrics.risk import RiskMetrics


@pytest.fixture()
def broker():
    return create_mock_broker()


@pytest.fixture()
def reporter(broker):
    return BacktestReporter(broker)


@pytest.fixture()
def mock_stats():
    return create_mock_trade_stats()


@pytest.fixture()
def mock_risk():
    return create_mock_risk_metrics()


@pytest.fixture(autouse=True)
def stub_report_calculations(monkeypatch, mock_stats, mock_risk):
    monkeypatch.setattr(report_module, "calculate_trade_statistics", lambda _broker: mock_stats)
    monkeypatch.setattr(report_module, "calculate_risk_metrics", lambda _broker: mock_risk)


class TestBacktestReporterGenerateCsvRow:
    """Tests for generate_csv_row method."""

    def test_generate_csv_row_returns_dict(self, reporter) -> None:
        row = reporter.generate_csv_row()

        assert isinstance(row, dict)

    def test_generate_csv_row_includes_all_keys(self, reporter) -> None:
        expected_keys = [
            "initial_equity",
            "final_equity",
            "total_return_pct",
            "total_return_usd",
            "max_drawdown_pct",
            "max_drawdown_usd",
            "sharpe_ratio",
            "sortino_ratio",
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate_pct",
            "profit_factor",
            "total_fees",
            "avg_slippage_bps",
            "funding_pnl",
        ]

        row = reporter.generate_csv_row()

        for key in expected_keys:
            assert key in row, f"Missing key: {key}"

    def test_generate_csv_row_values_are_numeric(self, reporter) -> None:
        row = reporter.generate_csv_row()

        for key, value in row.items():
            assert isinstance(
                value, (int, float)
            ), f"Key {key} has non-numeric value: {type(value)}"

    def test_generate_csv_row_handles_none_sharpe(self, monkeypatch, reporter) -> None:
        risk_metrics = RiskMetrics(
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

        monkeypatch.setattr(report_module, "calculate_risk_metrics", lambda _broker: risk_metrics)

        row = reporter.generate_csv_row()

        assert row["sharpe_ratio"] == 0.0
        assert row["sortino_ratio"] == 0.0

"""Tests for report generation helpers and outputs."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from tests.unit.gpt_trader.backtesting.metrics.report_test_utils import (  # naming: allow
    create_mock_broker,
    create_mock_risk_metrics,
    create_mock_trade_stats,
)

import gpt_trader.backtesting.metrics.report as report_module
from gpt_trader.backtesting.metrics.report import (
    BacktestReporter,
    generate_backtest_report,
)
from gpt_trader.backtesting.metrics.statistics import TradeStatistics

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 1, 31)


@pytest.fixture
def mock_stats() -> TradeStatistics:
    return create_mock_trade_stats()


@pytest.fixture
def mock_risk():
    return create_mock_risk_metrics()


@pytest.fixture
def metrics_mocks(
    monkeypatch: pytest.MonkeyPatch,
    mock_stats: TradeStatistics,
    mock_risk,
) -> dict[str, MagicMock]:
    trade_stats = MagicMock(return_value=mock_stats)
    risk_metrics = MagicMock(return_value=mock_risk)
    monkeypatch.setattr(report_module, "calculate_trade_statistics", trade_stats)
    monkeypatch.setattr(report_module, "calculate_risk_metrics", risk_metrics)
    return {"trade_stats": trade_stats, "risk_metrics": risk_metrics}


class TestGenerateBacktestReportFunction:
    """Tests for generate_backtest_report convenience function."""

    def test_returns_backtest_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        broker = create_mock_broker()
        mock_stats = create_mock_trade_stats()
        mock_risk = create_mock_risk_metrics()

        mock_calculate_trade_statistics = MagicMock(return_value=mock_stats)
        mock_calculate_risk_metrics = MagicMock(return_value=mock_risk)
        monkeypatch.setattr(
            report_module,
            "calculate_trade_statistics",
            mock_calculate_trade_statistics,
        )
        monkeypatch.setattr(
            report_module,
            "calculate_risk_metrics",
            mock_calculate_risk_metrics,
        )

        result = generate_backtest_report(
            broker=broker,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 1),
        )

        assert result is not None
        assert result.start_date == datetime(2024, 1, 1)
        assert result.end_date == datetime(2024, 2, 1)
        assert result.duration_days == 31
        mock_calculate_trade_statistics.assert_called_once_with(broker)
        mock_calculate_risk_metrics.assert_called_once_with(broker)


class TestBacktestReporterGenerateResult:
    """Tests for generate_result method."""

    def test_generate_result_returns_backtest_result(self, metrics_mocks) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)

        result = reporter.generate_result(
            start_date=START_DATE,
            end_date=END_DATE,
        )

        assert result.start_date == START_DATE
        assert result.end_date == END_DATE
        assert result.duration_days == 30

    def test_generate_result_populates_performance(self, metrics_mocks) -> None:
        broker = create_mock_broker(
            initial_equity=Decimal("100000"),
            final_equity=Decimal("120000"),
            total_return_pct=Decimal("20"),
            total_return_usd=Decimal("20000"),
        )
        reporter = BacktestReporter(broker)

        result = reporter.generate_result(
            start_date=START_DATE,
            end_date=END_DATE,
        )

        assert result.initial_equity == Decimal("100000")
        assert result.final_equity == Decimal("120000")
        assert result.total_return == Decimal("20")
        assert result.total_return_usd == Decimal("20000")

    def test_generate_result_populates_trade_stats(self, metrics_mocks) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)

        result = reporter.generate_result(
            start_date=START_DATE,
            end_date=END_DATE,
        )

        assert result.total_trades == 50
        assert result.winning_trades == 30
        assert result.losing_trades == 20
        assert result.win_rate == Decimal("60")

    def test_generate_result_populates_risk_metrics(self, metrics_mocks) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)

        result = reporter.generate_result(
            start_date=START_DATE,
            end_date=END_DATE,
        )

        assert result.max_drawdown == Decimal("15.5")
        assert result.max_drawdown_usd == Decimal("15500")
        assert result.sharpe_ratio == Decimal("1.5")
        assert result.sortino_ratio == Decimal("2.1")

    def test_generate_result_calculates_unrealized_pnl(self, metrics_mocks) -> None:
        broker = create_mock_broker()
        mock_position1 = MagicMock()
        mock_position1.unrealized_pnl = Decimal("500")
        mock_position2 = MagicMock()
        mock_position2.unrealized_pnl = Decimal("300")
        broker.positions = {"BTC-USD": mock_position1, "ETH-USD": mock_position2}

        reporter = BacktestReporter(broker)

        result = reporter.generate_result(
            start_date=START_DATE,
            end_date=END_DATE,
        )

        assert result.unrealized_pnl == Decimal("800")

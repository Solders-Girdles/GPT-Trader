"""Tests for `BacktestReporter` initialization and cached properties."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from tests.unit.gpt_trader.backtesting.metrics.report_test_utils import (  # naming: allow
    create_mock_broker,
    create_mock_risk_metrics,
    create_mock_trade_stats,
)

import gpt_trader.backtesting.metrics.report as report_module
from gpt_trader.backtesting.metrics.report import BacktestReporter


@pytest.fixture
def trade_statistics_calculator_mock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock_calculator = Mock()
    monkeypatch.setattr(report_module, "calculate_trade_statistics", mock_calculator)
    return mock_calculator


@pytest.fixture
def risk_metrics_calculator_mock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock_calculator = Mock()
    monkeypatch.setattr(report_module, "calculate_risk_metrics", mock_calculator)
    return mock_calculator


class TestBacktestReporterInit:
    """Tests for BacktestReporter initialization."""

    def test_init_stores_broker(self) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)

        assert reporter.broker is broker

    def test_init_stats_are_none(self) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)

        assert reporter._trade_stats is None
        assert reporter._risk_metrics is None


class TestBacktestReporterTradeStatistics:
    """Tests for trade_statistics property."""

    def test_trade_statistics_lazy_loads(
        self,
        trade_statistics_calculator_mock: Mock,
    ) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = create_mock_trade_stats()
        trade_statistics_calculator_mock.return_value = mock_stats

        stats = reporter.trade_statistics

        trade_statistics_calculator_mock.assert_called_once_with(broker)
        assert stats is mock_stats

    def test_trade_statistics_caches_result(
        self,
        trade_statistics_calculator_mock: Mock,
    ) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = create_mock_trade_stats()
        trade_statistics_calculator_mock.return_value = mock_stats

        stats1 = reporter.trade_statistics
        stats2 = reporter.trade_statistics

        # Should only call once due to caching
        trade_statistics_calculator_mock.assert_called_once()
        assert stats1 is stats2


class TestBacktestReporterRiskMetrics:
    """Tests for risk_metrics property."""

    def test_risk_metrics_lazy_loads(
        self,
        risk_metrics_calculator_mock: Mock,
    ) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_risk = create_mock_risk_metrics()
        risk_metrics_calculator_mock.return_value = mock_risk

        risk = reporter.risk_metrics

        risk_metrics_calculator_mock.assert_called_once_with(broker)
        assert risk is mock_risk

    def test_risk_metrics_caches_result(
        self,
        risk_metrics_calculator_mock: Mock,
    ) -> None:
        broker = create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_risk = create_mock_risk_metrics()
        risk_metrics_calculator_mock.return_value = mock_risk

        risk1 = reporter.risk_metrics
        risk2 = reporter.risk_metrics

        risk_metrics_calculator_mock.assert_called_once()
        assert risk1 is risk2

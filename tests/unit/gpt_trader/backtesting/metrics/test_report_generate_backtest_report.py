"""Tests for the `generate_backtest_report` convenience function."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from tests.unit.gpt_trader.backtesting.metrics.report_test_utils import (  # naming: allow
    create_mock_broker,
    create_mock_risk_metrics,
    create_mock_trade_stats,
)

import gpt_trader.backtesting.metrics.report as report_module
from gpt_trader.backtesting.metrics.report import generate_backtest_report


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

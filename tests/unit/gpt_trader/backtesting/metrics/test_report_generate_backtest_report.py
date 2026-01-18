"""Tests for the `generate_backtest_report` convenience function."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from tests.unit.gpt_trader.backtesting.metrics.report_test_utils import (  # naming: allow
    create_mock_broker,
    create_mock_risk_metrics,
    create_mock_trade_stats,
)

from gpt_trader.backtesting.metrics.report import generate_backtest_report


class TestGenerateBacktestReportFunction:
    """Tests for generate_backtest_report convenience function."""

    def test_returns_backtest_result(self) -> None:
        broker = create_mock_broker()
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
            result = generate_backtest_report(
                broker=broker,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 2, 1),
            )

            assert result is not None
            assert result.start_date == datetime(2024, 1, 1)
            assert result.end_date == datetime(2024, 2, 1)
            assert result.duration_days == 31

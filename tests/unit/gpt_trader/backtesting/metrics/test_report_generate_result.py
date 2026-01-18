"""Tests for `BacktestReporter.generate_result`."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from tests.unit.gpt_trader.backtesting.metrics.report_test_utils import (  # naming: allow
    create_mock_broker,
    create_mock_risk_metrics,
    create_mock_trade_stats,
)

from gpt_trader.backtesting.metrics.report import BacktestReporter


class TestBacktestReporterGenerateResult:
    """Tests for generate_result method."""

    def test_generate_result_returns_backtest_result(self) -> None:
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
            result = reporter.generate_result(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
            )

            assert result.start_date == datetime(2024, 1, 1)
            assert result.end_date == datetime(2024, 1, 31)
            assert result.duration_days == 30

    def test_generate_result_populates_performance(self) -> None:
        broker = create_mock_broker(
            initial_equity=Decimal("100000"),
            final_equity=Decimal("120000"),
            total_return_pct=Decimal("20"),
            total_return_usd=Decimal("20000"),
        )
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
            result = reporter.generate_result(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
            )

            assert result.initial_equity == Decimal("100000")
            assert result.final_equity == Decimal("120000")
            assert result.total_return == Decimal("20")
            assert result.total_return_usd == Decimal("20000")

    def test_generate_result_populates_trade_stats(self) -> None:
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
            result = reporter.generate_result(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
            )

            assert result.total_trades == 50
            assert result.winning_trades == 30
            assert result.losing_trades == 20
            assert result.win_rate == Decimal("60")

    def test_generate_result_populates_risk_metrics(self) -> None:
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
            result = reporter.generate_result(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
            )

            assert result.max_drawdown == Decimal("15.5")
            assert result.max_drawdown_usd == Decimal("15500")
            assert result.sharpe_ratio == Decimal("1.5")
            assert result.sortino_ratio == Decimal("2.1")

    def test_generate_result_calculates_unrealized_pnl(self) -> None:
        broker = create_mock_broker()
        # Add mock positions with unrealized PnL
        mock_position1 = MagicMock()
        mock_position1.unrealized_pnl = Decimal("500")
        mock_position2 = MagicMock()
        mock_position2.unrealized_pnl = Decimal("300")
        broker.positions = {"BTC-USD": mock_position1, "ETH-USD": mock_position2}

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
            result = reporter.generate_result(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
            )

            assert result.unrealized_pnl == Decimal("800")

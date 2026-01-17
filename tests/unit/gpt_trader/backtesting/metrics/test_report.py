"""Tests for backtest report generation module."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.backtesting.metrics.report import (
    BacktestReporter,
    generate_backtest_report,
)
from gpt_trader.backtesting.metrics.risk import RiskMetrics
from gpt_trader.backtesting.metrics.statistics import TradeStatistics


def _create_mock_broker(
    initial_equity: Decimal = Decimal("100000"),
    final_equity: Decimal = Decimal("110000"),
    total_return_pct: Decimal = Decimal("10"),
    total_return_usd: Decimal = Decimal("10000"),
    total_fees_paid: Decimal = Decimal("500"),
    funding_pnl: Decimal = Decimal("100"),
) -> MagicMock:
    """Create a mock SimulatedBroker."""
    broker = MagicMock()

    broker.get_statistics.return_value = {
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "total_return_usd": total_return_usd,
        "total_fees_paid": total_fees_paid,
        "funding_pnl": funding_pnl,
        "total_trades": 50,
        "winning_trades": 30,
        "losing_trades": 20,
    }

    # Mock positions for unrealized PnL calculation
    broker.positions = {}

    # Mock equity curve for risk metrics
    broker.equity_curve = [
        (datetime(2024, 1, 1), initial_equity),
        (datetime(2024, 1, 2), initial_equity + Decimal("500")),
        (datetime(2024, 1, 3), final_equity),
    ]

    # Mock orders for trade statistics
    broker.orders = []

    return broker


def _create_mock_trade_stats() -> TradeStatistics:
    """Create mock TradeStatistics."""
    return TradeStatistics(
        total_trades=50,
        winning_trades=30,
        losing_trades=20,
        breakeven_trades=0,
        win_rate=Decimal("60"),
        loss_rate=Decimal("40"),
        profit_factor=Decimal("3.0"),
        net_profit_factor=Decimal("2.0"),
        fee_drag_per_trade=Decimal("10"),
        total_pnl=Decimal("10000"),
        gross_profit=Decimal("15000"),
        gross_loss=Decimal("-5000"),
        avg_profit_per_trade=Decimal("200"),
        avg_win=Decimal("500"),
        avg_loss=Decimal("-250"),
        largest_win=Decimal("2000"),
        largest_loss=Decimal("-1000"),
        avg_position_size_usd=Decimal("10000"),
        max_position_size_usd=Decimal("25000"),
        avg_leverage=Decimal("5.0"),
        max_leverage=Decimal("10.0"),
        avg_slippage_bps=Decimal("2.5"),
        total_fees_paid=Decimal("500"),
        limit_orders_filled=40,
        limit_orders_cancelled=10,
        limit_fill_rate=Decimal("80"),
        avg_hold_time_minutes=Decimal("120"),
        max_hold_time_minutes=Decimal("480"),
        max_consecutive_wins=5,
        max_consecutive_losses=3,
        current_streak=2,
    )


def _create_mock_risk_metrics() -> RiskMetrics:
    """Create mock RiskMetrics."""
    return RiskMetrics(
        max_drawdown_pct=Decimal("15.5"),
        max_drawdown_usd=Decimal("15500"),
        avg_drawdown_pct=Decimal("5.0"),
        drawdown_duration_days=5,
        total_return_pct=Decimal("20"),
        annualized_return_pct=Decimal("73"),
        daily_return_avg=Decimal("0.05"),
        daily_return_std=Decimal("1.5"),
        sharpe_ratio=Decimal("1.5"),
        sortino_ratio=Decimal("2.1"),
        calmar_ratio=Decimal("0.8"),
        volatility_annualized=Decimal("23.7"),
        downside_volatility=Decimal("15.0"),
        max_leverage_used=Decimal("10.0"),
        avg_leverage_used=Decimal("5.0"),
        time_in_market_pct=Decimal("60"),
        var_95_daily=Decimal("2500"),
        var_99_daily=Decimal("4000"),
    )


class TestBacktestReporterInit:
    """Tests for BacktestReporter initialization."""

    def test_init_stores_broker(self) -> None:
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)

        assert reporter.broker is broker

    def test_init_stats_are_none(self) -> None:
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)

        assert reporter._trade_stats is None
        assert reporter._risk_metrics is None


class TestBacktestReporterTradeStatistics:
    """Tests for trade_statistics property."""

    def test_trade_statistics_lazy_loads(self) -> None:
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()

        with patch(
            "gpt_trader.backtesting.metrics.report.calculate_trade_statistics",
            return_value=mock_stats,
        ) as mock_calculator:
            stats = reporter.trade_statistics

            mock_calculator.assert_called_once_with(broker)
            assert stats is mock_stats

    def test_trade_statistics_caches_result(self) -> None:
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()

        with patch(
            "gpt_trader.backtesting.metrics.report.calculate_trade_statistics",
            return_value=mock_stats,
        ) as mock_calculator:
            stats1 = reporter.trade_statistics
            stats2 = reporter.trade_statistics

            # Should only call once due to caching
            mock_calculator.assert_called_once()
            assert stats1 is stats2


class TestBacktestReporterRiskMetrics:
    """Tests for risk_metrics property."""

    def test_risk_metrics_lazy_loads(self) -> None:
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_risk = _create_mock_risk_metrics()

        with patch(
            "gpt_trader.backtesting.metrics.report.calculate_risk_metrics",
            return_value=mock_risk,
        ) as mock_calculator:
            risk = reporter.risk_metrics

            mock_calculator.assert_called_once_with(broker)
            assert risk is mock_risk

    def test_risk_metrics_caches_result(self) -> None:
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_risk = _create_mock_risk_metrics()

        with patch(
            "gpt_trader.backtesting.metrics.report.calculate_risk_metrics",
            return_value=mock_risk,
        ) as mock_calculator:
            risk1 = reporter.risk_metrics
            risk2 = reporter.risk_metrics

            mock_calculator.assert_called_once()
            assert risk1 is risk2


class TestBacktestReporterGenerateResult:
    """Tests for generate_result method."""

    def test_generate_result_returns_backtest_result(self) -> None:
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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
        broker = _create_mock_broker(
            initial_equity=Decimal("100000"),
            final_equity=Decimal("120000"),
            total_return_pct=Decimal("20"),
            total_return_usd=Decimal("20000"),
        )
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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
        broker = _create_mock_broker()
        # Add mock positions with unrealized PnL
        mock_position1 = MagicMock()
        mock_position1.unrealized_pnl = Decimal("500")
        mock_position2 = MagicMock()
        mock_position2.unrealized_pnl = Decimal("300")
        broker.positions = {"BTC-USD": mock_position1, "ETH-USD": mock_position2}

        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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


class TestBacktestReporterGenerateSummary:
    """Tests for generate_summary method."""

    def test_generate_summary_returns_string(self) -> None:
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
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


class TestBacktestReporterGenerateCsvRow:
    """Tests for generate_csv_row method."""

    def test_generate_csv_row_returns_dict(self) -> None:
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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
            row = reporter.generate_csv_row()

            assert isinstance(row, dict)

    def test_generate_csv_row_includes_all_keys(self) -> None:
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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
            row = reporter.generate_csv_row()

            for key in expected_keys:
                assert key in row, f"Missing key: {key}"

    def test_generate_csv_row_values_are_numeric(self) -> None:
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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
            row = reporter.generate_csv_row()

            for key, value in row.items():
                assert isinstance(
                    value, (int, float)
                ), f"Key {key} has non-numeric value: {type(value)}"

    def test_generate_csv_row_handles_none_sharpe(self) -> None:
        broker = _create_mock_broker()
        reporter = BacktestReporter(broker)
        mock_stats = _create_mock_trade_stats()
        mock_risk = RiskMetrics(
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
            row = reporter.generate_csv_row()

            assert row["sharpe_ratio"] == 0.0
            assert row["sortino_ratio"] == 0.0


class TestGenerateBacktestReportFunction:
    """Tests for generate_backtest_report convenience function."""

    def test_returns_backtest_result(self) -> None:
        broker = _create_mock_broker()
        mock_stats = _create_mock_trade_stats()
        mock_risk = _create_mock_risk_metrics()

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

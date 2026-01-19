"""Tests for MetricsAggregator."""

from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.features.strategy_dev.monitor.metrics import (
    MetricsAggregator,
    PerformanceSnapshot,
    TradeRecord,
)


class TestMetricsAggregator:
    @pytest.fixture
    def aggregator(self):
        """Create aggregator with initial equity."""
        return MetricsAggregator(initial_equity=Decimal("10000"))

    def test_record_snapshot(self, aggregator):
        """Test recording snapshots."""
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            equity=Decimal("10100"),
            cash=Decimal("5000"),
            positions_value=Decimal("5100"),
            total_return=0.01,
            daily_return=0.01,
            unrealized_pnl=100.0,
            realized_pnl=0.0,
            drawdown=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            open_positions=1,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
        )

        aggregator.record_snapshot(snapshot)

        assert len(aggregator._snapshots) == 1
        assert aggregator._current_equity == Decimal("10100")

    def test_drawdown_calculation(self, aggregator):
        """Test drawdown tracking."""
        snapshot1 = PerformanceSnapshot(
            timestamp=datetime.now(),
            equity=Decimal("11000"),
            cash=Decimal("5000"),
            positions_value=Decimal("6000"),
            total_return=0.10,
            daily_return=0.10,
            unrealized_pnl=0,
            realized_pnl=0,
            drawdown=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            open_positions=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )
        aggregator.record_snapshot(snapshot1)

        snapshot2 = PerformanceSnapshot(
            timestamp=datetime.now(),
            equity=Decimal("10000"),
            cash=Decimal("5000"),
            positions_value=Decimal("5000"),
            total_return=0.0,
            daily_return=-0.09,
            unrealized_pnl=0,
            realized_pnl=0,
            drawdown=0.09,
            max_drawdown=0.09,
            volatility=0.0,
            open_positions=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )
        aggregator.record_snapshot(snapshot2)

        drawdown = aggregator.get_current_drawdown()
        assert drawdown == pytest.approx(1000 / 11000, rel=0.01)

    def test_sharpe_ratio(self, aggregator):
        """Test Sharpe ratio calculation."""
        for i in range(30):
            ret = 0.01 if i % 3 != 0 else -0.005
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                equity=Decimal("10000"),
                cash=Decimal("5000"),
                positions_value=Decimal("5000"),
                total_return=0.0,
                daily_return=ret,
                unrealized_pnl=0,
                realized_pnl=0,
                drawdown=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                open_positions=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
            )
            aggregator.record_snapshot(snapshot)

        sharpe = aggregator.get_sharpe_ratio()
        assert sharpe > 0

    def test_trade_statistics(self, aggregator):
        """Test trade statistics."""
        trades = [
            TradeRecord("BTC", "buy", Decimal("1"), Decimal("100"), Decimal("110"), pnl=10),
            TradeRecord("BTC", "buy", Decimal("1"), Decimal("100"), Decimal("105"), pnl=5),
            TradeRecord("BTC", "buy", Decimal("1"), Decimal("100"), Decimal("95"), pnl=-5),
        ]

        for trade in trades:
            aggregator.record_trade(trade)

        stats = aggregator.get_trade_statistics()

        assert stats["total_trades"] == 3
        assert stats["winning_trades"] == 2
        assert stats["losing_trades"] == 1
        assert stats["win_rate"] == pytest.approx(0.667, rel=0.01)

    def test_regime_performance(self, aggregator):
        """Test regime performance tracking."""
        regimes = ["BULL_QUIET", "BULL_QUIET", "BEAR_QUIET"]
        returns = [0.02, 0.01, -0.01]

        for regime, ret in zip(regimes, returns):
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                equity=Decimal("10000"),
                cash=Decimal("5000"),
                positions_value=Decimal("5000"),
                total_return=0.0,
                daily_return=ret,
                unrealized_pnl=0,
                realized_pnl=0,
                drawdown=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                open_positions=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                current_regime=regime,
            )
            aggregator.record_snapshot(snapshot)

        perf = aggregator.get_regime_performance()

        assert "BULL_QUIET" in perf
        assert "BEAR_QUIET" in perf
        assert perf["BULL_QUIET"]["sample_count"] == 2

    def test_summary(self, aggregator):
        """Test comprehensive summary."""
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            equity=Decimal("10500"),
            cash=Decimal("5000"),
            positions_value=Decimal("5500"),
            total_return=0.05,
            daily_return=0.01,
            unrealized_pnl=0,
            realized_pnl=0,
            drawdown=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            open_positions=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )
        aggregator.record_snapshot(snapshot)

        summary = aggregator.get_summary()

        assert "portfolio" in summary
        assert "returns" in summary
        assert "risk_adjusted" in summary
        assert "trades" in summary

    def test_reset(self, aggregator):
        """Test reset functionality."""
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            equity=Decimal("10500"),
            cash=Decimal("5000"),
            positions_value=Decimal("5500"),
            total_return=0.05,
            daily_return=0.01,
            unrealized_pnl=0,
            realized_pnl=0,
            drawdown=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            open_positions=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )
        aggregator.record_snapshot(snapshot)
        aggregator.record_trade(TradeRecord("BTC", "buy", Decimal("1"), Decimal("100")))

        aggregator.reset()

        assert len(aggregator._snapshots) == 0
        assert len(aggregator._trades) == 0

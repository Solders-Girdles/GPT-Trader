"""Tests for PerformanceSnapshot."""

from datetime import datetime
from decimal import Decimal

from gpt_trader.features.strategy_dev.monitor.metrics import PerformanceSnapshot


class TestPerformanceSnapshot:
    def test_create_snapshot(self):
        """Test creating a snapshot."""
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            equity=Decimal("10500"),
            cash=Decimal("5000"),
            positions_value=Decimal("5500"),
            total_return=0.05,
            daily_return=0.01,
            unrealized_pnl=500.0,
            realized_pnl=200.0,
            drawdown=0.02,
            max_drawdown=0.05,
            volatility=0.15,
            open_positions=2,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
        )

        assert snapshot.equity == Decimal("10500")
        assert snapshot.win_rate == 0.6

    def test_to_dict(self):
        """Test snapshot serialization."""
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            equity=Decimal("10000"),
            cash=Decimal("5000"),
            positions_value=Decimal("5000"),
            total_return=0.0,
            daily_return=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            drawdown=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            open_positions=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )

        data = snapshot.to_dict()

        assert "portfolio" in data
        assert "returns" in data
        assert "risk" in data

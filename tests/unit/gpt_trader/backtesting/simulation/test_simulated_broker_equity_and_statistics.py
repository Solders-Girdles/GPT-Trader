"""Tests for SimulatedBroker equity curve and statistics tracking."""

from decimal import Decimal

from gpt_trader.backtesting.simulation.broker import SimulatedBroker


class TestSimulatedBrokerEquityTracking:
    """Test equity tracking and drawdown calculation."""

    def test_equity_curve_initially_empty(self) -> None:
        """Test equity curve starts empty."""
        broker = SimulatedBroker()

        assert broker._equity_curve == []

    def test_get_equity_curve(self) -> None:
        """Test get_equity_curve returns stored curve."""
        from datetime import datetime

        broker = SimulatedBroker()
        broker._equity_curve = [
            (datetime(2024, 1, 1), Decimal("100000")),
            (datetime(2024, 1, 2), Decimal("101000")),
        ]

        curve = broker.get_equity_curve()

        assert len(curve) == 2
        assert curve[0][1] == Decimal("100000")

    def test_peak_equity_initialized(self) -> None:
        """Test peak equity starts at initial equity."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("75000"))

        assert broker._peak_equity == Decimal("75000")

    def test_max_drawdown_initially_zero(self) -> None:
        """Test max drawdown starts at zero."""
        broker = SimulatedBroker()

        assert broker._max_drawdown == Decimal("0")
        assert broker._max_drawdown_usd == Decimal("0")


class TestSimulatedBrokerStatistics:
    """Test statistics tracking."""

    def test_total_fees_initially_zero(self) -> None:
        """Test total fees starts at zero."""
        broker = SimulatedBroker()

        assert broker._total_fees_paid == Decimal("0")

    def test_trade_counters_initially_zero(self) -> None:
        """Test trade counters start at zero."""
        broker = SimulatedBroker()

        assert broker._total_trades == 0
        assert broker._winning_trades == 0
        assert broker._losing_trades == 0

    def test_slippage_initially_zero(self) -> None:
        """Test slippage tracking starts at zero."""
        broker = SimulatedBroker()

        assert broker._total_slippage_bps == Decimal("0")

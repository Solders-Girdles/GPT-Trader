"""Tests for SimulatedBroker margin calculations and account info."""

from decimal import Decimal

from gpt_trader.backtesting.simulation.broker import SimulatedBroker


class TestSimulatedBrokerMarginCalculation:
    """Test margin calculation methods."""

    def test_calculate_margin_used_no_positions(self) -> None:
        """Test margin calculation with no positions returns zero."""
        broker = SimulatedBroker()

        margin = broker._calculate_margin_used()

        assert margin == Decimal("0")

    def test_calculate_margin_used_with_position(self) -> None:
        """Test margin calculation with open position."""
        from gpt_trader.core import Position

        broker = SimulatedBroker()
        # Position: 1 BTC at $50000 with 5x leverage
        # Notional = 1 * 50000 = 50000
        # Margin = 50000 / 5 = 10000
        test_position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=5,
        )
        broker.positions["BTC-USD"] = test_position

        margin = broker._calculate_margin_used()

        assert margin == Decimal("10000")

    def test_calculate_margin_used_multiple_positions(self) -> None:
        """Test margin calculation with multiple positions."""
        from gpt_trader.core import Position

        broker = SimulatedBroker()
        # BTC: 1 * 50000 / 5 = 10000
        # ETH: 10 * 3000 / 3 = 10000
        # Total = 20000
        broker.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=5,
        )
        broker.positions["ETH-USD"] = Position(
            symbol="ETH-USD",
            quantity=Decimal("10.0"),
            side="long",
            entry_price=Decimal("3000"),
            mark_price=Decimal("3000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=3,
        )

        margin = broker._calculate_margin_used()

        assert margin == Decimal("20000")

    def test_calculate_margin_used_default_leverage(self) -> None:
        """Test margin calculation defaults to 1x leverage if not set."""
        from gpt_trader.core import Position

        broker = SimulatedBroker()
        # 1 BTC at $50000 with no leverage (defaults to 1x)
        # Margin = 50000 / 1 = 50000
        test_position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=None,  # type: ignore[arg-type]
        )
        broker.positions["BTC-USD"] = test_position

        margin = broker._calculate_margin_used()

        assert margin == Decimal("50000")


class TestSimulatedBrokerOrderCancellation:
    """Test order cancellation."""

    def test_cancel_open_order(self) -> None:
        """Test cancelling an open order succeeds."""
        from gpt_trader.core import (
            Order,
            OrderSide,
            OrderStatus,
            OrderType,
        )

        broker = SimulatedBroker()
        # Add an open order directly
        order = Order(
            id="test-order-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            status=OrderStatus.SUBMITTED,
            price=Decimal("45000"),
        )
        broker._open_orders["test-order-123"] = order

        result = broker.cancel_order("test-order-123")

        assert result is True
        assert "test-order-123" not in broker._open_orders
        assert "test-order-123" in broker._cancelled_orders
        assert broker._cancelled_orders["test-order-123"].status == OrderStatus.CANCELLED


class TestSimulatedBrokerAccountInfoDetailed:
    """Test detailed account info methods."""

    def test_get_account_info_all_fields(self) -> None:
        """Test get_account_info returns all expected fields."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("100000"))

        info = broker.get_account_info()

        assert "cash" in info
        assert "equity" in info
        assert "unrealized_pnl" in info
        assert "realized_pnl" in info
        assert "margin_used" in info

    def test_get_account_info_with_unrealized_pnl(self) -> None:
        """Test get_account_info includes unrealized PnL from positions."""
        from gpt_trader.core import Position

        broker = SimulatedBroker(initial_equity_usd=Decimal("100000"))
        # Add position with unrealized profit
        broker.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("55000"),
            unrealized_pnl=Decimal("5000"),
            realized_pnl=Decimal("0"),
            leverage=5,
        )

        info = broker.get_account_info()

        assert info["unrealized_pnl"] == Decimal("5000")
        # Equity should include unrealized PnL
        assert info["equity"] == Decimal("105000")

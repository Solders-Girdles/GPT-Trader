"""
Test type consolidation for live_trade module.

This test ensures that the live_trade module properly uses core interfaces
and that the adapters correctly convert between core and local types.
"""

import pytest
from decimal import Decimal
from datetime import datetime

# Import core interfaces
from bot_v2.features.brokerages.core.interfaces import (
    Order as CoreOrder,
    Position as CorePosition,
    Quote as CoreQuote,
    OrderType as CoreOrderType,
    OrderSide as CoreOrderSide,
    OrderStatus as CoreOrderStatus,
    TimeInForce as CoreTimeInForce,
)

# Note: live_trade.types now re-exports core types with deprecation warning
# We'll verify the re-export works but test with core types directly

# Import adapters (simplified - only normalization helpers)
from bot_v2.features.live_trade.adapters import to_core_tif, to_core_side, to_core_type, to_decimal


class TestCoreTypeUsage:
    """Test that core types are used throughout."""

    def test_core_order_creation(self):
        """Test creating core Order directly."""
        core_order = CoreOrder(
            id="TEST123",
            client_id="CLIENT456",
            symbol="AAPL",
            side=CoreOrderSide.BUY,
            type=CoreOrderType.LIMIT,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            stop_price=None,
            tif=CoreTimeInForce.GTC,
            status=CoreOrderStatus.SUBMITTED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Verify core fields
        assert core_order.id == "TEST123"
        assert core_order.symbol == "AAPL"
        assert core_order.side == CoreOrderSide.BUY
        assert core_order.type == CoreOrderType.LIMIT
        assert core_order.quantity == Decimal("100")
        assert core_order.price == Decimal("150.50")
        assert core_order.status == CoreOrderStatus.SUBMITTED
        assert core_order.filled_quantity == Decimal("0")

    def test_core_position_creation(self):
        """Test creating core Position directly."""
        core_pos = CorePosition(
            symbol="GOOGL",
            quantity=Decimal("25"),
            entry_price=Decimal("120.00"),
            mark_price=Decimal("125.00"),
            unrealized_pnl=Decimal("125.00"),
            realized_pnl=Decimal("0.00"),
            leverage=None,
            side="long",
        )

        # Verify core fields
        assert core_pos.symbol == "GOOGL"
        assert core_pos.quantity == Decimal("25")
        assert core_pos.entry_price == Decimal("120.00")
        assert core_pos.mark_price == Decimal("125.00")
        assert core_pos.unrealized_pnl == Decimal("125.00")
        assert core_pos.side == "long"


class TestQuoteUsage:
    """Test that Quote type is used correctly."""

    def test_core_quote_creation(self):
        """Test creating core Quote directly."""
        ts = datetime.now()
        core_quote = CoreQuote(
            symbol="SPY",
            bid=Decimal("450.25"),
            ask=Decimal("450.35"),
            last=Decimal("450.30"),
            ts=ts,
        )

        # Verify core fields
        assert core_quote.symbol == "SPY"
        assert core_quote.bid == Decimal("450.25")
        assert core_quote.ask == Decimal("450.35")
        assert core_quote.last == Decimal("450.30")
        assert core_quote.ts == ts


class TestAdapterHelpers:
    """Test adapter normalization helpers."""

    def test_to_core_tif(self):
        """Test converting string TIF to core enum."""
        assert to_core_tif("day") == CoreTimeInForce.GTC
        assert to_core_tif("gtc") == CoreTimeInForce.GTC
        assert to_core_tif("ioc") == CoreTimeInForce.IOC
        assert to_core_tif("fok") == CoreTimeInForce.FOK
        assert to_core_tif("unknown") == CoreTimeInForce.GTC  # Default

    def test_to_core_side(self):
        """Test converting to core OrderSide."""
        assert to_core_side("buy") == CoreOrderSide.BUY
        assert to_core_side("sell") == CoreOrderSide.SELL
        assert to_core_side(CoreOrderSide.BUY) == CoreOrderSide.BUY

    def test_to_decimal(self):
        """Test Decimal conversion."""
        from decimal import Decimal

        assert to_decimal(100) == Decimal("100")
        assert to_decimal("150.50") == Decimal("150.50")
        assert to_decimal(None) is None


class TestImportConsistency:
    """Test that imports are using core interfaces where appropriate."""

    def test_broker_imports(self):
        """Test that broker implementations import from core."""
        import bot_v2.features.live_trade.brokers as brokers_module

        # Check that core types are imported
        assert hasattr(brokers_module, "Order")
        assert hasattr(brokers_module, "Position")
        assert hasattr(brokers_module, "Quote")
        assert hasattr(brokers_module, "OrderType")
        assert hasattr(brokers_module, "OrderSide")

    def test_live_trade_imports(self):
        """Test that live_trade main module imports from core."""
        import bot_v2.features.live_trade.live_trade as live_trade_module

        # Check that core types are imported
        assert hasattr(live_trade_module, "Order")
        assert hasattr(live_trade_module, "Position")
        assert hasattr(live_trade_module, "Quote")

    def test_types_reexport(self):
        """Test that types.py re-exports core types."""
        # The deprecation warning occurs at import time of types module
        # Since it may have been imported earlier, we just verify the re-export
        import bot_v2.features.live_trade.types as types_module

        # Check that core types are re-exported
        assert hasattr(types_module, "Order")
        assert hasattr(types_module, "Position")
        assert hasattr(types_module, "Quote")
        assert hasattr(types_module, "OrderType")
        assert hasattr(types_module, "OrderSide")
        assert hasattr(types_module, "OrderStatus")

        # Verify these are actually the core types
        from bot_v2.features.brokerages.core.interfaces import Order as CoreOrder

        assert types_module.Order is CoreOrder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

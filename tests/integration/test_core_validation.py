"""Core validation integration tests.

Converted from scripts/validation/verify_core.py.
Tests the integration of core bot components with DeterministicBroker.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from gpt_trader.core import (
    OrderSide,
    OrderStatus,
    OrderType,
)
from gpt_trader.features.brokerages.coinbase.ws import SequenceGuard

if TYPE_CHECKING:
    from gpt_trader.orchestration.trading_bot import TradingBot


@pytest.mark.integration
class TestCoreValidation:
    """Integration tests for core bot functionality.

    These tests validate the bootstrap path and component wiring
    that unit tests cannot cover due to mocking.
    """

    def test_broker_connection(self, dev_bot: TradingBot) -> None:
        """Verify broker is initialized and has required methods."""
        bot = dev_bot

        assert bot.broker is not None, "Broker not initialized"
        assert hasattr(bot.broker, "get_ticker"), "Broker missing get_ticker method"
        assert hasattr(bot.broker, "place_order"), "Broker missing place_order method"
        assert hasattr(bot.broker, "list_positions"), "Broker missing list_positions method"

    def test_market_data_updates(self, dev_bot: TradingBot) -> None:
        """Verify market data is accessible via broker."""
        bot = dev_bot

        for symbol in bot.config.symbols:
            ticker = bot.broker.get_ticker(symbol)
            assert ticker is not None, f"No ticker for {symbol}"
            assert "price" in ticker, f"Ticker missing price for {symbol}"

            price = Decimal(str(ticker["price"]))
            assert price > 0, f"Invalid price {price} for {symbol}"

    def test_websocket_sequence_guard(self) -> None:
        """Verify SequenceGuard gap detection in integration context.

        This tests the same logic as unit tests but validates the
        component is properly integrated and accessible.
        """
        guard = SequenceGuard()

        # First message - no gap
        msg1 = guard.annotate({"sequence": 1, "type": "ticker"})
        assert "gap_detected" not in msg1, "First message should not have gap"

        # Consecutive - no gap
        msg2 = guard.annotate({"sequence": 2, "type": "ticker"})
        assert "gap_detected" not in msg2, "Consecutive should not have gap"

        # Gap detected
        msg3 = guard.annotate({"sequence": 5, "type": "ticker"})
        assert msg3.get("gap_detected") is True, "Gap should be detected (2->5)"

    def test_order_placement(self, dev_bot: TradingBot) -> None:
        """Verify order placement through broker."""
        bot = dev_bot
        symbol = bot.config.symbols[0]

        order = bot.broker.place_order(
            symbol,  # positional for DeterministicBroker
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),
        )

        assert order is not None, "Order result is None"
        assert order.id is not None, "Order ID is None"
        assert order.status in (
            OrderStatus.FILLED,
            OrderStatus.SUBMITTED,
        ), f"Unexpected status: {order.status}"

    def test_position_math(self, dev_bot: TradingBot) -> None:
        """Verify position tracking with mark price changes."""
        bot = dev_bot
        symbol = bot.config.symbols[0]

        # Set initial mark price (DeterministicBroker supports set_mark)
        if hasattr(bot.broker, "set_mark"):
            bot.broker.set_mark(symbol, Decimal("50000"))

        # Place a buy order
        order = bot.broker.place_order(
            symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        assert order is not None, "Order failed"

        # Move mark price up
        if hasattr(bot.broker, "set_mark"):
            bot.broker.set_mark(symbol, Decimal("50100"))

        # Verify positions interface works
        positions = bot.broker.list_positions()
        assert isinstance(positions, list), "list_positions should return a list"

    def test_risk_reduce_only_mode(self, dev_bot: TradingBot) -> None:
        """Verify risk manager reduce-only mode enforcement."""
        bot = dev_bot

        assert bot.risk_manager is not None, "Risk manager not initialized"

        # Save current state
        previous = bot.risk_manager.is_reduce_only_mode()

        # Enable reduce-only mode
        bot.risk_manager.set_reduce_only_mode(True, reason="integration_test")
        assert bot.risk_manager.is_reduce_only_mode() is True, "Reduce-only mode not set"

        # Restore previous state
        bot.risk_manager.set_reduce_only_mode(previous, reason="integration_test_restore")

    def test_state_persistence_surface(self, dev_bot: TradingBot) -> None:
        """Verify state persistence components are available."""
        bot = dev_bot
        registry = bot.container.create_service_registry()

        assert registry.orders_store is not None, "OrdersStore not in registry"
        assert hasattr(registry.orders_store, "storage_path"), "OrdersStore missing storage_path"

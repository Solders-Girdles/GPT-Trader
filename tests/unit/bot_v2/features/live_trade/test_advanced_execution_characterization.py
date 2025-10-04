"""
Characterization tests for AdvancedExecutionEngine.

These tests lock in the current behavior of the execution engine
before Phase 1+ refactoring. They focus on integration behavior
between components (normalizer, validator, stop manager, metrics).
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock
from typing import Any

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
    Quote,
    TimeInForce,
    MarketType,
)
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.features.live_trade.advanced_execution_models.models import OrderConfig
from bot_v2.features.live_trade.risk import PositionSizingAdvice


# ============================================================================
# Test Fixtures
# ============================================================================


class MockBroker:
    """Mock broker for testing order placement."""

    def __init__(self) -> None:
        self.orders: list[Order] = []
        self.cancelled_orders: list[str] = []
        self._order_counter = 0
        self._positions: list[Position] = []

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        client_id: str,
        reduce_only: bool = False,
        leverage: int | None = None,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        **kwargs: Any,
    ) -> Order:
        """Place order and return Order instance."""
        self._order_counter += 1
        order = Order(
            id=f"order-{self._order_counter}",
            client_id=client_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=limit_price,
            stop_price=stop_price,
            tif=kwargs.get("tif", TimeInForce.GTC),
            status=OrderStatus.SUBMITTED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at=datetime.now(),
            updated_at=datetime.now(),
        )
        order.reduce_only = reduce_only
        self.orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        self.cancelled_orders.append(order_id)
        return True

    def get_product(self, symbol: str) -> Product:
        """Get product info."""
        return Product(
            symbol=symbol,
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=10,
            contract_size=Decimal("1"),
            funding_rate=Decimal("0.0001"),
            next_funding_time=None,
        )

    def get_quote(self, symbol: str) -> Quote:
        """Get current quote."""
        return Quote(
            symbol=symbol,
            bid=Decimal("50000"),
            ask=Decimal("50001"),
            last=Decimal("50000.5"),
            ts=datetime.now(),
        )

    def get_positions(self) -> list[Position]:
        """Get current positions."""
        return self._positions

    def set_positions(self, positions: list[Position]) -> None:
        """Set positions for testing."""
        self._positions = positions


class MockRiskManager:
    """Mock risk manager for testing."""

    def __init__(self, advice: PositionSizingAdvice | None = None) -> None:
        self.advice = advice
        self.config = RiskConfig(
            enable_dynamic_position_sizing=bool(advice),
            position_sizing_method="intelligent",
            position_sizing_multiplier=1.0,
        )
        self.positions: dict[str, dict[str, Decimal]] = {}
        self.start_of_day_equity = Decimal("10000")

    def size_position(self, context: Any) -> PositionSizingAdvice:
        """Return sizing advice."""
        if self.advice:
            return self.advice
        # Default advice
        return PositionSizingAdvice(
            symbol=context.symbol,
            side=context.side.value,
            target_notional=context.target_notional,
            target_quantity=context.quantity,
            used_dynamic=False,
            reduce_only=False,
            reason="no_risk_manager",
        )

    def pre_trade_validate(self, **kwargs: Any) -> None:
        """Validate trade."""
        pass

    def is_reduce_only_mode(self) -> bool:
        """Check if in reduce-only mode."""
        return False


# ============================================================================
# Characterization Tests
# ============================================================================


class TestOrderPlacementOrchestration:
    """Test full order placement orchestration through all components."""

    def test_market_order_full_flow(self):
        """Market order: normalize → validate → submit → track."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
            client_id="test-market-1",
        )

        # Verify order placed
        assert order is not None
        assert order.id == "order-1"
        assert order.client_id == "test-market-1"
        assert order.symbol == "BTC-PERP"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.MARKET
        assert order.quantity == Decimal("0.1")

        # Verify tracking
        assert order.id in engine.pending_orders
        assert engine.client_order_map["test-market-1"] == order.id

        # Verify metrics
        assert engine.order_metrics["placed"] == 1
        assert engine.order_metrics["rejected"] == 0

    def test_limit_order_full_flow(self):
        """Limit order: normalize → validate → submit → track."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            quantity=Decimal("0.5"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("51000"),
            time_in_force=TimeInForce.GTC,
            client_id="test-limit-1",
        )

        # Verify order placed
        assert order is not None
        assert order.type == OrderType.LIMIT
        assert order.price == Decimal("51000")
        assert order.tif == TimeInForce.GTC

        # Verify tracking
        assert order.id in engine.pending_orders
        assert engine.metrics_reporter.get_metrics_dict()["placed"] == 1

    def test_stop_order_registers_trigger(self):
        """Stop order: registers trigger in StopTriggerManager."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            quantity=Decimal("0.2"),
            order_type=OrderType.STOP,
            stop_price=Decimal("48000"),
            client_id="test-stop-1",
        )

        # Verify order placed
        assert order is not None
        assert order.stop_price == Decimal("48000")

        # Verify stop trigger registered
        assert "test-stop-1" in engine.stop_triggers
        trigger = engine.stop_triggers["test-stop-1"]
        assert trigger.symbol == "BTC-PERP"
        assert trigger.trigger_price == Decimal("48000")
        assert trigger.side == OrderSide.SELL
        assert trigger.quantity == Decimal("0.2")
        assert trigger.triggered is False

    def test_stop_limit_order_validation_requires_limit_price_in_params(self):
        """Stop-limit order: validation currently requires special handling.

        NOTE: Current behavior rejects stop-limit orders without proper
        broker parameter mapping. This is existing validation behavior.
        """
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Currently rejected due to validation logic
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.3"),
            order_type=OrderType.STOP_LIMIT,
            stop_price=Decimal("52000"),
            limit_price=Decimal("52100"),
            client_id="test-stop-limit-1",
        )

        # Current behavior: rejected
        assert order is None
        assert engine.order_metrics["rejected"] == 1


class TestDuplicateOrderHandling:
    """Test duplicate order detection across components."""

    def test_duplicate_client_id_returns_existing_order(self):
        """Duplicate client_id returns original order, no new submission."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Place first order
        order1 = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
            client_id="duplicate-test",
        )

        # Attempt duplicate
        order2 = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,  # Different side
            quantity=Decimal("0.5"),  # Different quantity
            order_type=OrderType.LIMIT,  # Different type
            limit_price=Decimal("51000"),
            client_id="duplicate-test",  # SAME client_id
        )

        # Should return same order
        assert order1 is not None
        assert order2 is not None
        assert order1.id == order2.id
        assert order2.side == OrderSide.BUY  # Original side
        assert order2.quantity == Decimal("0.1")  # Original quantity

        # Only one order submitted to broker
        assert len(broker.orders) == 1
        assert engine.order_metrics["placed"] == 1

    def test_unique_client_ids_create_separate_orders(self):
        """Different client_ids create separate orders."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        order1 = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
            client_id="order-1",
        )

        order2 = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
            client_id="order-2",
        )

        # Different orders
        assert order1.id != order2.id
        assert len(broker.orders) == 2
        assert engine.order_metrics["placed"] == 2


class TestValidationIntegration:
    """Test validation pipeline integration."""

    def test_post_only_crossing_buy_rejected(self):
        """Post-only buy at/above ask is rejected by validator."""
        broker = MockBroker()
        config = OrderConfig(reject_on_cross=True)
        engine = AdvancedExecutionEngine(broker=broker, config=config)

        # Ask is 50001, buying at ask would cross
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50001"),  # At ask
            post_only=True,
        )

        # Should be rejected
        assert order is None
        assert engine.order_metrics["post_only_rejected"] == 1
        assert engine.order_metrics["rejected"] == 1
        assert len(broker.orders) == 0

    def test_post_only_crossing_sell_rejected(self):
        """Post-only sell at/below bid is rejected by validator."""
        broker = MockBroker()
        config = OrderConfig(reject_on_cross=True)
        engine = AdvancedExecutionEngine(broker=broker, config=config)

        # Bid is 50000, selling at bid would cross
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50000"),  # At bid
            post_only=True,
        )

        # Should be rejected
        assert order is None
        assert engine.order_metrics["post_only_rejected"] == 1

    def test_post_only_not_crossing_accepted(self):
        """Post-only order that doesn't cross is accepted."""
        broker = MockBroker()
        config = OrderConfig(reject_on_cross=True)
        engine = AdvancedExecutionEngine(broker=broker, config=config)

        # Buy below ask
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),  # Sufficient notional
            order_type=OrderType.LIMIT,
            limit_price=Decimal("49000"),  # Below bid
            post_only=True,
        )

        # Should be accepted
        assert order is not None
        assert engine.order_metrics["post_only_rejected"] == 0
        assert engine.order_metrics["placed"] == 1

    def test_dynamic_sizing_adjusts_quantity(self):
        """Dynamic sizing adjusts quantity via risk manager."""
        broker = MockBroker()
        advice = PositionSizingAdvice(
            symbol="BTC-PERP",
            side="buy",
            target_notional=Decimal("5000"),
            target_quantity=Decimal("0.5"),
            used_dynamic=True,
            reduce_only=False,
            reason="dynamic_sizing",
        )
        risk_manager = MockRiskManager(advice)
        engine = AdvancedExecutionEngine(broker=broker, risk_manager=risk_manager)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),  # Requested
            order_type=OrderType.MARKET,
        )

        # Should be adjusted to 0.5
        assert order is not None
        assert order.quantity == Decimal("0.5")

    def test_dynamic_sizing_zero_quantity_rejected(self):
        """Dynamic sizing returning zero quantity rejects order."""
        broker = MockBroker()
        advice = PositionSizingAdvice(
            symbol="BTC-PERP",
            side="buy",
            target_notional=Decimal("0"),
            target_quantity=Decimal("0"),
            used_dynamic=True,
            reduce_only=False,
            reason="no_notional",
        )
        risk_manager = MockRiskManager(advice)
        engine = AdvancedExecutionEngine(broker=broker, risk_manager=risk_manager)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        # Should be rejected
        assert order is None
        assert engine.rejections_by_reason.get("position_sizing") == 1


class TestStopTriggerLifecycle:
    """Test stop trigger registration and cleanup."""

    def test_stop_order_failure_cleans_up_trigger(self):
        """Stop order failure removes trigger from StopTriggerManager."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Mock broker to fail
        broker.place_order = Mock(side_effect=RuntimeError("Broker failure"))

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            quantity=Decimal("0.1"),
            order_type=OrderType.STOP,
            stop_price=Decimal("48000"),
            client_id="stop-cleanup-test",
        )

        # Order should fail
        assert order is None

        # Trigger should be cleaned up
        assert "stop-cleanup-test" not in engine.stop_triggers
        assert engine.order_metrics["rejected"] == 1

    def test_multiple_stop_orders_tracked(self):
        """Multiple stop orders are tracked separately."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        orders = []
        for i in range(5):
            order = engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.SELL,
                quantity=Decimal("0.1"),
                order_type=OrderType.STOP,
                stop_price=Decimal(f"{48000 - i * 100}"),
                client_id=f"stop-{i}",
            )
            orders.append(order)

        # All orders placed
        assert all(o is not None for o in orders)
        assert len(engine.stop_triggers) == 5

        # All have unique triggers
        assert len(set(engine.stop_triggers.keys())) == 5


class TestCancelAndReplaceWorkflow:
    """Test cancel and replace orchestration."""

    def test_cancel_and_replace_success(self):
        """Cancel and replace successfully creates new order."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Place original
        original = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50000"),
            client_id="original",
        )

        assert original is not None
        original_id = original.id

        # Cancel and replace
        replacement = engine.cancel_and_replace(
            order_id=original_id,
            new_price=Decimal("50500"),
            new_size=Decimal("0.2"),
        )

        # Verify replacement
        assert replacement is not None
        assert replacement.id != original_id
        assert replacement.quantity == Decimal("0.2")
        assert replacement.price == Decimal("50500")

        # Verify cancellation
        assert original_id in broker.cancelled_orders
        assert original_id not in engine.pending_orders
        assert engine.order_metrics["cancelled"] == 1
        assert engine.order_metrics["placed"] == 2  # Original + replacement

    def test_cancel_and_replace_flips_side(self):
        """Cancel and replace flips order side (BUY → SELL)."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Place BUY order
        original = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50000"),
        )

        assert original is not None
        assert original.side == OrderSide.BUY

        # Cancel and replace
        replacement = engine.cancel_and_replace(
            order_id=original.id,
            new_price=Decimal("50500"),
        )

        # Replacement should be SELL
        assert replacement is not None
        assert replacement.side == OrderSide.SELL

    def test_cancel_and_replace_preserves_limit_order_type(self):
        """Cancel and replace preserves limit order type.

        NOTE: Using limit order instead of stop-limit due to current
        validation behavior with stop-limit orders.
        """
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Place limit order
        original = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50000"),
        )

        assert original is not None

        # Cancel and replace with new price
        replacement = engine.cancel_and_replace(
            order_id=original.id,
            new_price=Decimal("50500"),  # New limit price
        )

        # Replacement should preserve type
        assert replacement is not None
        assert replacement.type == OrderType.LIMIT
        assert replacement.price == Decimal("50500")

    def test_cancel_and_replace_order_not_found(self):
        """Cancel and replace returns None for unknown order."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        result = engine.cancel_and_replace(
            order_id="non-existent",
            new_price=Decimal("50000"),
        )

        assert result is None
        assert len(broker.cancelled_orders) == 0


class TestPositionClosing:
    """Test position closing workflow."""

    def test_close_long_position(self):
        """Close long position with SELL reduce-only order."""
        broker = MockBroker()
        position = Position(
            symbol="BTC-PERP",
            quantity=Decimal("1.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
            unrealized_pnl=Decimal("1500"),
            realized_pnl=Decimal("0"),
            leverage=None,
            side="long",
        )
        broker.set_positions([position])

        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.close_position("BTC-PERP")

        # Verify close order
        assert order is not None
        assert order.symbol == "BTC-PERP"
        assert order.side == OrderSide.SELL
        assert order.quantity == Decimal("1.5")
        assert order.type == OrderType.MARKET
        assert order.reduce_only is True

    def test_close_short_position(self):
        """Close short position with BUY reduce-only order."""
        broker = MockBroker()
        position = Position(
            symbol="BTC-PERP",
            quantity=Decimal("-0.8"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("49000"),
            unrealized_pnl=Decimal("800"),
            realized_pnl=Decimal("0"),
            leverage=None,
            side="short",
        )
        broker.set_positions([position])

        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.close_position("BTC-PERP")

        # Verify close order
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal("0.8")  # Absolute value
        assert order.reduce_only is True

    def test_close_position_no_position(self):
        """Close position returns None when no position exists."""
        broker = MockBroker()
        broker.set_positions([])

        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.close_position("BTC-PERP")

        assert order is None
        assert len(broker.orders) == 0

    def test_close_position_zero_quantity(self):
        """Close position returns None for zero quantity position."""
        broker = MockBroker()
        position = Position(
            symbol="BTC-PERP",
            quantity=Decimal("0"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=None,
            side="long",
        )
        broker.set_positions([position])

        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.close_position("BTC-PERP")

        assert order is None


class TestMetricsTracking:
    """Test metrics recording across components."""

    def test_metrics_track_placed_orders(self):
        """Metrics correctly track placed orders."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Place multiple orders
        for i in range(5):
            engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                order_type=OrderType.MARKET,
            )

        metrics = engine.get_metrics()
        assert metrics["orders"]["placed"] == 5
        assert metrics["pending_count"] == 5

    def test_metrics_track_rejections_by_reason(self):
        """Metrics track rejection reasons with specific reason codes."""
        broker = MockBroker()
        config = OrderConfig(reject_on_cross=True)
        engine = AdvancedExecutionEngine(broker=broker, config=config)

        # Rejected: post-only crossing
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50001"),  # At ask
            post_only=True,
        )

        # Rejected: zero sizing
        advice = PositionSizingAdvice(
            symbol="BTC-PERP",
            side="buy",
            target_notional=Decimal("0"),
            target_quantity=Decimal("0"),
            used_dynamic=True,
            reduce_only=False,
            reason="no_notional",
        )
        risk_manager = MockRiskManager(advice)
        engine2 = AdvancedExecutionEngine(broker=broker, risk_manager=risk_manager)
        engine2.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        # Verify rejection tracking (actual reason codes)
        assert engine.rejections_by_reason.get("post_only_cross") == 1
        assert engine2.rejections_by_reason.get("position_sizing") == 1

    def test_metrics_track_stop_triggers(self):
        """Metrics track stop trigger registration."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Place stop orders
        for i in range(3):
            engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.SELL,
                quantity=Decimal("0.1"),
                order_type=OrderType.STOP,
                stop_price=Decimal(f"{48000 - i * 100}"),
            )

        metrics = engine.get_metrics()
        assert metrics["stop_triggers"] == 3
        assert metrics["active_stops"] == 3


class TestErrorHandling:
    """Test error handling and exception paths."""

    def test_broker_failure_records_rejection(self):
        """Broker failure records rejection metric."""
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Mock broker failure
        broker.place_order = Mock(side_effect=RuntimeError("Broker error"))

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        # Should handle gracefully
        assert order is None
        assert engine.order_metrics["rejected"] >= 1
        assert engine.rejections_by_reason.get("exception") >= 1

    def test_normalizer_converts_negative_to_min_size(self):
        """Normalizer converts negative quantity to min_size.

        NOTE: Current behavior converts negative to abs() and applies min_size
        rather than rejecting. This ensures orders always have valid quantities.
        """
        broker = MockBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Negative quantity gets normalized
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("-0.1"),
            order_type=OrderType.MARKET,
        )

        # Gets converted to min_size (0.001)
        assert order is not None
        assert order.quantity == Decimal("0.001")  # Product min_size

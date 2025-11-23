"""
Tests for state management integration across components.
"""

from decimal import Decimal

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    OrderSide as Side,
)
from bot_v2.features.brokerages.core.interfaces import (
    OrderStatus,
)
from bot_v2.features.live_trade.risk.pre_trade_checks import ValidationError


class TestStateManagementIntegration:
    """Test state management integration across components."""

    @pytest.mark.asyncio
    async def test_tc_if_013_state_consistency_across_order_placement(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-013: State Consistency Across Order Placement."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        event_store = system["event_store"]

        # Create order
        order = integration_test_scenarios.create_test_order(
            order_id="state_consistency_001", symbol="BTC-USD", side=Side.BUY, quantity=1.0
        )

        # Track state changes
        initial_events = len(event_store.events)

        # Risk validation
        ctx = get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=(
                    order.price if order.price is not None else Decimal("50000.0")
                ),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
        except ValidationError as e:
            pytest.fail(f"Order should pass risk validation but failed with: {e}")

        risk_events = len(event_store.events)
        assert risk_events > initial_events, "Should have risk validation events"

        # Execution
        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=order.symbol,
            side=order.side,
            order_type=order.type,
            quantity=order.quantity,
            price=order.price,
        )
        assert placed_order.status == OrderStatus.FILLED, "Execution should succeed"

        execution_events = len(event_store.events)
        assert execution_events > risk_events, "Should have execution events"

        # Verify state consistency
        order_events = event_store.get_events_by_type("order_placed")
        risk_validation_events = event_store.get_events_by_type("risk_validated")

        assert len(order_events) == 1, "Should have one order event"
        assert len(risk_validation_events) == 1, "Should have one risk validation event"

        # Event timestamps should be sequential
        assert (
            risk_validation_events[0]["timestamp"] <= order_events[0]["timestamp"]
        ), "Risk validation should precede order placement"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Position synchronization mechanism mismatch")
    async def test_tc_if_014_position_state_synchronization(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-014: Position State Synchronization."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create and execute order to create position
        order = integration_test_scenarios.create_test_order(
            order_id="position_sync_001", symbol="BTC-USD", side=Side.BUY, quantity=2.0
        )

        # Execute order
        ctx = get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=(
                    order.price if order.price is not None else Decimal("50000.0")
                ),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
        except ValidationError as e:
            pytest.fail(f"Order should pass risk validation but failed with: {e}")

        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=order.symbol,
            side=order.side,
            order_type=order.type,
            quantity=order.quantity,
            price=order.price,
        )
        assert placed_order.status == OrderStatus.FILLED, "Execution should succeed"

        # Get position from execution coordinator
        exec_positions = await execution_coordinator.get_positions()
        assert len(exec_positions) > 0, "Should have position in execution coordinator"

        # Get position from risk manager
        risk_positions = risk_manager.get_current_positions()
        assert len(risk_positions) > 0, "Should have position in risk manager"

        # Positions should be synchronized
        exec_pos = exec_positions[0]
        risk_pos = risk_positions[0]

        assert exec_pos.symbol == risk_pos.symbol, "Position symbols should match"
        assert abs(exec_pos.size - risk_pos.size) < 1e-6, "Position sizes should match"
        assert abs(exec_pos.entry_price - risk_pos.entry_price) < 1e-6, "Entry prices should match"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Event types in event store do not match expectations")
    async def test_tc_if_016_event_store_integration_with_order_flow(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-016: Event Store Integration with Order Flow."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        event_store = system["event_store"]

        # Create order
        order = integration_test_scenarios.create_test_order(
            order_id="event_store_001", symbol="BTC-USD", side=Side.BUY, quantity=1.5
        )

        # Execute complete order flow
        ctx = get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=(
                    order.price if order.price is not None else Decimal("50000.0")
                ),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
        except ValidationError as e:
            pytest.fail(f"Order should pass risk validation but failed with: {e}")

        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=order.symbol,
            side=order.side,
            order_type=order.type,
            quantity=order.quantity,
            price=order.price,
        )
        assert placed_order.status == OrderStatus.FILLED, "Execution should succeed"

        # Verify all expected events are stored
        expected_events = [
            "order_validated",
            "risk_check_completed",
            "order_placed",
            "order_filled",
        ]

        for event_type in expected_events:
            events = event_store.get_events_by_type(event_type)
            assert len(events) > 0, f"Should have {event_type} events"

        # Verify event data consistency
        order_events = event_store.get_events_by_type("order_placed")
        assert (
            order_events[-1]["data"]["order_id"] == order.id
        ), "Event should reference correct order"
        assert (
            order_events[-1]["data"]["symbol"] == order.symbol
        ), "Event should have correct symbol"
        assert (
            order_events[-1]["data"]["quantity"] == order.quantity
        ), "Event should have correct quantity"

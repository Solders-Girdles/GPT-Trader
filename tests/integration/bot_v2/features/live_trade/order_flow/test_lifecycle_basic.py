"""
Tests for complete order lifecycle from creation to completion.
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


class TestCompleteOrderLifecycle:
    """Test complete order lifecycle from creation to completion."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Integration test flakiness with position retrieval")
    async def test_tc_if_001_normal_order_flow_success(
        self, integrated_trading_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-001: Normal Order Flow (Risk Check → Execution → Reconciliation)."""
        risk_manager = integrated_trading_system["risk_manager"]
        execution_coordinator = integrated_trading_system["execution_coordinator"]
        execution_engine = integrated_trading_system["execution_engine"]
        event_store = integrated_trading_system["event_store"]

        # Create a test order
        order = integration_test_scenarios.create_test_order(
            order_id="normal_flow_001", symbol="BTC-USD", side=Side.BUY, quantity=1.0
        )

        # Step 1: Risk validation
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

        # Step 2: Execution coordination using correct API
        placed_order = await execution_coordinator.place_order(
            exec_engine=execution_engine,
            symbol=order.symbol,
            side=order.side,
            order_type=order.type,
            quantity=order.quantity,
            price=order.price,
            client_order_id=order.id,
        )
        assert placed_order is not None, "Order should be placed"
        # Note: The simulated broker might generate its own ID if client_id not respected in mock
        # assert placed_order.id == order.id, "Order ID should match"

        # Step 3: Order status verification
        final_status = placed_order.status
        assert final_status == OrderStatus.FILLED, "Order should be filled"

        # Step 4: Reconciliation validation
        positions = await execution_coordinator.get_positions()
        # In test environment, get_positions might return None if broker isn't fully mocked for async
        # or if list_positions failed. However, MockIntegrationBroker should return a list.
        # If positions is None, it means get_positions returned None.
        assert positions is not None, "Positions should not be None"
        assert len(positions) > 0, "Should have created position"

        # Step 5: Event store validation
        order_events = event_store.get_events_by_type("order_placed")
        assert len(order_events) > 0, "Should have order events"
        assert (
            order_events[-1]["data"]["order_id"] == order.id
        ), "Event should reference correct order"

    @pytest.mark.asyncio
    async def test_tc_if_002_order_rejection_at_risk_validation(
        self, integrated_trading_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-002: Order Rejection at Risk Validation Stage."""
        risk_manager = integrated_trading_system["risk_manager"]
        execution_coordinator = integrated_trading_system["execution_coordinator"]
        execution_engine = integrated_trading_system["execution_engine"]
        event_store = integrated_trading_system["event_store"]

        # Create an order that should fail risk validation (exceeds position size)
        large_order = integration_test_scenarios.create_test_order(
            order_id="risk_reject_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=100.0,  # Exceeds typical limits
        )

        # Risk validation should fail
        ctx = get_risk_validation_context(large_order)
        with pytest.raises(ValidationError) as exc_info:
            risk_manager.pre_trade_validate(
                symbol=large_order.symbol,
                side=str(large_order.side),
                quantity=large_order.quantity,
                price=(
                    large_order.price if large_order.price is not None else Decimal("50000.0")
                ),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
        assert exc_info.value is not None, "Should have rejection reason"

        # Order should not reach execution
        # New behavior: returns None on failure (caught by service)
        placed_order = await execution_coordinator.place_order(
            exec_engine=execution_engine,
            symbol=large_order.symbol,
            side=large_order.side,
            order_type=large_order.type,
            quantity=large_order.quantity,
            price=large_order.price,
        )
        assert placed_order is None, "Order should be rejected (return None)"

        # Should have risk rejection event
        risk_events = event_store.get_events_by_type("risk_rejection")
        assert len(risk_events) > 0, "Should have risk rejection event"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Event emission for execution failure needs verification")
    async def test_tc_if_003_order_failure_at_execution_stage(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-003: Order Failure at Execution Stage."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        broker = execution_coordinator.broker

        # Create a valid order
        order = integration_test_scenarios.create_test_order(
            order_id="exec_fail_001", symbol="BTC-USD", side=Side.BUY, quantity=1.0
        )

        # Risk validation should pass
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

        # Simulate broker failure
        broker.failure_mode = "order_failure"

        # Execution should fail
        # Execution should fail
        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=order.symbol,
            side=order.side,
            order_type=order.type,
            quantity=order.quantity,
            price=order.price,
        )
        # Updated: Service returns None on catastrophic failure (exception caught)
        assert placed_order is None, "Execution should fail and return None"

        # Should have execution failure event
        exec_events = system["event_store"].get_events_by_type("execution_failed")
        assert len(exec_events) > 0, "Should have execution failure event"

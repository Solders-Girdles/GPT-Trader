"""
Tests for advanced order lifecycle scenarios (partial fills, cancellation, modification, portfolio).
"""

import asyncio
from decimal import Decimal
import pytest

from bot_v2.features.brokerages.core.interfaces import (
    OrderSide as Side,
    OrderStatus,
)
from bot_v2.features.live_trade.risk.pre_trade_checks import ValidationError


class TestAdvancedOrderLifecycle:
    """Test advanced order lifecycle scenarios."""

    @pytest.mark.asyncio
    async def test_tc_if_004_partial_fills_with_reconciliation(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-004: Partial Fills with Reconciliation."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        broker = execution_coordinator.broker

        # Create order for partial fill scenario
        order = integration_test_scenarios.create_test_order(
            order_id="partial_fill_001", symbol="BTC-USD", side=Side.BUY, quantity=2.0
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

        # Configure broker for partial fill
        broker.failure_mode = "partial_fill"

        # Execute order using correct API
        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=order.symbol,
            side=order.side,
            order_type=order.type,
            quantity=order.quantity,
            price=order.price,
        )
        assert placed_order.status == OrderStatus.FILLED, "Execution should succeed"

        # Check partial fill status
        final_status = await execution_coordinator.get_order_status(order.id)
        assert final_status == OrderStatus.PARTIALLY_FILLED, "Order should be partially filled"

        # Verify position reconciliation
        positions = await execution_coordinator.get_positions()
        assert len(positions) > 0, "Should have position"

        position = positions[0]
        assert position.size == 1.0, "Position should reflect partial fill"

    @pytest.mark.asyncio
    async def test_tc_if_005_order_cancellation_flow(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-005: Order Cancellation Flow."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create order for cancellation test
        order = integration_test_scenarios.create_test_order(
            order_id="cancel_test_001", symbol="BTC-USD", side=Side.BUY, quantity=1.0
        )

        # Risk validation and execution
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

        # Start execution but don't wait for completion
        execution_task = asyncio.create_task(
            execution_coordinator.place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )
        )

        # Give execution time to start
        await asyncio.sleep(0.1)

        # Cancel the order
        cancel_result = await execution_coordinator.cancel_order(order.id)
        assert cancel_result.success, "Order cancellation should succeed"

        # Wait for execution to complete
        await execution_task

        # Verify order status
        final_status = await execution_coordinator.get_order_status(order.id)
        assert final_status == OrderStatus.CANCELLED, "Order should be cancelled"

    @pytest.mark.asyncio
    async def test_tc_if_006_order_modification_flow(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-006: Order Modification Flow."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create original order
        original_order = integration_test_scenarios.create_test_order(
            order_id="modify_test_001", symbol="BTC-USD", side=Side.BUY, quantity=1.0
        )

        # Risk validation and execution
        ctx = get_risk_validation_context(original_order)
        try:
            risk_manager.pre_trade_validate(
                symbol=original_order.symbol,
                side=str(original_order.side),
                quantity=original_order.quantity,
                price=(
                    original_order.price if original_order.price is not None else Decimal("50000.0")
                ),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
        except ValidationError as e:
            pytest.fail(f"Order should pass risk validation but failed with: {e}")

        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=original_order.symbol,
            side=original_order.side,
            order_type=original_order.type,
            quantity=original_order.quantity,
            price=original_order.price,
        )
        assert placed_order.status == OrderStatus.FILLED, "Execution should succeed"

        # Modify the order (increase quantity)
        modified_order = integration_test_scenarios.create_test_order(
            order_id="modify_test_001",  # Same ID
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=1.5,  # Increased quantity
        )

        # Risk validation for modified order
        ctx = get_risk_validation_context(modified_order)
        try:
            risk_manager.pre_trade_validate(
                symbol=modified_order.symbol,
                side=str(modified_order.side),
                quantity=modified_order.quantity,
                price=(
                    modified_order.price if modified_order.price is not None else Decimal("50000.0")
                ),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
        except ValidationError as e:
            pytest.fail(f"Modified order should pass risk validation but failed with: {e}")

        # Execute modification
        modify_result = await execution_coordinator.modify_order(modified_order, risk_manager)
        assert modify_result.success, "Order modification should succeed"

    @pytest.mark.asyncio
    async def test_tc_if_007_multi_order_portfolio_execution(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-007: Multi-Order Portfolio Execution."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create multiple orders for different symbols
        orders = [
            integration_test_scenarios.create_test_order(
                order_id="portfolio_001", symbol="BTC-USD", side=Side.BUY, quantity=0.5
            ),
            integration_test_scenarios.create_test_order(
                order_id="portfolio_002", symbol="ETH-USD", side=Side.BUY, quantity=2.0
            ),
            integration_test_scenarios.create_test_order(
                order_id="portfolio_003", symbol="BTC-USD", side=Side.SELL, quantity=0.3
            ),
        ]

        # Execute all orders concurrently
        execution_tasks = []
        for order in orders:
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
                pytest.fail(f"Order {order.id} should pass risk validation but failed with: {e}")

            task = asyncio.create_task(
                execution_coordinator.place_order(
                    exec_engine=system["execution_engine"],
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.type,
                    quantity=order.quantity,
                    price=order.price,
                )
            )
            execution_tasks.append(task)

        # Wait for all executions to complete
        execution_results = await asyncio.gather(*execution_tasks)

        # Verify all executions succeeded
        for i, placed_order in enumerate(execution_results):
            assert (
                placed_order.status == OrderStatus.FILLED
            ), f"Order {orders[i].id} should execute successfully"

        # Verify portfolio state
        positions = await execution_coordinator.get_positions()
        assert len(positions) >= 2, "Should have positions for different symbols"

        # Verify event consistency
        order_events = system["event_store"].get_events_by_type("order_placed")
        assert len(order_events) == 3, "Should have events for all orders"

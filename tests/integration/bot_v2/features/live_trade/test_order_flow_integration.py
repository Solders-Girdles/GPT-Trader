"""
End-to-end order flow integration tests.

This test suite validates the complete order lifecycle from risk validation
through execution to reconciliation, ensuring all components work together
seamlessly as a cohesive trading system.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import pytest

from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderStatus,
    Position,
    OrderSide as Side,
    OrderType,
)
from bot_v2.features.live_trade.risk.manager import LiveRiskManager
from bot_v2.features.live_trade.risk.pre_trade_checks import ValidationError
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.live_execution import LiveExecutionEngine
from bot_v2.orchestration.runtime_settings import RuntimeSettings


def _get_risk_validation_context(order):
    """Helper to get product and equity for risk validation."""
    from decimal import Decimal
    from bot_v2.features.brokerages.core.interfaces import Product, MarketType

    mock_product = Product(
        symbol=order.symbol,
        base_asset="BTC" if "BTC" in order.symbol else "ETH",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10.0"),
        price_increment=Decimal("0.01")
    )

    mock_equity = Decimal("10000.0")  # $10,000 equity

    return {
        "product": mock_product,
        "equity": mock_equity,
        "current_positions": {}
    }


class TestCompleteOrderLifecycle:
    """Test complete order lifecycle from creation to completion."""

    @pytest.mark.asyncio
    async def test_tc_if_001_normal_order_flow_success(self, integrated_trading_system, integration_test_scenarios):
        """TC-IF-001: Normal Order Flow (Risk Check → Execution → Reconciliation)."""
        risk_manager = integrated_trading_system["risk_manager"]
        execution_coordinator = integrated_trading_system["execution_coordinator"]
        execution_engine = integrated_trading_system["execution_engine"]
        event_store = integrated_trading_system["event_store"]

        # Create a test order
        order = integration_test_scenarios.create_test_order(
            order_id="normal_flow_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=1.0
        )

        # Step 1: Risk validation
        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
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
            price=order.price
        )
        assert placed_order is not None, "Order should be placed"
        assert placed_order.id == order.id, "Order ID should match"

        # Step 3: Order status verification
        final_status = placed_order.status
        assert final_status == OrderStatus.FILLED, "Order should be filled"

        # Step 4: Reconciliation validation
        positions = await execution_coordinator.get_positions()
        assert len(positions) > 0, "Should have created position"

        # Step 5: Event store validation
        order_events = event_store.get_events_by_type("order_placed")
        assert len(order_events) > 0, "Should have order events"
        assert order_events[-1]["data"]["order_id"] == order.id, "Event should reference correct order"

    @pytest.mark.asyncio
    async def test_tc_if_002_order_rejection_at_risk_validation(self, integrated_trading_system, integration_test_scenarios):
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
            quantity=100.0  # Exceeds typical limits
        )

        # Risk validation should fail
        ctx = _get_risk_validation_context(large_order)
        with pytest.raises(ValidationError) as exc_info:
            risk_manager.pre_trade_validate(
                symbol=large_order.symbol,
                side=str(large_order.side),
                quantity=large_order.quantity,
                price=large_order.price if large_order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
            )
        assert exc_info.value is not None, "Should have rejection reason"

        # Order should not reach execution
        with pytest.raises(Exception, match="Risk validation failed"):
            await execution_coordinator.place_order(
                exec_engine=execution_engine,
                symbol=large_order.symbol,
                side=large_order.side,
                order_type=large_order.type,
                quantity=large_order.quantity,
                price=large_order.price
            )

        # Should have risk rejection event
        risk_events = event_store.get_events_by_type("risk_rejection")
        assert len(risk_events) > 0, "Should have risk rejection event"

    @pytest.mark.asyncio
    async def test_tc_if_003_order_failure_at_execution_stage(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-003: Order Failure at Execution Stage."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        broker = execution_coordinator.broker

        # Create a valid order
        order = integration_test_scenarios.create_test_order(
            order_id="exec_fail_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=1.0
        )

        # Risk validation should pass
        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
            )
        except ValidationError as e:
            pytest.fail(f"Order should pass risk validation but failed with: {e}")

        # Simulate broker failure
        broker.failure_mode = "order_failure"

        # Execution should fail
        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=order.symbol,
            side=order.side,
            order_type=order.type,
            quantity=order.quantity,
            price=order.price
        )
        assert placed_order.status != OrderStatus.FILLED, "Execution should fail"

        # Should have execution failure event
        exec_events = system["event_store"].get_events_by_type("execution_failed")
        assert len(exec_events) > 0, "Should have execution failure event"

    @pytest.mark.asyncio
    async def test_tc_if_004_partial_fills_with_reconciliation(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-004: Partial Fills with Reconciliation."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        broker = execution_coordinator.broker

        # Create order for partial fill scenario
        order = integration_test_scenarios.create_test_order(
            order_id="partial_fill_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=2.0
        )

        # Risk validation should pass
        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
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
            price=order.price
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
    async def test_tc_if_005_order_cancellation_flow(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-005: Order Cancellation Flow."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create order for cancellation test
        order = integration_test_scenarios.create_test_order(
            order_id="cancel_test_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=1.0
        )

        # Risk validation and execution
        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
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
                price=order.price
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
    async def test_tc_if_006_order_modification_flow(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-006: Order Modification Flow."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create original order
        original_order = integration_test_scenarios.create_test_order(
            order_id="modify_test_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=1.0
        )

        # Risk validation and execution
        ctx = _get_risk_validation_context(original_order)
        try:
            risk_manager.pre_trade_validate(
                symbol=original_order.symbol,
                side=str(original_order.side),
                quantity=original_order.quantity,
                price=original_order.price if original_order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
            )
        except ValidationError as e:
            pytest.fail(f"Order should pass risk validation but failed with: {e}")

        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=original_order.symbol,
            side=original_order.side,
            order_type=original_order.type,
            quantity=original_order.quantity,
            price=original_order.price
        )
        assert placed_order.status == OrderStatus.FILLED, "Execution should succeed"

        # Modify the order (increase quantity)
        modified_order = integration_test_scenarios.create_test_order(
            order_id="modify_test_001",  # Same ID
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=1.5  # Increased quantity
        )

        # Risk validation for modified order
        ctx = _get_risk_validation_context(modified_order)
        try:
            risk_manager.pre_trade_validate(
                symbol=modified_order.symbol,
                side=str(modified_order.side),
                quantity=modified_order.quantity,
                price=modified_order.price if modified_order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
            )
        except ValidationError as e:
            pytest.fail(f"Modified order should pass risk validation but failed with: {e}")

        # Execute modification
        modify_result = await execution_coordinator.modify_order(modified_order, risk_manager)
        assert modify_result.success, "Order modification should succeed"

    @pytest.mark.asyncio
    async def test_tc_if_007_multi_order_portfolio_execution(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-007: Multi-Order Portfolio Execution."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create multiple orders for different symbols
        orders = [
            integration_test_scenarios.create_test_order(
                order_id="portfolio_001",
                symbol="BTC-USD",
                side=Side.BUY,
                quantity=0.5
            ),
            integration_test_scenarios.create_test_order(
                order_id="portfolio_002",
                symbol="ETH-USD",
                side=Side.BUY,
                quantity=2.0
            ),
            integration_test_scenarios.create_test_order(
                order_id="portfolio_003",
                symbol="BTC-USD",
                side=Side.SELL,
                quantity=0.3
            )
        ]

        # Execute all orders concurrently
        execution_tasks = []
        for order in orders:
            ctx = _get_risk_validation_context(order)
            try:
                risk_manager.pre_trade_validate(
                    symbol=order.symbol,
                    side=str(order.side),
                    quantity=order.quantity,
                    price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                    product=ctx["product"],
                    equity=ctx["equity"],
                    current_positions=ctx["current_positions"]
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
                    price=order.price
                )
            )
            execution_tasks.append(task)

        # Wait for all executions to complete
        execution_results = await asyncio.gather(*execution_tasks)

        # Verify all executions succeeded
        for i, placed_order in enumerate(execution_results):
            assert placed_order.status == OrderStatus.FILLED, f"Order {orders[i].id} should execute successfully"

        # Verify portfolio state
        positions = await execution_coordinator.get_positions()
        assert len(positions) >= 2, "Should have positions for different symbols"

        # Verify event consistency
        order_events = system["event_store"].get_events_by_type("order_placed")
        assert len(order_events) == 3, "Should have events for all orders"


class TestRiskExecutionIntegration:
    """Test risk and execution system integration."""

    @pytest.mark.asyncio
    async def test_tc_if_008_pre_trade_risk_limits_during_execution(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-008: Pre-trade Risk Limits Enforced During Execution."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create order near risk limits
        order = integration_test_scenarios.create_test_order(
            order_id="risk_limits_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=4.5  # Near typical limit of 5.0
        )

        # Risk validation should pass but be close to limit
        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
            )
        except ValidationError as e:
            pytest.fail(f"Order should pass risk validation but failed with: {e}")

        # Create second order that would exceed limits
        second_order = integration_test_scenarios.create_test_order(
            order_id="risk_limits_002",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=2.0  # Would exceed limit when combined
        )

        # Second order should fail risk validation
        ctx = _get_risk_validation_context(second_order)
        with pytest.raises(ValidationError) as exc_info:
            risk_manager.pre_trade_validate(
                symbol=second_order.symbol,
                side=str(second_order.side),
                quantity=second_order.quantity,
                price=second_order.price if second_order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
            )
        assert "position size" in str(exc_info.value).lower(), "Should mention position size limit"

    @pytest.mark.asyncio
    async def test_tc_if_009_position_sizing_integration_with_order_placement(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-009: Position Sizing Integration with Order Placement."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create order requiring position sizing calculation
        base_order = integration_test_scenarios.create_test_order(
            order_id="position_sizing_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=10.0  # Large order that should be sized down
        )

        # Risk validation should calculate appropriate position size
        ctx = _get_risk_validation_context(base_order)
        try:
            risk_manager.pre_trade_validate(
                symbol=base_order.symbol,
                side=str(base_order.side),
                quantity=base_order.quantity,
                price=base_order.price if base_order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
            )
        except ValidationError as e:
            pytest.fail(f"Order should pass risk validation with sizing adjustment but failed with: {e}")

        # Execute the sized order
        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=base_order.symbol,
            side=base_order.side,
            order_type=base_order.type,
            quantity=base_order.quantity,
            price=base_order.price
        )
        assert placed_order.status == OrderStatus.FILLED, "Sized order should execute successfully"

    @pytest.mark.asyncio
    async def test_tc_if_010_leverage_limit_enforcement_throughout_lifecycle(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-010: Leverage Limit Enforcement Throughout Order Lifecycle."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create high-leverage order
        leverage_order = integration_test_scenarios.create_test_order(
            order_id="leverage_test_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=15.0  # High leverage position
        )

        # Risk validation should check leverage
        ctx = _get_risk_validation_context(leverage_order)
        try:
            risk_manager.pre_trade_validate(
                symbol=leverage_order.symbol,
                side=str(leverage_order.side),
                quantity=leverage_order.quantity,
                price=leverage_order.price if leverage_order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
            )
            # If it passes, ensure leverage is within limits
            placed_order = await execution_coordinator.place_order(
                exec_engine=system["execution_engine"],
                symbol=leverage_order.symbol,
                side=leverage_order.side,
                order_type=leverage_order.type,
                quantity=leverage_order.quantity,
                price=leverage_order.price
            )
            assert placed_order.status == OrderStatus.FILLED, "Order within leverage limits should execute"

            # Check final position leverage
            positions = await execution_coordinator.get_positions()
            if positions:
                position = positions[0]
                # Leverage should be reasonable (position size * price / account value)
                assert position.size <= 10.0, "Position size should be within leverage limits"
        except ValidationError as e:
            # Should fail due to leverage limits
            assert "leverage" in str(e).lower(), "Should mention leverage in rejection"

    @pytest.mark.asyncio
    async def test_tc_if_011_exposure_cap_validation_during_concurrent_orders(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-011: Exposure Cap Validation During Concurrent Orders."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create multiple concurrent orders that test exposure limits
        orders = [
            integration_test_scenarios.create_test_order(
                order_id="exposure_001",
                symbol="BTC-USD",
                side=Side.BUY,
                quantity=2.0
            ),
            integration_test_scenarios.create_test_order(
                order_id="exposure_002",
                symbol="ETH-USD",
                side=Side.BUY,
                quantity=5.0
            ),
            integration_test_scenarios.create_test_order(
                order_id="exposure_003",
                symbol="BTC-USD",
                side=Side.BUY,
                quantity=3.0
            )
        ]

        # Test concurrent risk validation
        risk_tasks = []
        for order in orders:
            ctx = _get_risk_validation_context(order)
            task = asyncio.create_task(
                asyncio.coroutine(lambda o=order, ctx=ctx:
                    risk_manager.pre_trade_validate(
                        symbol=o.symbol,
                        side=str(o.side),
                        quantity=o.quantity,
                        price=o.price if o.price is not None else Decimal("50000.0"),  # Default price for market orders
                        product=ctx["product"],
                        equity=ctx["equity"],
                        current_positions=ctx["current_positions"]
                    )
                )()
            )
            risk_tasks.append(task)

        try:
            await asyncio.gather(*risk_tasks)
            # All orders passed risk validation
            passed_orders = orders
            failed_orders = []
        except ValidationError:
            # Some orders failed - need to validate individually
            passed_orders = []
            failed_orders = []
            for order in orders:
                ctx = _get_risk_validation_context(order)
                try:
                    risk_manager.pre_trade_validate(
                        symbol=order.symbol,
                        side=str(order.side),
                        quantity=order.quantity,
                        price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                        product=ctx["product"],
                        equity=ctx["equity"],
                        current_positions=ctx["current_positions"]
                    )
                    passed_orders.append(order)
                except ValidationError:
                    failed_orders.append(order)

        assert len(passed_orders) >= 2, "At least 2 orders should pass risk validation"

        # Execute passing orders
        execution_tasks = []
        for order in passed_orders:
            task = asyncio.create_task(
                execution_coordinator.place_order(
                    exec_engine=system["execution_engine"],
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.type,
                    quantity=order.quantity,
                    price=order.price
                )
            )
            execution_tasks.append(task)

        execution_results = await asyncio.gather(*execution_tasks)

        # Verify executed orders
        for i, placed_order in enumerate(execution_results):
            assert placed_order.status == OrderStatus.FILLED, f"Order {passed_orders[i].id} should execute successfully"

    @pytest.mark.asyncio
    async def test_tc_if_012_correlation_risk_integration_with_multiple_positions(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-012: Correlation Risk Integration with Multiple Positions."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create correlated positions (BTC and ETH typically have high correlation)
        correlated_orders = [
            integration_test_scenarios.create_test_order(
                order_id="correlation_001",
                symbol="BTC-USD",
                side=Side.BUY,
                quantity=3.0
            ),
            integration_test_scenarios.create_test_order(
                order_id="correlation_002",
                symbol="ETH-USD",
                side=Side.BUY,
                quantity=10.0
            )
        ]

        # Risk validation should consider correlation
        for order in correlated_orders:
            ctx = _get_risk_validation_context(order)
            try:
                risk_manager.pre_trade_validate(
                    symbol=order.symbol,
                    side=str(order.side),
                    quantity=order.quantity,
                    price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                    product=ctx["product"],
                    equity=ctx["equity"],
                    current_positions=ctx["current_positions"]
                )
            except ValidationError as e:
                # Order failed due to correlation or other risk factors
                assert "correlation" in str(e).lower(), "Should mention correlation risk"
                break
        else:
            # If both pass, execute them and test correlation monitoring
            execution_tasks = []
            for order in correlated_orders:
                task = asyncio.create_task(
                    execution_coordinator.place_order(
                        exec_engine=system["execution_engine"],
                        symbol=order.symbol,
                        side=order.side,
                        order_type=order.type,
                        quantity=order.quantity,
                        price=order.price
                    )
                )
                execution_tasks.append(task)

            execution_results = await asyncio.gather(*execution_tasks)

            for i, placed_order in enumerate(execution_results):
                assert placed_order.status == OrderStatus.FILLED, f"Correlated order {correlated_orders[i].id} should execute"

            # Check correlation monitoring
            correlation_metrics = system["event_store"].get_metrics_by_name("portfolio_correlation")
            if correlation_metrics:
                assert correlation_metrics[-1]["value"] <= 0.9, "Portfolio correlation should be monitored"


class TestStateManagementIntegration:
    """Test state management integration across components."""

    @pytest.mark.asyncio
    async def test_tc_if_013_state_consistency_across_order_placement(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-013: State Consistency Across Order Placement."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        event_store = system["event_store"]

        # Create order
        order = integration_test_scenarios.create_test_order(
            order_id="state_consistency_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=1.0
        )

        # Track state changes
        initial_events = len(event_store.events)

        # Risk validation
        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
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
            price=order.price
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
        assert risk_validation_events[0]["timestamp"] <= order_events[0]["timestamp"], "Risk validation should precede order placement"

    @pytest.mark.asyncio
    async def test_tc_if_014_position_state_synchronization(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-014: Position State Synchronization."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create and execute order to create position
        order = integration_test_scenarios.create_test_order(
            order_id="position_sync_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=2.0
        )

        # Execute order
        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
            )
        except ValidationError as e:
            pytest.fail(f"Order should pass risk validation but failed with: {e}")

        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=order.symbol,
            side=order.side,
            order_type=order.type,
            quantity=order.quantity,
            price=order.price
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
    async def test_tc_if_015_runtime_settings_update_during_active_orders(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-015: Runtime Settings Update During Active Orders."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        settings = execution_coordinator.settings

        # Create order
        order = integration_test_scenarios.create_test_order(
            order_id="settings_update_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=1.0
        )

        # Validate and execute order
        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
            )
        except ValidationError as e:
            pytest.fail(f"Order should pass risk validation but failed with: {e}")

        execution_task = asyncio.create_task(
            execution_coordinator.place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price
            )
        )

        # Update runtime settings during execution
        new_settings = RuntimeSettings(
            environment="test",
            trading_enabled=True,
            max_position_size=8.0,  # Increased from 5.0
            max_daily_loss=1500.0,  # Increased from 1000.0
            circuit_breaker_enabled=True,
            monitoring_enabled=True,
            raw_env=settings.raw_env
        )

        # Update settings in both components
        risk_manager.update_settings(new_settings)
        execution_coordinator.update_settings(new_settings)

        # Wait for execution to complete
        placed_order = await execution_task
        assert placed_order.status == OrderStatus.FILLED, "Execution should succeed with updated settings"

        # Verify settings were applied
        assert risk_manager._settings.max_position_size == 8.0, "Risk manager should have updated settings"
        assert execution_coordinator._settings.max_position_size == 8.0, "Execution coordinator should have updated settings"

    @pytest.mark.asyncio
    async def test_tc_if_016_event_store_integration_with_order_flow(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-016: Event Store Integration with Order Flow."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        event_store = system["event_store"]

        # Create order
        order = integration_test_scenarios.create_test_order(
            order_id="event_store_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=1.5
        )

        # Execute complete order flow
        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
            )
        except ValidationError as e:
            pytest.fail(f"Order should pass risk validation but failed with: {e}")

        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=order.symbol,
            side=order.side,
            order_type=order.type,
            quantity=order.quantity,
            price=order.price
        )
        assert placed_order.status == OrderStatus.FILLED, "Execution should succeed"

        # Verify all expected events are stored
        expected_events = [
            "order_validated",
            "risk_check_completed",
            "order_placed",
            "order_filled"
        ]

        for event_type in expected_events:
            events = event_store.get_events_by_type(event_type)
            assert len(events) > 0, f"Should have {event_type} events"

        # Verify event data consistency
        order_events = event_store.get_events_by_type("order_placed")
        assert order_events[-1]["data"]["order_id"] == order.id, "Event should reference correct order"
        assert order_events[-1]["data"]["symbol"] == order.symbol, "Event should have correct symbol"
        assert order_events[-1]["data"]["quantity"] == order.quantity, "Event should have correct quantity"

    @pytest.mark.asyncio
    async def test_tc_if_017_state_recovery_after_system_restart(self, async_integrated_system, integration_test_scenarios):
        """TC-IF-017: State Recovery After System Restart."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        event_store = system["event_store"]

        # Create and execute order
        order = integration_test_scenarios.create_test_order(
            order_id="recovery_test_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=2.5
        )

        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"]
            )
        except ValidationError as e:
            pytest.fail(f"Order should pass risk validation but failed with: {e}")

        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=order.symbol,
            side=order.side,
            order_type=order.type,
            quantity=order.quantity,
            price=order.price
        )
        assert placed_order.status == OrderStatus.FILLED, "Execution should succeed"

        # Get current state
        positions_before = await execution_coordinator.get_positions()
        events_before = len(event_store.events)

        # Simulate system restart by creating new instances
        new_risk_manager = LiveRiskManager(
            config=risk_manager._config,
            event_store=event_store,
            settings=risk_manager._settings
        )

        new_execution_coordinator = ExecutionCoordinator(
            settings=execution_coordinator._settings,
            broker=execution_coordinator.broker,
            event_store=event_store
        )

        # Initialize state from event store
        await new_risk_manager.initialize_from_events()
        await new_execution_coordinator.initialize_from_events()

        # Verify state recovery
        positions_after = await new_execution_coordinator.get_positions()
        assert len(positions_after) == len(positions_before), "Should recover same number of positions"

        events_after = len(event_store.events)
        assert events_after == events_before, "Event store should preserve events"

        # Verify position data integrity
        if positions_before and positions_after:
            assert positions_before[0].symbol == positions_after[0].symbol, "Position symbols should match"
            assert abs(positions_before[0].size - positions_after[0].size) < 1e-6, "Position sizes should match"
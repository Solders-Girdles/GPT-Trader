"""
Tests for advanced risk integration scenarios (exposure, correlation).
"""

import asyncio
from decimal import Decimal

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    OrderSide as Side,
)
from bot_v2.features.brokerages.core.interfaces import (
    OrderStatus,
)
from bot_v2.features.live_trade.risk.pre_trade_checks import ValidationError


class TestAdvancedRiskIntegration:
    """Test advanced risk integration scenarios."""

    @pytest.mark.asyncio
    async def test_tc_if_011_exposure_cap_validation_during_concurrent_orders(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-011: Exposure Cap Validation During Concurrent Orders."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create multiple concurrent orders that test exposure limits
        orders = [
            integration_test_scenarios.create_test_order(
                order_id="exposure_001", symbol="BTC-USD", side=Side.BUY, quantity=2.0
            ),
            integration_test_scenarios.create_test_order(
                order_id="exposure_002", symbol="ETH-USD", side=Side.BUY, quantity=5.0
            ),
            integration_test_scenarios.create_test_order(
                order_id="exposure_003", symbol="BTC-USD", side=Side.BUY, quantity=3.0
            ),
        ]

        # Test concurrent risk validation
        risk_tasks = []
        for order in orders:
            ctx = get_risk_validation_context(order)
            task = asyncio.create_task(
                asyncio.coroutine(
                    lambda o=order, ctx=ctx: risk_manager.pre_trade_validate(
                        symbol=o.symbol,
                        side=str(o.side),
                        quantity=o.quantity,
                        price=(
                            o.price if o.price is not None else Decimal("50000.0")
                        ),  # Default price for market orders
                        product=ctx["product"],
                        equity=ctx["equity"],
                        current_positions=ctx["current_positions"],
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
                    price=order.price,
                )
            )
            execution_tasks.append(task)

        execution_results = await asyncio.gather(*execution_tasks)

        # Verify executed orders
        for i, placed_order in enumerate(execution_results):
            assert (
                placed_order.status == OrderStatus.FILLED
            ), f"Order {passed_orders[i].id} should execute successfully"

    @pytest.mark.asyncio
    async def test_tc_if_012_correlation_risk_integration_with_multiple_positions(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-012: Correlation Risk Integration with Multiple Positions."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create correlated positions (BTC and ETH typically have high correlation)
        correlated_orders = [
            integration_test_scenarios.create_test_order(
                order_id="correlation_001", symbol="BTC-USD", side=Side.BUY, quantity=3.0
            ),
            integration_test_scenarios.create_test_order(
                order_id="correlation_002", symbol="ETH-USD", side=Side.BUY, quantity=10.0
            ),
        ]

        # Risk validation should consider correlation
        for order in correlated_orders:
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
                        price=order.price,
                    )
                )
                execution_tasks.append(task)

            execution_results = await asyncio.gather(*execution_tasks)

            for i, placed_order in enumerate(execution_results):
                assert (
                    placed_order.status == OrderStatus.FILLED
                ), f"Correlated order {correlated_orders[i].id} should execute"

            # Check correlation monitoring
            correlation_metrics = system["event_store"].get_metrics_by_name("portfolio_correlation")
            if correlation_metrics:
                assert (
                    correlation_metrics[-1]["value"] <= 0.9
                ), "Portfolio correlation should be monitored"

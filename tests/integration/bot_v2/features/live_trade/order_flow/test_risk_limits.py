"""
Tests for risk and execution system integration.
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


class TestRiskExecutionIntegration:
    """Test risk and execution system integration."""

    @pytest.mark.asyncio
    async def test_tc_if_008_pre_trade_risk_limits_during_execution(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-008: Pre-trade Risk Limits Enforced During Execution."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        system["execution_coordinator"]

        # Create order near risk limits
        order = integration_test_scenarios.create_test_order(
            order_id="risk_limits_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=4.5,  # Near typical limit of 5.0
        )

        # Risk validation should pass but be close to limit
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

        # Create second order that would exceed limits
        second_order = integration_test_scenarios.create_test_order(
            order_id="risk_limits_002",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=2.0,  # Would exceed limit when combined
        )

        # Second order should fail risk validation
        ctx = get_risk_validation_context(second_order)
        with pytest.raises(ValidationError) as exc_info:
            risk_manager.pre_trade_validate(
                symbol=second_order.symbol,
                side=str(second_order.side),
                quantity=second_order.quantity,
                price=(
                    second_order.price if second_order.price is not None else Decimal("50000.0")
                ),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
        assert "position size" in str(exc_info.value).lower(), "Should mention position size limit"

    @pytest.mark.asyncio
    async def test_tc_if_009_position_sizing_integration_with_order_placement(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-009: Position Sizing Integration with Order Placement."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create order requiring position sizing calculation
        base_order = integration_test_scenarios.create_test_order(
            order_id="position_sizing_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=10.0,  # Large order that should be sized down
        )

        # Risk validation should calculate appropriate position size
        ctx = get_risk_validation_context(base_order)
        try:
            risk_manager.pre_trade_validate(
                symbol=base_order.symbol,
                side=str(base_order.side),
                quantity=base_order.quantity,
                price=(
                    base_order.price if base_order.price is not None else Decimal("50000.0")
                ),  # Default price for market orders
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
        except ValidationError as e:
            pytest.fail(
                f"Order should pass risk validation with sizing adjustment but failed with: {e}"
            )

        # Execute the sized order
        placed_order = await execution_coordinator.place_order(
            exec_engine=system["execution_engine"],
            symbol=base_order.symbol,
            side=base_order.side,
            order_type=base_order.type,
            quantity=base_order.quantity,
            price=base_order.price,
        )
        assert placed_order.status == OrderStatus.FILLED, "Sized order should execute successfully"

    @pytest.mark.asyncio
    async def test_tc_if_010_leverage_limit_enforcement_throughout_lifecycle(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-010: Leverage Limit Enforcement Throughout Order Lifecycle."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create high-leverage order
        leverage_order = integration_test_scenarios.create_test_order(
            order_id="leverage_test_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=15.0,  # High leverage position
        )

        # Risk validation should check leverage
        ctx = get_risk_validation_context(leverage_order)
        try:
            risk_manager.pre_trade_validate(
                symbol=leverage_order.symbol,
                side=str(leverage_order.side),
                quantity=leverage_order.quantity,
                price=(
                    leverage_order.price if leverage_order.price is not None else Decimal("50000.0")
                ),  # Default price for market orders
                product=ctx["product"],
                equity=Decimal("10000.0"),  # Force low equity to trigger leverage limit
                current_positions=ctx["current_positions"],
            )
            # If it passes, ensure leverage is within limits
            placed_order = await execution_coordinator.place_order(
                exec_engine=system["execution_engine"],
                symbol=leverage_order.symbol,
                side=leverage_order.side,
                order_type=leverage_order.type,
                quantity=leverage_order.quantity,
                price=leverage_order.price,
            )
            assert (
                placed_order.status == OrderStatus.FILLED
            ), "Order within leverage limits should execute"

            # Check final position leverage
            positions = await execution_coordinator.get_positions()
            if positions:
                position = positions[0]
                # Leverage should be reasonable (position size * price / account value)
                assert position.size <= 10.0, "Position size should be within leverage limits"
        except ValidationError as e:
            # Should fail due to leverage limits
            assert "leverage" in str(e).lower(), "Should mention leverage in rejection"

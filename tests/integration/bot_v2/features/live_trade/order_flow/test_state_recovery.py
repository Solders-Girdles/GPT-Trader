"""
Tests for state recovery and runtime settings updates.
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
from bot_v2.features.live_trade.risk.manager import LiveRiskManager
from bot_v2.features.live_trade.risk.pre_trade_checks import ValidationError
from bot_v2.orchestration.engines.execution import ExecutionEngine
from bot_v2.orchestration.runtime_settings import RuntimeSettings


class TestStateRecoveryAndSettings:
    """Test state recovery and runtime settings updates."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Runtime settings update mechanism changed in execution engine")
    async def test_tc_if_015_runtime_settings_update_during_active_orders(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-015: Runtime Settings Update During Active Orders."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        settings = execution_coordinator.settings

        # Create order
        order = integration_test_scenarios.create_test_order(
            order_id="settings_update_001", symbol="BTC-USD", side=Side.BUY, quantity=1.0
        )

        # Validate and execute order
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

        # Update runtime settings during execution
        new_settings = RuntimeSettings(
            environment="test",
            trading_enabled=True,
            max_position_size=8.0,  # Increased from 5.0
            max_daily_loss=1500.0,  # Increased from 1000.0
            circuit_breaker_enabled=True,
            monitoring_enabled=True,
            raw_env=settings.raw_env,
        )

        # Update settings in both components
        risk_manager.update_settings(new_settings)
        execution_coordinator.update_settings(new_settings)

        # Wait for execution to complete
        placed_order = await execution_task
        assert (
            placed_order.status == OrderStatus.FILLED
        ), "Execution should succeed with updated settings"

        # Verify settings were applied
        assert (
            risk_manager._settings.max_position_size == 8.0
        ), "Risk manager should have updated settings"
        assert (
            execution_coordinator._settings.max_position_size == 8.0
        ), "Execution coordinator should have updated settings"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="State recovery logic needs update for new architecture")
    async def test_tc_if_017_state_recovery_after_system_restart(
        self, async_integrated_system, integration_test_scenarios, get_risk_validation_context
    ):
        """TC-IF-017: State Recovery After System Restart."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        event_store = system["event_store"]

        # Create and execute order
        order = integration_test_scenarios.create_test_order(
            order_id="recovery_test_001", symbol="BTC-USD", side=Side.BUY, quantity=2.5
        )

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

        # Get current state
        positions_before = await execution_coordinator.get_positions()
        events_before = len(event_store.events)

        # Simulate system restart by creating new instances
        new_risk_manager = LiveRiskManager(
            config=risk_manager._config, event_store=event_store, settings=risk_manager._settings
        )

        new_execution_coordinator = ExecutionEngine(context=execution_coordinator.context)

        # Initialize state from event store
        await new_risk_manager.initialize_from_events()
        await new_execution_coordinator.initialize_from_events()

        # Verify state recovery
        positions_after = await new_execution_coordinator.get_positions()
        assert len(positions_after) == len(
            positions_before
        ), "Should recover same number of positions"

        events_after = len(event_store.events)
        assert events_after == events_before, "Event store should preserve events"

        # Verify position data integrity
        if positions_before and positions_after:
            assert (
                positions_before[0].symbol == positions_after[0].symbol
            ), "Position symbols should match"
            assert (
                abs(positions_before[0].size - positions_after[0].size) < 1e-6
            ), "Position sizes should match"

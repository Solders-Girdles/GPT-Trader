"""
Advanced execute_decision flows (reduce-only, kwargs, errors).
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.core.interfaces import TimeInForce
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision


class TestExecutionCoordinatorExecuteDecisionAdvanced:
    """Cover advanced engines, reduce-only, and error handling."""

    @pytest.mark.asyncio
    async def test_applies_global_reduce_only(
        self, execution_coordinator, execution_context, fake_product, fake_order
    ):
        initialized_context = execution_coordinator.initialize(execution_context)
        execution_coordinator.update_context(initialized_context)

        decision = Decision(action=Action.BUY, quantity=Decimal("0.1"), reason="test")

        config_controller = Mock()
        config_controller.is_reduce_only_mode = Mock(return_value=True)
        execution_coordinator._config_controller = config_controller

        runtime_state = execution_coordinator.context.runtime_state
        assert runtime_state is not None and runtime_state.exec_engine is not None
        runtime_state.exec_engine.place_order = Mock(return_value=fake_order)

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )

        runtime_state.exec_engine.place_order.assert_called_once()
        call_args = runtime_state.exec_engine.place_order.call_args
        assert call_args[1]["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_parses_time_in_force_for_advanced_engine(
        self, execution_coordinator, execution_context, fake_product, fake_order
    ):
        execution_context.risk_manager.config.enable_dynamic_position_sizing = True

        initialized_context = execution_coordinator.initialize(execution_context)
        execution_coordinator.update_context(initialized_context)

        runtime_state = execution_coordinator.context.runtime_state
        assert runtime_state is not None and runtime_state.exec_engine is not None
        runtime_state.exec_engine.place_order = Mock(return_value=fake_order)

        decision = Decision(
            action=Action.BUY, quantity=Decimal("0.1"), time_in_force="IOC", reason="test"
        )

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )

        runtime_state.exec_engine.place_order.assert_called_once()
        call_args = runtime_state.exec_engine.place_order.call_args
        assert call_args[1]["time_in_force"] == TimeInForce.IOC

    @pytest.mark.asyncio
    async def test_handles_invalid_time_in_force(
        self, execution_coordinator, execution_context, fake_product
    ):
        execution_context.runtime_state.exec_engine = Mock()
        execution_context.runtime_state.exec_engine.place_order = Mock(return_value=Mock())
        decision = Decision(
            action=Action.BUY, quantity=Decimal("0.1"), time_in_force="INVALID", reason="test"
        )

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )

        execution_context.runtime_state.exec_engine.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_advanced_engine(
        self, execution_coordinator, execution_context, fake_product
    ):
        execution_context.runtime_state.exec_engine = Mock(spec=AdvancedExecutionEngine)
        execution_context.runtime_state.exec_engine.place_order = Mock(return_value=Mock())
        decision = Decision(
            action=Action.BUY,
            quantity=Decimal("0.1"),
            limit_price=Decimal("51000"),
            stop_trigger=Decimal("49000"),
            time_in_force="GTC",
            leverage=2,
            reason="test",
        )

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )

        execution_context.runtime_state.exec_engine.place_order.assert_called_once()
        call_args = execution_context.runtime_state.exec_engine.place_order.call_args
        assert call_args[1]["limit_price"] == Decimal("51000")
        assert call_args[1]["stop_price"] == Decimal("49000")
        assert call_args[1]["time_in_force"] == TimeInForce.GTC
        assert call_args[1]["leverage"] == 2

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_live_engine(
        self, execution_coordinator, execution_context, fake_product
    ):
        execution_context.runtime_state.exec_engine = Mock()
        execution_context.runtime_state.exec_engine.place_order = Mock(return_value=Mock())

        decision = Decision(
            action=Action.BUY,
            quantity=Decimal("0.1"),
            limit_price=Decimal("51000"),
            stop_trigger=Decimal("49000"),
            time_in_force="IOC",
            reason="test",
        )

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )

        execution_context.runtime_state.exec_engine.place_order.assert_called_once()
        call_args = execution_context.runtime_state.exec_engine.place_order.call_args
        assert call_args[1]["product"] == fake_product
        assert call_args[1]["price"] == Decimal("51000")
        assert call_args[1]["stop_price"] == Decimal("49000")
        assert call_args[1]["tif"] == TimeInForce.IOC

    @pytest.mark.asyncio
    async def test_logs_exception_without_raising(
        self, execution_coordinator, execution_context, fake_product
    ):
        execution_context.runtime_state.exec_engine = Mock()
        execution_context.runtime_state.exec_engine.place_order = Mock(
            side_effect=Exception("Order failed")
        )
        decision = Decision(action=Action.BUY, quantity=Decimal("0.1"), reason="test")

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )

        execution_context.runtime_state.exec_engine.place_order.assert_called_once()

"""
Error-handling tests for ExecutionCoordinator.place_order.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType
from bot_v2.features.live_trade.risk import ValidationError as RiskValidationError


class TestExecutionCoordinatorPlaceOrderErrors:
    """Ensure ExecutionCoordinator propagates place_order failures."""

    @pytest.mark.asyncio
    async def test_validation_error_is_raised(self, execution_coordinator, execution_context):
        """Runtime ValidationError bubbles up to callers."""
        initialized_context = execution_coordinator.initialize(execution_context)
        execution_coordinator.update_context(initialized_context)

        runtime_state = execution_coordinator.context.runtime_state
        assert runtime_state is not None and runtime_state.exec_engine is not None

        runtime_state.exec_engine.place_order = Mock(side_effect=ValidationError("Invalid order"))

        with pytest.raises(ValidationError):
            await execution_coordinator.place_order(
                runtime_state.exec_engine,
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                order_type=OrderType.MARKET,
            )

    @pytest.mark.asyncio
    async def test_risk_validation_error_is_raised(self, execution_coordinator, execution_context):
        """RiskValidationError should be surfaced to callers."""
        initialized_context = execution_coordinator.initialize(execution_context)
        execution_coordinator.update_context(initialized_context)

        runtime_state = execution_coordinator.context.runtime_state
        assert runtime_state is not None and runtime_state.exec_engine is not None

        runtime_state.exec_engine.place_order = Mock(
            side_effect=RiskValidationError("Risk check failed")
        )

        with pytest.raises(RiskValidationError):
            await execution_coordinator.place_order(
                runtime_state.exec_engine,
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                order_type=OrderType.MARKET,
            )

    @pytest.mark.asyncio
    async def test_execution_error_is_raised(self, execution_coordinator, execution_context):
        """ExecutionError is propagated without suppression."""
        initialized_context = execution_coordinator.initialize(execution_context)
        execution_coordinator.update_context(initialized_context)

        runtime_state = execution_coordinator.context.runtime_state
        assert runtime_state is not None and runtime_state.exec_engine is not None

        runtime_state.exec_engine.place_order = Mock(side_effect=ExecutionError("Execution failed"))

        with pytest.raises(ExecutionError):
            await execution_coordinator.place_order(
                runtime_state.exec_engine,
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                order_type=OrderType.MARKET,
            )

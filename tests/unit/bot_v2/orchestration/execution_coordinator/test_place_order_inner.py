"""
Success-path tests for ExecutionCoordinator.place_order_inner.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType


class TestExecutionCoordinatorPlaceOrderInner:
    """Exercise place_order_inner across engine variations."""

    @pytest.mark.asyncio
    async def test_returns_order_with_advanced_engine(
        self, execution_coordinator, execution_context, fake_order
    ):
        execution_context.risk_manager.config.enable_dynamic_position_sizing = True

        initialized_context = execution_coordinator.initialize(execution_context)
        execution_coordinator.update_context(initialized_context)

        runtime_state = execution_coordinator.context.runtime_state
        assert runtime_state is not None and runtime_state.exec_engine is not None

        runtime_state.exec_engine.place_order = Mock(return_value=fake_order)

        orders_store = Mock()
        execution_coordinator.update_context(
            execution_coordinator.context.with_updates(orders_store=orders_store)
        )

        result = await execution_coordinator.place_order_inner(
            runtime_state.exec_engine,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        assert result == fake_order
        orders_store.upsert.assert_called_once_with(fake_order)

    @pytest.mark.asyncio
    async def test_fetches_order_with_live_engine(
        self, execution_coordinator, execution_context, fake_order
    ):
        initialized_context = execution_coordinator.initialize(execution_context)
        execution_coordinator.update_context(initialized_context)

        runtime_state = execution_coordinator.context.runtime_state
        assert runtime_state is not None and runtime_state.exec_engine is not None

        runtime_state.exec_engine.place_order = Mock(return_value="order-123")

        broker_mock = Mock()
        broker_mock.get_order = AsyncMock(return_value=fake_order)
        orders_store = Mock()

        execution_coordinator.update_context(
            execution_coordinator.context.with_updates(
                broker=broker_mock, orders_store=orders_store
            )
        )

        with patch(
            "bot_v2.orchestration.coordinators.execution.run_in_thread"
        ) as mock_run_in_thread:
            mock_run_in_thread.return_value = fake_order

            result = await execution_coordinator.place_order_inner(
                runtime_state.exec_engine,
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                order_type=OrderType.MARKET,
            )

            assert result == fake_order
            orders_store.upsert.assert_called_once_with(fake_order)

    @pytest.mark.asyncio
    async def test_returns_none_when_runtime_state_missing(
        self, execution_coordinator, execution_context
    ):
        execution_context = execution_context.with_updates(runtime_state=None)
        execution_coordinator.update_context(execution_context)

        result = await execution_coordinator.place_order_inner(
            Mock(),
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        assert result is None

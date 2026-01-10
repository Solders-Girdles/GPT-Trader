"""Tests for OrderRouter."""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.core import Order, OrderSide
from gpt_trader.features.live_trade.execution.router import OrderResult, OrderRouter
from gpt_trader.features.live_trade.strategies.hybrid.types import (
    Action,
    HybridDecision,
    TradingMode,
)


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_success_result(self) -> None:
        """Creates successful result."""
        order = Mock(spec=Order)
        order.order_id = "test-123"
        order.symbol = "BTC-USD"

        result = OrderResult(success=True, order=order)

        assert result.success is True
        assert result.order == order
        assert result.error is None

    def test_failure_result(self) -> None:
        """Creates failure result."""
        result = OrderResult(
            success=False,
            error="Test error",
            error_code="TEST_ERROR",
        )

        assert result.success is False
        assert result.error == "Test error"
        assert result.error_code == "TEST_ERROR"

    def test_to_dict_success(self) -> None:
        """Serializes successful result."""
        order = Mock(spec=Order)
        order.order_id = "test-123"
        order.symbol = "BTC-USD"

        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
        )

        result = OrderResult(success=True, order=order, decision=decision)
        data = result.to_dict()

        assert data["success"] is True
        assert data["order_id"] == "test-123"
        assert data["mode"] == "spot_only"

    def test_to_dict_failure(self) -> None:
        """Serializes failure result."""
        result = OrderResult(
            success=False,
            error="Test error",
            error_code="TEST_ERROR",
        )
        data = result.to_dict()

        assert data["success"] is False
        assert data["error"] == "Test error"
        assert data["error_code"] == "TEST_ERROR"


class TestResolveOrderIdFallback:
    """Tests for _resolve_order_id fallback behavior."""

    def test_resolve_order_id_with_id_attribute(self) -> None:
        """Test that 'id' attribute is used first."""
        from gpt_trader.features.live_trade.execution.router import _resolve_order_id

        order = Mock()
        order.id = "primary-id-123"
        order.order_id = "fallback-id-456"

        result = _resolve_order_id(order)
        assert result == "primary-id-123"

    def test_resolve_order_id_fallback_to_order_id(self) -> None:
        """Test fallback to 'order_id' when 'id' is None."""
        from gpt_trader.features.live_trade.execution.router import _resolve_order_id

        order = Mock()
        order.id = None
        order.order_id = "fallback-id-456"

        result = _resolve_order_id(order)
        assert result == "fallback-id-456"

    def test_resolve_order_id_fallback_when_id_missing(self) -> None:
        """Test fallback to 'order_id' when 'id' attribute doesn't exist."""
        from gpt_trader.features.live_trade.execution.router import _resolve_order_id

        order = Mock(spec=["order_id"])
        order.order_id = "fallback-id-789"

        result = _resolve_order_id(order)
        assert result == "fallback-id-789"

    def test_resolve_order_id_returns_none_for_none_order(self) -> None:
        """Test that None order returns None."""
        from gpt_trader.features.live_trade.execution.router import _resolve_order_id

        result = _resolve_order_id(None)
        assert result is None

    def test_resolve_order_id_returns_none_when_both_missing(self) -> None:
        """Test that None is returned when both id and order_id are missing/None."""
        from gpt_trader.features.live_trade.execution.router import _resolve_order_id

        order = Mock(spec=[])

        result = _resolve_order_id(order)
        assert result is None


class TestOrderRouterAsyncExecution:
    """Tests for OrderRouter.execute_async() canonical path."""

    @pytest.mark.asyncio
    async def test_execute_async_delegates_to_submitter(self) -> None:
        """execute_async delegates to the submitter callable."""
        submitter = AsyncMock()
        equity_provider = Mock(return_value=Decimal("10000"))

        router = OrderRouter(
            submitter=submitter,
            equity_provider=equity_provider,
        )

        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("0.1"),
            confidence=0.85,
        )
        price = Decimal("50000")

        result = await router.execute_async(decision, price)

        assert result.success is True
        submitter.assert_awaited_once()
        call_args = submitter.call_args
        assert call_args[0][0] == "BTC-USD"
        assert call_args[0][1] == OrderSide.BUY
        assert call_args[0][2] == Decimal("50000")
        assert call_args[0][3] == Decimal("10000")
        assert call_args[1]["quantity_override"] == Decimal("0.1")
        assert call_args[1]["reduce_only"] is False

    @pytest.mark.asyncio
    async def test_execute_async_passes_reduce_only_for_close(self) -> None:
        """execute_async sets reduce_only for close actions."""
        submitter = AsyncMock()
        equity_provider = Mock(return_value=Decimal("10000"))

        router = OrderRouter(
            submitter=submitter,
            equity_provider=equity_provider,
        )

        decision = HybridDecision(
            action=Action.CLOSE_LONG,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("0.1"),
        )
        price = Decimal("50000")

        result = await router.execute_async(decision, price)

        assert result.success is True
        call_args = submitter.call_args
        assert call_args[1]["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_execute_async_handles_submitter_error(self) -> None:
        """execute_async returns failure on submitter exception."""
        submitter = AsyncMock(side_effect=Exception("Guard rejected"))
        equity_provider = Mock(return_value=Decimal("10000"))

        router = OrderRouter(
            submitter=submitter,
            equity_provider=equity_provider,
        )

        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("0.1"),
        )
        price = Decimal("50000")

        result = await router.execute_async(decision, price)

        assert result.success is False
        assert result.error is not None
        assert "Guard rejected" in result.error
        assert result.error_code == "EXECUTION_ERROR"

    @pytest.mark.asyncio
    async def test_execute_async_hold_is_not_actionable(self) -> None:
        """execute_async returns success without calling submitter for HOLD."""
        submitter = AsyncMock()
        equity_provider = Mock(return_value=Decimal("10000"))

        router = OrderRouter(
            submitter=submitter,
            equity_provider=equity_provider,
        )

        decision = HybridDecision(
            action=Action.HOLD,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
        )
        price = Decimal("50000")

        result = await router.execute_async(decision, price)

        assert result.success is True
        assert result.error is not None
        assert "not actionable" in result.error
        submitter.assert_not_awaited()

"""Tests for OrderRouter.execute_async."""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.core import OrderSide
from gpt_trader.features.live_trade.execution.router import OrderRouter
from gpt_trader.features.live_trade.execution.submission_result import (
    OrderSubmissionResult,
    OrderSubmissionStatus,
)
from gpt_trader.features.live_trade.strategies.hybrid.types import (
    Action,
    HybridDecision,
    TradingMode,
)


@pytest.fixture(autouse=True)
def reset_metrics() -> None:
    from gpt_trader.monitoring.metrics_collector import reset_all

    reset_all()
    yield
    reset_all()


class TestOrderRouterAsyncExecution:
    """Tests for OrderRouter.execute_async() canonical path."""

    @pytest.mark.asyncio
    async def test_execute_async_delegates_to_submitter(self) -> None:
        """execute_async delegates to the submitter callable."""
        submitter = AsyncMock(
            return_value=OrderSubmissionResult(
                status=OrderSubmissionStatus.SUCCESS,
                order_id="order-123",
            )
        )
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
        submitter = AsyncMock(
            return_value=OrderSubmissionResult(
                status=OrderSubmissionStatus.SUCCESS,
                order_id="order-123",
            )
        )
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
    async def test_execute_async_handles_submitter_failure_result(self) -> None:
        """execute_async returns failure on submitter failure result."""
        submitter = AsyncMock(
            return_value=OrderSubmissionResult(
                status=OrderSubmissionStatus.FAILED,
                error="Guard rejected",
            )
        )
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
    async def test_execute_async_handles_submitter_blocked_result(self) -> None:
        """execute_async returns blocked on guard rejection result."""
        submitter = AsyncMock(
            return_value=OrderSubmissionResult(
                status=OrderSubmissionStatus.BLOCKED,
                reason="paused:degraded",
            )
        )
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
        assert "paused" in result.error
        assert result.error_code == "ORDER_BLOCKED"

        from gpt_trader.monitoring.metrics_collector import get_metrics_collector

        collector = get_metrics_collector()
        assert collector.counters["gpt_trader_trades_blocked_total"] == 1

    @pytest.mark.asyncio
    async def test_execute_async_handles_submitter_exception(self) -> None:
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

"""Tests for OrderResult."""

from unittest.mock import Mock

from gpt_trader.core import Order
from gpt_trader.features.live_trade.execution.router import OrderResult
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

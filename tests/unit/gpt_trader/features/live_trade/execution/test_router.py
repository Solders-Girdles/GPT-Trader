"""Tests for OrderRouter."""

from decimal import Decimal
from unittest.mock import MagicMock, Mock

from gpt_trader.core import Order, OrderSide, OrderType
from gpt_trader.features.live_trade.execution.router import OrderResult, OrderRouter
from gpt_trader.features.live_trade.risk.manager import LiveRiskManager, ValidationError
from gpt_trader.features.live_trade.strategies.hybrid.types import (
    Action,
    HybridDecision,
    TradingMode,
)


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_success_result(self):
        """Creates successful result."""
        order = Mock(spec=Order)
        order.order_id = "test-123"
        order.symbol = "BTC-USD"

        result = OrderResult(success=True, order=order)

        assert result.success is True
        assert result.order == order
        assert result.error is None

    def test_failure_result(self):
        """Creates failure result."""
        result = OrderResult(
            success=False,
            error="Test error",
            error_code="TEST_ERROR",
        )

        assert result.success is False
        assert result.error == "Test error"
        assert result.error_code == "TEST_ERROR"

    def test_to_dict_success(self):
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

    def test_to_dict_failure(self):
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


class TestOrderRouterExecuteSpot:
    """Tests for OrderRouter spot execution."""

    def test_execute_spot_buy(self):
        """Executes spot buy order."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order.order_id = "spot-123"
        order_service.place_order.return_value = order

        router = OrderRouter(order_service=order_service)

        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("1.0"),
        )

        result = router.execute(decision)

        assert result.success is True
        assert result.order == order
        order_service.place_order.assert_called_once_with(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            reduce_only=False,
        )

    def test_execute_spot_sell(self):
        """Executes spot sell order."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order.order_id = "spot-456"
        order_service.place_order.return_value = order

        router = OrderRouter(order_service=order_service)

        decision = HybridDecision(
            action=Action.SELL,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("0.5"),
        )

        result = router.execute(decision)

        assert result.success is True
        order_service.place_order.assert_called_once_with(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            reduce_only=False,
        )

    def test_execute_spot_close(self):
        """Executes spot close order with reduce_only."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order.order_id = "spot-789"
        order_service.place_order.return_value = order

        router = OrderRouter(order_service=order_service)

        decision = HybridDecision(
            action=Action.CLOSE,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("1.0"),
        )

        result = router.execute(decision)

        assert result.success is True
        order_service.place_order.assert_called_once_with(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            reduce_only=True,
        )

    def test_execute_spot_error(self):
        """Handles spot execution error."""
        order_service = MagicMock()
        order_service.place_order.side_effect = Exception("Connection failed")

        router = OrderRouter(order_service=order_service)

        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("1.0"),
        )

        result = router.execute(decision)

        assert result.success is False
        assert "Connection failed" in result.error
        assert result.error_code == "EXECUTION_ERROR"


class TestOrderRouterExecuteCFM:
    """Tests for OrderRouter CFM execution."""

    def test_execute_cfm_buy(self):
        """Executes CFM buy order."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order.order_id = "cfm-123"
        order_service.place_order.return_value = order

        router = OrderRouter(order_service=order_service)

        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-20DEC30-CDE",
            mode=TradingMode.CFM_ONLY,
            quantity=Decimal("0.5"),
            leverage=3,
        )

        result = router.execute(decision)

        assert result.success is True
        assert result.order == order
        order_service.place_order.assert_called_once_with(
            symbol="BTC-20DEC30-CDE",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            reduce_only=False,
            leverage=3,
        )

    def test_execute_cfm_with_leverage_1(self):
        """CFM with leverage 1 doesn't pass leverage parameter."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order.order_id = "cfm-456"
        order_service.place_order.return_value = order

        router = OrderRouter(order_service=order_service)

        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-20DEC30-CDE",
            mode=TradingMode.CFM_ONLY,
            quantity=Decimal("1.0"),
            leverage=1,
        )

        result = router.execute(decision)

        assert result.success is True
        order_service.place_order.assert_called_once_with(
            symbol="BTC-20DEC30-CDE",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            reduce_only=False,
            leverage=None,  # Not passed for leverage=1
        )

    def test_execute_cfm_close_short(self):
        """Executes CFM close short (buy to cover)."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order.order_id = "cfm-789"
        order_service.place_order.return_value = order

        router = OrderRouter(order_service=order_service)

        decision = HybridDecision(
            action=Action.CLOSE_SHORT,
            symbol="BTC-20DEC30-CDE",
            mode=TradingMode.CFM_ONLY,
            quantity=Decimal("1.0"),
        )

        result = router.execute(decision)

        assert result.success is True
        order_service.place_order.assert_called_once_with(
            symbol="BTC-20DEC30-CDE",
            side=OrderSide.BUY,  # Close short = buy
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            reduce_only=True,
            leverage=None,
        )

    def test_execute_cfm_leverage_validation(self):
        """Validates leverage with risk manager."""
        order_service = MagicMock()
        risk_manager = MagicMock(spec=LiveRiskManager)
        risk_manager.validate_cfm_leverage.side_effect = ValidationError("Exceeds limit")
        risk_manager.is_cfm_reduce_only_mode.return_value = False
        risk_manager.is_reduce_only_mode.return_value = False

        router = OrderRouter(order_service=order_service, risk_manager=risk_manager)

        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-20DEC30-CDE",
            mode=TradingMode.CFM_ONLY,
            quantity=Decimal("1.0"),
            leverage=10,
        )

        result = router.execute(decision)

        assert result.success is False
        assert "Exceeds limit" in result.error
        assert result.error_code == "LEVERAGE_EXCEEDED"
        order_service.place_order.assert_not_called()


class TestOrderRouterReduceOnlyMode:
    """Tests for reduce-only mode enforcement."""

    def test_blocks_new_position_in_reduce_only(self):
        """Blocks new positions when in global reduce-only mode."""
        order_service = MagicMock()
        risk_manager = MagicMock(spec=LiveRiskManager)
        risk_manager.is_reduce_only_mode.return_value = True
        risk_manager.is_cfm_reduce_only_mode.return_value = False

        router = OrderRouter(order_service=order_service, risk_manager=risk_manager)

        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("1.0"),
        )

        result = router.execute(decision)

        assert result.success is False
        assert result.error_code == "REDUCE_ONLY"
        order_service.place_order.assert_not_called()

    def test_allows_close_in_reduce_only(self):
        """Allows position closes in reduce-only mode."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order.order_id = "close-123"
        order_service.place_order.return_value = order

        risk_manager = MagicMock(spec=LiveRiskManager)
        risk_manager.is_reduce_only_mode.return_value = True
        risk_manager.is_cfm_reduce_only_mode.return_value = False

        router = OrderRouter(order_service=order_service, risk_manager=risk_manager)

        decision = HybridDecision(
            action=Action.CLOSE,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("1.0"),
        )

        result = router.execute(decision)

        assert result.success is True
        order_service.place_order.assert_called_once()

    def test_blocks_cfm_new_position_in_cfm_reduce_only(self):
        """Blocks new CFM positions when in CFM reduce-only mode."""
        order_service = MagicMock()
        risk_manager = MagicMock(spec=LiveRiskManager)
        risk_manager.is_reduce_only_mode.return_value = False
        risk_manager.is_cfm_reduce_only_mode.return_value = True

        router = OrderRouter(order_service=order_service, risk_manager=risk_manager)

        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-20DEC30-CDE",
            mode=TradingMode.CFM_ONLY,
            quantity=Decimal("1.0"),
        )

        result = router.execute(decision)

        assert result.success is False
        assert result.error_code == "CFM_REDUCE_ONLY"

    def test_allows_spot_in_cfm_reduce_only(self):
        """Allows spot trades when only CFM reduce-only is active."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order.order_id = "spot-123"
        order_service.place_order.return_value = order

        risk_manager = MagicMock(spec=LiveRiskManager)
        risk_manager.is_reduce_only_mode.return_value = False
        risk_manager.is_cfm_reduce_only_mode.return_value = True

        router = OrderRouter(order_service=order_service, risk_manager=risk_manager)

        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("1.0"),
        )

        result = router.execute(decision)

        assert result.success is True
        order_service.place_order.assert_called_once()


class TestOrderRouterHold:
    """Tests for HOLD decision handling."""

    def test_hold_is_not_actionable(self):
        """HOLD decisions return success without execution."""
        order_service = MagicMock()
        router = OrderRouter(order_service=order_service)

        decision = HybridDecision(
            action=Action.HOLD,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
        )

        result = router.execute(decision)

        assert result.success is True
        assert "not actionable" in result.error
        order_service.place_order.assert_not_called()


class TestOrderRouterBatch:
    """Tests for batch execution."""

    def test_execute_batch_success(self):
        """Executes batch of decisions successfully."""
        order_service = MagicMock()
        order1 = Mock(spec=Order)
        order1.order_id = "order-1"
        order2 = Mock(spec=Order)
        order2.order_id = "order-2"
        order_service.place_order.side_effect = [order1, order2]

        router = OrderRouter(order_service=order_service)

        decisions = [
            HybridDecision(
                action=Action.BUY,
                symbol="BTC-USD",
                mode=TradingMode.SPOT_ONLY,
                quantity=Decimal("1.0"),
            ),
            HybridDecision(
                action=Action.SELL,
                symbol="ETH-USD",
                mode=TradingMode.SPOT_ONLY,
                quantity=Decimal("5.0"),
            ),
        ]

        results = router.execute_batch(decisions)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert order_service.place_order.call_count == 2

    def test_execute_batch_skips_hold(self):
        """Skips HOLD decisions in batch."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order.order_id = "order-1"
        order_service.place_order.return_value = order

        router = OrderRouter(order_service=order_service)

        decisions = [
            HybridDecision(
                action=Action.HOLD,
                symbol="BTC-USD",
                mode=TradingMode.SPOT_ONLY,
            ),
            HybridDecision(
                action=Action.BUY,
                symbol="ETH-USD",
                mode=TradingMode.SPOT_ONLY,
                quantity=Decimal("1.0"),
            ),
        ]

        results = router.execute_batch(decisions)

        assert len(results) == 2
        assert results[0].success is True  # HOLD is "success" but no execution
        assert results[1].success is True
        assert order_service.place_order.call_count == 1  # Only BUY executed


class TestOrderRouterActionMapping:
    """Tests for action to side mapping."""

    def test_buy_maps_to_buy(self):
        """BUY action maps to BUY side."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order_service.place_order.return_value = order
        router = OrderRouter(order_service=order_service)

        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("1.0"),
        )
        router.execute(decision)

        call_args = order_service.place_order.call_args
        assert call_args.kwargs["side"] == OrderSide.BUY

    def test_sell_maps_to_sell(self):
        """SELL action maps to SELL side."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order_service.place_order.return_value = order
        router = OrderRouter(order_service=order_service)

        decision = HybridDecision(
            action=Action.SELL,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("1.0"),
        )
        router.execute(decision)

        call_args = order_service.place_order.call_args
        assert call_args.kwargs["side"] == OrderSide.SELL

    def test_close_long_maps_to_sell(self):
        """CLOSE_LONG action maps to SELL side."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order_service.place_order.return_value = order
        router = OrderRouter(order_service=order_service)

        decision = HybridDecision(
            action=Action.CLOSE_LONG,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("1.0"),
        )
        router.execute(decision)

        call_args = order_service.place_order.call_args
        assert call_args.kwargs["side"] == OrderSide.SELL

    def test_close_short_maps_to_buy(self):
        """CLOSE_SHORT action maps to BUY side."""
        order_service = MagicMock()
        order = Mock(spec=Order)
        order_service.place_order.return_value = order
        router = OrderRouter(order_service=order_service)

        decision = HybridDecision(
            action=Action.CLOSE_SHORT,
            symbol="BTC-20DEC30-CDE",
            mode=TradingMode.CFM_ONLY,
            quantity=Decimal("1.0"),
        )
        router.execute(decision)

        call_args = order_service.place_order.call_args
        assert call_args.kwargs["side"] == OrderSide.BUY

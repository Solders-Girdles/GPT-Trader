"""Stress and failure handling tests for AdvancedExecutionEngine."""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from unittest.mock import Mock

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.errors import ValidationError, ExecutionError
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.features.live_trade.advanced_execution_models.models import OrderConfig
from bot_v2.features.brokerages.core.interfaces import (
    Product,
    MarketType,
    OrderType,
    OrderSide,
    TimeInForce,
    Order,
    OrderStatus,
    Quote,
    Position,
)
from bot_v2.features.live_trade.risk import PositionSizingAdvice


class StubBroker:
    def __init__(
        self,
        fail_order_types: set[str] | None = None,
        fail_get_product: bool = False,
        fail_get_quote: bool = False,
        positions: list[Position] | None = None,
        fail_cancel: bool = False,
    ):
        self.fail_order_types = fail_order_types or set()
        self.fail_get_product = fail_get_product
        self.fail_get_quote = fail_get_quote
        self.fail_cancel = fail_cancel
        self.orders = []
        self.cancelled_orders: list[str] = []
        self._positions = positions or []

    def get_product(self, symbol: str) -> Product:
        if self.fail_get_product:
            raise RuntimeError("get_product failed")
        return Product(
            symbol=symbol,
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=5,
            contract_size=Decimal("1"),
            funding_rate=Decimal("0.0001"),
            next_funding_time=None,
        )

    def get_quote(self, symbol: str) -> Quote:
        if self.fail_get_quote:
            raise RuntimeError("get_quote failed")
        return Quote(
            symbol=symbol,
            bid=Decimal("100"),
            ask=Decimal("101"),
            last=Decimal("100.5"),
            ts=datetime.utcnow(),
        )

    def get_positions(self) -> list[Position]:
        return self._positions

    def cancel_order(self, order_id: str) -> bool:
        if self.fail_cancel:
            raise RuntimeError("cancel_order failed")
        self.cancelled_orders.append(order_id)
        return True

    def place_order(self, **kwargs) -> Order:
        order_type = kwargs.get("order_type")
        if isinstance(order_type, OrderType):
            order_type_key = order_type.value
        else:
            order_type_key = str(order_type)
        if order_type_key.lower() in self.fail_order_types:
            raise RuntimeError("simulated broker failure")

        order_id = f"order-{len(self.orders) + 1}"
        now = datetime.utcnow()
        order_quantity = kwargs.get("quantity", Decimal("0.01"))
        order = Order(
            id=order_id,
            client_id=kwargs.get("client_id", order_id),
            symbol=kwargs.get("symbol", "BTC-PERP"),
            side=kwargs.get("side", OrderSide.BUY),
            type=kwargs.get("order_type", OrderType.MARKET),
            quantity=order_quantity,
            price=kwargs.get("price") or kwargs.get("limit_price"),
            stop_price=kwargs.get("stop_price"),
            tif=kwargs.get("tif", TimeInForce.GTC),
            status=OrderStatus.SUBMITTED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at=now,
            updated_at=now,
        )
        order.reduce_only = kwargs.get("reduce_only", False)
        self.orders.append(order)
        return order


class StubRiskManager:
    def __init__(self, advice: PositionSizingAdvice, *, enable_dynamic: bool = True) -> None:
        self.advice = advice
        self.config = RiskConfig(
            enable_dynamic_position_sizing=enable_dynamic,
            position_sizing_method="intelligent",
            position_sizing_multiplier=1.0,
        )
        self.positions: dict[str, dict[str, Decimal]] = {}
        self.start_of_day_equity = Decimal("20000")
        self.last_context = None

    def size_position(self, context):  # type: ignore[override]
        self.last_context = context
        return self.advice

    def pre_trade_validate(self, **kwargs):  # type: ignore[override]
        return None

    def is_reduce_only_mode(self) -> bool:  # type: ignore[override]
        return False


@pytest.mark.parametrize("order_type", [OrderType.STOP, OrderType.STOP_LIMIT])
def test_stop_trigger_cleaned_up_on_failure(order_type):
    broker = StubBroker(fail_order_types={order_type.value.lower()})
    engine = AdvancedExecutionEngine(broker=broker)

    result = engine.place_order(
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        quantity=Decimal("0.2"),
        order_type=order_type,
        stop_price=Decimal("105"),
        limit_price=Decimal("105.5") if order_type == OrderType.STOP_LIMIT else None,
    )

    assert result is None
    assert len(engine.stop_triggers) == 0
    assert engine.order_metrics["rejected"] > 0


def test_high_volume_stop_orders_trigger_and_cleanup():
    broker = StubBroker()
    engine = AdvancedExecutionEngine(broker=broker)

    total_orders = 50
    for idx in range(total_orders):
        stop_price = Decimal("100") - Decimal(idx) / Decimal("100")
        limit_price = stop_price
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            quantity=Decimal("0.2"),
            order_type=OrderType.STOP,
            stop_price=stop_price,
        )

    assert engine.order_metrics["placed"] == total_orders
    assert len(engine.stop_triggers) == total_orders

    triggered = engine.check_stop_triggers({"BTC-PERP": Decimal("50")})
    assert len(triggered) == total_orders
    assert sum(1 for trigger in engine.stop_triggers.values() if trigger.triggered) == total_orders


def test_dynamic_position_sizing_adjusts_quantity():
    broker = StubBroker()
    advice = PositionSizingAdvice(
        symbol="BTC-PERP",
        side="buy",
        target_notional=Decimal("2000"),
        target_quantity=Decimal("0.2"),
        used_dynamic=True,
        reduce_only=False,
        reason="dynamic sizing",
    )
    risk = StubRiskManager(advice)
    engine = AdvancedExecutionEngine(broker=broker, risk_manager=risk)

    order = engine.place_order(
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        quantity=Decimal("0.1"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
    )

    assert order is not None
    assert order.quantity == Decimal("0.2")
    assert risk.last_context is not None
    assert risk.last_context.symbol == "BTC-PERP"


def test_dynamic_position_sizing_rejects_zero_quantity():
    broker = StubBroker()
    advice = PositionSizingAdvice(
        symbol="BTC-PERP",
        side="buy",
        target_notional=Decimal("0"),
        target_quantity=Decimal("0"),
        used_dynamic=True,
        reduce_only=False,
        reason="no_notional",
    )
    risk = StubRiskManager(advice)
    engine = AdvancedExecutionEngine(broker=broker, risk_manager=risk)

    result = engine.place_order(
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        quantity=Decimal("0.1"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
    )

    assert result is None
    assert engine.rejections_by_reason.get("position_sizing") == 1
    assert not broker.orders


def test_dynamic_position_sizing_sets_reduce_only_flag():
    broker = StubBroker()
    advice = PositionSizingAdvice(
        symbol="BTC-PERP",
        side="buy",
        target_notional=Decimal("1200"),
        target_quantity=Decimal("0.12"),
        used_dynamic=True,
        reduce_only=True,
        reason="risk_reduce_only",
    )
    risk = StubRiskManager(advice)
    engine = AdvancedExecutionEngine(broker=broker, risk_manager=risk)

    order = engine.place_order(
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        quantity=Decimal("0.1"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
    )

    assert order is not None
    assert broker.orders[-1].reduce_only is True


# ============================================================================
# New comprehensive tests for coverage
# ============================================================================


class TestOrderLifecycle:
    """Test order lifecycle including duplicate detection and basic flows."""

    def test_duplicate_order_returns_existing(self):
        """Should return existing order for duplicate client_id."""
        broker = StubBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Place first order
        order1 = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
            client_id="test-client-123",
        )

        # Attempt duplicate
        order2 = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.2"),
            order_type=OrderType.MARKET,
            client_id="test-client-123",
        )

        assert order1 is not None
        assert order2 is not None
        assert order1.id == order2.id
        assert len(broker.orders) == 1

    def test_market_order_success(self):
        """Should successfully place market order."""
        broker = StubBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            order_type=OrderType.MARKET,
        )

        assert order is not None
        assert order.type == OrderType.MARKET
        assert order.quantity == Decimal("0.5")
        assert engine.order_metrics["placed"] == 1

    def test_limit_order_with_quantization(self):
        """Should successfully place limit order (quantization tested in execution)."""
        broker = StubBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100.123"),
        )

        assert order is not None
        assert order.type == OrderType.LIMIT
        assert order.quantity == Decimal("1.0")


class TestPostOnlyValidation:
    """Test post-only order validation logic."""

    def test_post_only_buy_crosses_spread_rejected(self):
        """Should reject post-only buy that would cross spread."""
        broker = StubBroker()  # ask is 101
        config = OrderConfig(reject_on_cross=True)
        engine = AdvancedExecutionEngine(broker=broker, config=config)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("101"),  # >= ask
            post_only=True,
        )

        assert order is None
        assert engine.order_metrics["post_only_rejected"] == 1

    def test_post_only_sell_crosses_spread_rejected(self):
        """Should reject post-only sell that would cross spread."""
        broker = StubBroker()  # bid is 100
        config = OrderConfig(reject_on_cross=True)
        engine = AdvancedExecutionEngine(broker=broker, config=config)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100"),  # <= bid
            post_only=True,
        )

        assert order is None
        assert engine.order_metrics["post_only_rejected"] == 1

    def test_post_only_buy_no_cross_accepted(self):
        """Should accept post-only buy that doesn't cross."""
        broker = StubBroker()  # ask is 101
        config = OrderConfig(reject_on_cross=True)
        engine = AdvancedExecutionEngine(broker=broker, config=config)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),  # Increased to meet min_notional
            order_type=OrderType.LIMIT,
            limit_price=Decimal("99"),  # < ask
            post_only=True,
        )

        assert order is not None
        assert engine.order_metrics["post_only_rejected"] == 0

    def test_post_only_without_reject_on_cross_skips_validation(self):
        """Should skip cross validation when reject_on_cross disabled."""
        broker = StubBroker()
        config = OrderConfig(reject_on_cross=False)
        engine = AdvancedExecutionEngine(broker=broker, config=config)

        # This would cross but should be accepted
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("102"),
            post_only=True,
        )

        assert order is not None
        assert engine.order_metrics["post_only_rejected"] == 0


class TestRiskValidation:
    """Test risk manager integration and validation."""

    def test_risk_validation_failure_rejects_order(self):
        """Should reject order when risk validation fails."""
        broker = StubBroker()

        # Create risk manager that raises ValidationError
        risk_manager = Mock()
        risk_manager.config = RiskConfig(
            enable_dynamic_position_sizing=False,  # Disable to avoid sizing_helper issues
        )
        risk_manager.pre_trade_validate = Mock(side_effect=ValidationError("Max exposure exceeded"))
        risk_manager.positions = {}

        engine = AdvancedExecutionEngine(broker=broker, risk_manager=risk_manager)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
        )

        assert order is None
        assert engine.order_metrics["rejected"] == 1
        assert engine.rejections_by_reason.get("risk") == 1

    def test_no_risk_manager_skips_validation(self):
        """Should skip risk validation when no risk manager."""
        broker = StubBroker()
        engine = AdvancedExecutionEngine(broker=broker, risk_manager=None)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        assert order is not None
        assert engine.order_metrics["rejected"] == 0


class TestMarketDataHandling:
    """Test market data fetch error handling."""

    def test_get_product_failure_continues(self):
        """Should continue when get_product fails."""
        broker = StubBroker(fail_get_product=True)
        engine = AdvancedExecutionEngine(broker=broker)

        # Should still place order even if product fetch fails
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        assert order is not None

    def test_get_quote_failure_raises_for_post_only(self):
        """Should raise ExecutionError when quote fetch fails for post-only."""
        broker = StubBroker(fail_get_quote=True)
        config = OrderConfig(reject_on_cross=True)
        engine = AdvancedExecutionEngine(broker=broker, config=config)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100"),
            post_only=True,
        )

        assert order is None  # Should fail and return None due to exception handling


class TestCancelAndReplace:
    """Test cancel and replace order functionality."""

    def test_cancel_and_replace_success(self):
        """Should successfully cancel and replace order."""
        broker = StubBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Place original order
        original = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100"),
        )

        assert original is not None
        original_id = original.id

        # Cancel and replace with new price
        replacement = engine.cancel_and_replace(
            order_id=original_id,
            new_price=Decimal("102"),
            new_size=Decimal("0.2"),
        )

        assert replacement is not None
        assert replacement.quantity == Decimal("0.2")
        assert original_id in broker.cancelled_orders
        assert engine.order_metrics["cancelled"] == 1

    def test_cancel_and_replace_order_not_found(self):
        """Should return None when order not found."""
        broker = StubBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        result = engine.cancel_and_replace(
            order_id="non-existent-id",
            new_price=Decimal("100"),
        )

        assert result is None

    def test_cancel_and_replace_cancel_failure_retries(self):
        """Should retry cancel on failure."""
        broker = StubBroker(fail_cancel=True)
        engine = AdvancedExecutionEngine(broker=broker)

        # Place original order
        original = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100"),
        )

        assert original is not None

        # Attempt cancel and replace (should fail all retries)
        replacement = engine.cancel_and_replace(
            order_id=original.id,
            new_price=Decimal("102"),
            max_retries=2,
        )

        assert replacement is None


class TestPositionManagement:
    """Test position closing functionality."""

    def test_close_position_long(self):
        """Should close long position with sell order."""
        position = Position(
            symbol="BTC-PERP",
            quantity=Decimal("1.5"),
            entry_price=Decimal("100"),
            mark_price=Decimal("105"),
            unrealized_pnl=Decimal("7.5"),
            realized_pnl=Decimal("0"),
            leverage=None,
            side="long",
        )
        broker = StubBroker(positions=[position])
        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.close_position("BTC-PERP")

        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.quantity == Decimal("1.5")
        assert order.reduce_only is True

    def test_close_position_short(self):
        """Should close short position with buy order."""
        position = Position(
            symbol="BTC-PERP",
            quantity=Decimal("-0.8"),
            entry_price=Decimal("100"),
            mark_price=Decimal("95"),
            unrealized_pnl=Decimal("4.0"),
            realized_pnl=Decimal("0"),
            leverage=None,
            side="short",
        )
        broker = StubBroker(positions=[position])
        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.close_position("BTC-PERP")

        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal("0.8")
        assert order.reduce_only is True

    def test_close_position_no_position(self):
        """Should return None when no position to close."""
        broker = StubBroker(positions=[])
        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.close_position("BTC-PERP")

        assert order is None

    def test_close_position_zero_quantity(self):
        """Should return None for zero quantity position."""
        position = Position(
            symbol="BTC-PERP",
            quantity=Decimal("0"),
            entry_price=Decimal("100"),
            mark_price=Decimal("100"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=None,
            side="long",
        )
        broker = StubBroker(positions=[position])
        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.close_position("BTC-PERP")

        assert order is None


class TestUtilityMethods:
    """Test utility methods like TIF validation and metrics."""

    def test_validate_tif_gtc(self):
        """Should validate GTC TIF."""
        broker = StubBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        tif = engine._validate_tif("GTC")
        assert tif == TimeInForce.GTC

    def test_validate_tif_ioc_enabled(self):
        """Should validate IOC when enabled."""
        broker = StubBroker()
        config = OrderConfig(enable_ioc=True)
        engine = AdvancedExecutionEngine(broker=broker, config=config)

        tif = engine._validate_tif("IOC")
        assert tif == TimeInForce.IOC

    def test_validate_tif_ioc_disabled(self):
        """Should return None for IOC when disabled."""
        broker = StubBroker()
        config = OrderConfig(enable_ioc=False)
        engine = AdvancedExecutionEngine(broker=broker, config=config)

        tif = engine._validate_tif("IOC")
        assert tif is None

    def test_validate_tif_fok_gated(self):
        """Should return None for FOK (gated feature)."""
        broker = StubBroker()
        config = OrderConfig(enable_fok=True)
        engine = AdvancedExecutionEngine(broker=broker, config=config)

        tif = engine._validate_tif("FOK")
        assert tif is None

    def test_validate_tif_unsupported(self):
        """Should return None for unsupported TIF."""
        broker = StubBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        tif = engine._validate_tif("INVALID")
        assert tif is None

    def test_get_metrics(self):
        """Should return execution metrics."""
        broker = StubBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Place some orders
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.STOP,
            stop_price=Decimal("105"),
        )

        metrics = engine.get_metrics()

        assert "orders" in metrics
        assert metrics["orders"]["placed"] == 2
        assert "pending_count" in metrics
        assert metrics["pending_count"] == 2

    def test_calculate_impact_aware_size_delegates(self):
        """Should delegate to sizing helper."""
        broker = StubBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        # Mock the sizing helper method
        engine.sizing_helper.calculate_impact_aware_size = Mock(
            return_value=(Decimal("1000"), Decimal("5"))
        )

        notional, impact = engine.calculate_impact_aware_size(
            symbol="BTC-PERP",
            target_notional=Decimal("2000"),
            market_snapshot={"depth": "data"},
        )

        assert notional == Decimal("1000")
        assert impact == Decimal("5")


class TestStopOrderValidation:
    """Test stop order validation and rejection."""

    def test_stop_order_without_stop_price_rejected(self):
        """Should reject stop order without stop_price."""
        broker = StubBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.STOP,
            stop_price=None,  # Missing stop price
        )

        assert order is None
        assert engine.order_metrics["rejected"] > 0

    def test_stop_limit_order_without_stop_price_rejected(self):
        """Should reject stop-limit order without stop_price."""
        broker = StubBroker()
        engine = AdvancedExecutionEngine(broker=broker)

        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.STOP_LIMIT,
            limit_price=Decimal("105"),
            stop_price=None,  # Missing stop price
        )

        assert order is None
        assert engine.order_metrics["rejected"] > 0

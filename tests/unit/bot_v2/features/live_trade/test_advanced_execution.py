"""Stress and failure handling tests for AdvancedExecutionEngine."""

from datetime import datetime
from decimal import Decimal

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import (
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Product,
    Quote,
    TimeInForce,
)
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.features.live_trade.risk import PositionSizingAdvice


class StubBroker:
    def __init__(self, fail_order_types: set[str] | None = None):
        self.fail_order_types = fail_order_types or set()
        self.orders = []

    def get_product(self, symbol: str) -> Product:
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
        return Quote(
            symbol=symbol,
            bid=Decimal("100"),
            ask=Decimal("101"),
            last=Decimal("100.5"),
            ts=datetime.utcnow(),
        )

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
            price=kwargs.get("price"),
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

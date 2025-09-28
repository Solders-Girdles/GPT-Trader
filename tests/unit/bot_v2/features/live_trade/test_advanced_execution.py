"""Stress and failure handling tests for AdvancedExecutionEngine."""

from datetime import datetime
from decimal import Decimal
from typing import Optional

import pytest

from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.features.brokerages.core.interfaces import (
    Product,
    MarketType,
    OrderType,
    OrderSide,
    TimeInForce,
    Order,
    OrderStatus,
    Quote,
)


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
        self.orders.append(order)
        return order


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

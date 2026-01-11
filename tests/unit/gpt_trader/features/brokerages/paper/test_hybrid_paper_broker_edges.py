from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from gpt_trader.core import OrderSide, OrderType, Position
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


@pytest.fixture
def broker() -> HybridPaperBroker:
    with patch("gpt_trader.features.brokerages.paper.hybrid.CoinbaseClient"):
        with patch("gpt_trader.features.brokerages.paper.hybrid.SimpleAuth"):
            broker = HybridPaperBroker(
                api_key="test_key",
                private_key="test_private_key",
                slippage_bps=10,
            )
            broker._client = Mock()
            return broker


def test_limit_buy_uses_limit_below_slippage(broker: HybridPaperBroker) -> None:
    broker._last_prices["BTC-USD"] = Decimal("100")

    order = broker.place_order(
        symbol_or_payload="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1"),
        limit_price=Decimal("99"),
    )

    assert order.avg_fill_price == Decimal("99")


def test_limit_sell_uses_limit_above_slippage(broker: HybridPaperBroker) -> None:
    broker._last_prices["BTC-USD"] = Decimal("100")

    order = broker.place_order(
        symbol_or_payload="BTC-USD",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1"),
        limit_price=Decimal("101"),
    )

    assert order.avg_fill_price == Decimal("101")


def test_quote_size_payload_computes_quantity(broker: HybridPaperBroker) -> None:
    broker._last_prices["ETH-USD"] = Decimal("200")

    order = broker.place_order(
        symbol_or_payload={
            "product_id": "ETH-USD",
            "side": "BUY",
            "order_configuration": {"market_market_ioc": {"quote_size": "100"}},
        }
    )

    assert order.filled_quantity == Decimal("0.5")


def test_short_position_add_recalculates_average(broker: HybridPaperBroker) -> None:
    broker._positions["BTC-USD"] = Position(
        symbol="BTC-USD",
        quantity=Decimal("-1"),
        entry_price=Decimal("100"),
        mark_price=Decimal("100"),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
        side="short",
        leverage=1,
    )

    broker._update_position("BTC-USD", OrderSide.SELL, Decimal("1"), Decimal("120"))

    pos = broker._positions["BTC-USD"]
    assert pos.quantity == Decimal("-2")
    assert pos.entry_price == Decimal("110")


def test_short_position_reduce_keeps_entry_price(broker: HybridPaperBroker) -> None:
    broker._positions["BTC-USD"] = Position(
        symbol="BTC-USD",
        quantity=Decimal("-1"),
        entry_price=Decimal("100"),
        mark_price=Decimal("100"),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
        side="short",
        leverage=1,
    )

    broker._update_position("BTC-USD", OrderSide.BUY, Decimal("0.5"), Decimal("90"))

    pos = broker._positions["BTC-USD"]
    assert pos.quantity == Decimal("-0.5")
    assert pos.entry_price == Decimal("100")


def test_synthetic_product_default_quote_asset(broker: HybridPaperBroker) -> None:
    product = broker._synthetic_product("BTC")

    assert product.base_asset == "BTC"
    assert product.quote_asset == "USD"

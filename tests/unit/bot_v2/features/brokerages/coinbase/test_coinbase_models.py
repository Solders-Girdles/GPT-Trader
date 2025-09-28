"""Consolidated Coinbase model conversion tests."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from bot_v2.features.brokerages.coinbase.models import to_candle, to_order, to_product, to_quote
from bot_v2.features.brokerages.core.interfaces import MarketType, OrderSide, OrderType, TimeInForce


def test_to_product_handles_spot_payload():
    payload = {
        "id": "BTC-USD",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "base_min_size": "0.0001",
        "base_increment": "0.00000001",
        "quote_increment": "0.01",
    }
    product = to_product(payload)
    assert product.symbol == "BTC-USD"
    assert product.base_asset == "BTC"
    assert product.market_type == MarketType.SPOT
    assert product.min_size == Decimal("0.0001")


def test_to_order_mapping():
    payload = {
        "id": "o1",
        "product_id": "ETH-USD",
        "side": "buy",
        "type": "limit",
        "size": "1.5",
        "price": "2000.00",
        "time_in_force": "gtc",
        "status": "open",
        "filled_size": "0",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:01",
    }
    order = to_order(payload)
    assert order.id == "o1"
    assert order.symbol == "ETH-USD"
    assert order.side == OrderSide.BUY
    assert order.type == OrderType.LIMIT
    assert order.price == Decimal("2000.00")
    assert order.tif == TimeInForce.GTC


def test_to_quote_and_to_candle():
    quote = to_quote(
        {
            "product_id": "BTC-USD",
            "best_bid": "100.0",
            "best_ask": "101.0",
            "price": "100.5",
            "time": datetime(2024, 1, 1, 0, 0, 0).isoformat(),
        }
    )
    assert quote.symbol == "BTC-USD"
    assert quote.bid == Decimal("100.0")
    assert quote.last == Decimal("100.5")

    candle = to_candle(
        {
            "time": datetime(2024, 1, 1, 0, 0, 0).isoformat(),
            "open": "1",
            "high": "2",
            "low": "0.5",
            "close": "1.5",
            "volume": "10",
        }
    )
    assert candle.open == Decimal("1")
    assert candle.close == Decimal("1.5")

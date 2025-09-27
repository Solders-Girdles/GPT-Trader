from datetime import datetime
from decimal import Decimal

from bot_v2.features.brokerages.coinbase.client import CoinbaseAuth
from bot_v2.features.brokerages.coinbase.models import to_product, to_order, to_quote, to_candle
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce, MarketType


def test_sign_headers_deterministic(monkeypatch):
    # Freeze time to ensure deterministic signature
    monkeypatch.setattr("time.time", lambda: 1_700_000_000)
    auth = CoinbaseAuth(api_key="k", api_secret="s", passphrase="p")
    headers = auth.sign("POST", "/api/v3/brokerage/orders", {"a": 1, "b": 2})
    assert headers["CB-ACCESS-KEY"] == "k"
    assert headers["CB-ACCESS-PASSPHRASE"] == "p"
    assert headers["CB-ACCESS-TIMESTAMP"] == str(1_700_000_000)
    # Signature is base64-encoded HMAC; just ensure present and non-empty
    assert isinstance(headers["CB-ACCESS-SIGN"], str) and len(headers["CB-ACCESS-SIGN"]) > 10
    assert headers["Content-Type"] == "application/json"


def test_to_product_spot_minimal():
    payload = {
        "id": "BTC-USD",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "base_min_size": "0.0001",
        "base_increment": "0.00000001",
        "quote_increment": "0.01",
    }
    p = to_product(payload)
    assert p.symbol == "BTC-USD"
    assert p.base_asset == "BTC"
    assert p.quote_asset == "USD"
    assert p.market_type == MarketType.SPOT
    assert p.min_size == Decimal("0.0001")


def test_to_order_mapping_basic():
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
    o = to_order(payload)
    assert o.id == "o1"
    assert o.symbol == "ETH-USD"
    assert o.side == OrderSide.BUY
    assert o.type == OrderType.LIMIT
    assert o.price == Decimal("2000.00")
    assert o.tif == TimeInForce.GTC


def test_to_quote_and_candle():
    q = to_quote({
        "product_id": "BTC-USD",
        "best_bid": "100.0",
        "best_ask": "101.0",
        "price": "100.5",
        "time": "2024-01-01T00:00:00",
    })
    assert q.symbol == "BTC-USD"
    assert q.bid == Decimal("100.0")
    assert q.ask == Decimal("101.0")
    assert q.last == Decimal("100.5")

    c = to_candle({
        "time": "2024-01-01T00:00:00",
        "open": "1",
        "high": "2",
        "low": "0.5",
        "close": "1.5",
        "volume": "10",
    })
    assert c.open == Decimal("1")
    assert c.close == Decimal("1.5")

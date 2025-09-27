from decimal import Decimal

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce
from bot_v2.features.brokerages.coinbase import models
from bot_v2.features.brokerages.coinbase import adapter as adapter_mod
from bot_v2.features.brokerages.coinbase import client as client_mod
from bot_v2.features.brokerages.core.interfaces import MarketType
from bot_v2.features.brokerages.coinbase.utils import quantize_to_increment
from bot_v2.features.brokerages.coinbase.models import to_product
from bot_v2.features.brokerages.coinbase import utils as utils_mod
import pytest
from bot_v2.errors import ValidationError


def make_broker():
    cfg = APIConfig(api_key="k", api_secret="s", passphrase=None, base_url="https://api")
    b = CoinbaseBrokerage(cfg)
    return b


def test_quantize_helper():
    assert quantize_to_increment(Decimal("1.2345"), Decimal("0.01")) == Decimal("1.23")
    assert quantize_to_increment(Decimal("0.0009"), Decimal("0.001")) == Decimal("0.000")


def test_place_order_applies_rounding_and_builds_payload(monkeypatch):
    broker = make_broker()

    # Mock products
    product_payload = {
        "id": "BTC-USD",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "base_min_size": "0.001",
        "base_increment": "0.001",
        "quote_increment": "0.1",
        # Keep min_notional small to avoid blocking this test case
        "min_notional": "0.01",
        "contract_type": None,
    }

    def fake_get_products():
        return {"products": [product_payload]}

    monkeypatch.setattr(client_mod.CoinbaseClient, "get_products", lambda self: fake_get_products())

    captured = {}

    def fake_place_order(self, payload):
        captured.update(payload)
        # Return minimal order response compatible with to_order
        return {
            "id": "oid",
            "product_id": payload["product_id"],
            "side": payload["side"],
            "type": "limit",
            "size": payload.get("order_configuration", {}).get("limit_limit_gtc", {}).get("base_size", "0"),
            "price": payload.get("order_configuration", {}).get("limit_limit_gtc", {}).get("limit_price"),
            "status": "open",
            "filled_size": "0",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

    monkeypatch.setattr(client_mod.CoinbaseClient, "place_order", fake_place_order)

    # Qty will be floored to 0.001, price to 100.0
    o = broker.place_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.0014"),
        price=Decimal("100.05"),
        tif=TimeInForce.GTC,
    )
    cfg = captured["order_configuration"]["limit_limit_gtc"]
    assert cfg["base_size"] == "0.001"
    assert cfg["limit_price"] == "100.0"
    assert captured["product_id"] == "BTC-USD"


def test_min_notional_enforced(monkeypatch):
    broker = make_broker()

    product_payload = {
        "id": "BTC-USD",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "base_min_size": "0.001",
        "base_increment": "0.001",
        "quote_increment": "0.1",
        "min_notional": "50",
    }
    monkeypatch.setattr(client_mod.CoinbaseClient, "get_products", lambda self: {"products": [product_payload]})

    with pytest.raises(ValidationError):
        broker.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=Decimal("0.001"),
            price=Decimal("10.0"),  # qty * price = 0.01 < min_notional 50
            tif=TimeInForce.GTC,
        )

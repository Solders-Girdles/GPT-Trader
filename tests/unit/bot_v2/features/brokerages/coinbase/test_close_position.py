from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.core.interfaces import Position, MarketType, Product, OrderSide, OrderType
from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError


def _adapter_with_product():
    cfg = APIConfig(
        api_key="k",
        api_secret="s",
        passphrase=None,
        base_url="https://api.coinbase.com",
        api_mode="advanced",
        sandbox=False,
        enable_derivatives=True,
        auth_type="HMAC",
    )
    a = CoinbaseBrokerage(cfg)
    class Cat:
        def get(self, client, symbol):
            return Product(
                symbol=symbol,
                base_asset="BTC",
                quote_asset="USD",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=None,
                price_increment=Decimal("0.01"),
            )
        def get_funding(self, client, symbol):
            return None, None
    a.product_catalog = Cat()
    return a


def test_close_position_uses_endpoint(monkeypatch):
    a = _adapter_with_product()

    # Mock list_positions to return a long position
    a.list_positions = lambda: [
        Position(symbol="BTC-USD-PERP", qty=Decimal("1.0"), entry_price=Decimal("50000"), mark_price=Decimal("50500"),
                 unrealized_pnl=Decimal("500"), realized_pnl=Decimal("0"), leverage=3, side="long")
    ]  # type: ignore

    payloads = {}
    def close_position(payload):
        payloads['cp'] = payload
        return {"order_id": "cp123", "status": "open", "product_id": payload.get('product_id'), "type": "market", "side": payload.get('side').lower(), "time_in_force": "ioc"}
    a.client.close_position = close_position  # type: ignore[attr-defined]

    order = a.close_position("BTC-USD-PERP")
    assert order.id == "cp123"
    assert payloads['cp']['product_id'] == "BTC-USD-PERP"
    assert payloads['cp']['size'] == "1.0"
    assert payloads['cp']['reduce_only'] is True


def test_close_position_fallback_to_market(monkeypatch):
    a = _adapter_with_product()
    a.list_positions = lambda: [
        Position(symbol="BTC-USD-PERP", qty=Decimal("-2.0"), entry_price=Decimal("30000"), mark_price=Decimal("29500"),
                 unrealized_pnl=Decimal("-1000"), realized_pnl=Decimal("0"), leverage=2, side="short")
    ]  # type: ignore

    def close_position(payload):
        raise InvalidRequestError("endpoint not available")
    a.client.close_position = close_position  # type: ignore[attr-defined]

    called = {}
    def place_order(**kwargs):
        from types import SimpleNamespace
        called.update(kwargs)
        return SimpleNamespace(id="fallback123")
    # Monkeypatch adapter place_order
    from types import MethodType
    a.place_order = MethodType(lambda self, **kw: place_order(**kw), a)

    order_id = a.close_position("BTC-USD-PERP").id
    assert order_id == "fallback123"
    assert called['reduce_only'] is True
    assert called['order_type'] == OrderType.MARKET
    assert called['side'] == OrderSide.BUY

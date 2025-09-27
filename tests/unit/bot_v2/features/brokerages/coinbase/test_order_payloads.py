from __future__ import annotations

import os
from decimal import Decimal
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.core.interfaces import MarketType, Product, OrderType, OrderSide, TimeInForce
from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError


def _make_adapter():
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
    # Stub product catalog to avoid network
    class ProdCat:
        def get(self, client, symbol):
            return Product(
                symbol=symbol,
                base_asset=symbol.split('-')[0],
                quote_asset="USD",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=None,
                price_increment=Decimal("0.01"),
            )
        def get_funding(self, client, symbol):
            return None, None
    a.product_catalog = ProdCat()
    return a


def test_payload_mapping_market_ioc(monkeypatch):
    a = _make_adapter()
    # Enable preview gating
    monkeypatch.setenv("ORDER_PREVIEW_ENABLED", "1")
    # Capture payloads
    captured = {}
    def prev(payload):
        captured['preview'] = payload
        return {"ok": True}
    def place(payload):
        captured['place'] = payload
        return {"order_id": "abc", "status": "open", "product_id": payload.get('product_id'), "type": "market", "side": payload.get('side').lower(), "time_in_force": "ioc", "client_order_id": payload.get('client_order_id'), "size": payload['order_configuration']['market_market_ioc'].get('base_size')}
    a.client.preview_order = prev  # type: ignore[attr-defined]
    a.client.place_order = place  # type: ignore[attr-defined]

    order = a.place_order(
        symbol="BTC-USD-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.1234"),
        tif=TimeInForce.IOC,
    )
    # Verify mapping
    assert 'preview' in captured and 'place' in captured
    p = captured['place']
    assert p['order_configuration']['market_market_ioc']['base_size'] == '0.123'  # quantized to step
    assert p['side'] == 'BUY'
    assert 'client_order_id' in p and p['client_order_id']
    assert order.type == OrderType.MARKET


def test_payload_mapping_limit_tifs(monkeypatch):
    a = _make_adapter()
    monkeypatch.setenv("ORDER_PREVIEW_ENABLED", "1")
    captured = {}
    a.client.preview_order = lambda payload: {"ok": True}  # type: ignore[attr-defined]
    def place(payload):
        captured['place'] = payload
        return {"order_id": "xyz", "status": "open", "product_id": payload.get('product_id'), "type": "limit", "side": payload.get('side').lower(), "time_in_force": "gtc", "price": payload['order_configuration']['limit_limit_gtc'].get('limit_price'), "client_order_id": payload.get('client_order_id')}
    a.client.place_order = place  # type: ignore[attr-defined]

    _ = a.place_order(
        symbol="ETH-USD-PERP",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        qty=Decimal("1.0"),
        price=Decimal("2500.123"),
        tif=TimeInForce.GTC,
    )
    p = captured['place']
    assert 'limit_limit_gtc' in p['order_configuration']
    assert p['order_configuration']['limit_limit_gtc']['limit_price'] == '2500.12'

    # IOC mapping
    def place_ioc(payload):
        captured['place_ioc'] = payload
        return {"order_id": "ioc", "status": "open", "product_id": payload.get('product_id'), "type": "limit", "side": payload.get('side').lower(), "time_in_force": "ioc", "price": payload['order_configuration']['limit_limit_ioc'].get('limit_price'), "client_order_id": payload.get('client_order_id')}
    a.client.place_order = place_ioc  # type: ignore[attr-defined]
    _ = a.place_order(
        symbol="ETH-USD-PERP",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        qty=Decimal("1.0"),
        price=Decimal("2500.00"),
        tif=TimeInForce.IOC,
    )
    assert 'limit_limit_ioc' in captured['place_ioc']['order_configuration']

    # FOK mapping
    def place_fok(payload):
        captured['place_fok'] = payload
        return {"order_id": "fok", "status": "open", "product_id": payload.get('product_id'), "type": "limit", "side": payload.get('side').lower(), "time_in_force": "fok", "price": payload['order_configuration']['limit_limit_fok'].get('limit_price'), "client_order_id": payload.get('client_order_id')}
    a.client.place_order = place_fok  # type: ignore[attr-defined]
    _ = a.place_order(
        symbol="ETH-USD-PERP",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        qty=Decimal("1.0"),
        price=Decimal("2500.00"),
        tif=TimeInForce.FOK,
    )
    assert 'limit_limit_fok' in captured['place_fok']['order_configuration']


def test_duplicate_client_order_id_resolution(monkeypatch):
    a = _make_adapter()
    # Disable preview
    monkeypatch.delenv("ORDER_PREVIEW_ENABLED", raising=False)
    # Raise duplicate on place_order; resolve via list_orders
    def place(payload):
        raise InvalidRequestError("duplicate client_order_id")
    a.client.place_order = place  # type: ignore[attr-defined]
    # Provide two candidates: an old one (beyond window) and a recent one
    matching = {
        "orders": [
            {"order_id": "old1", "status": "open", "client_order_id": "same123", "product_id": "BTC-USD-PERP", "type": "market", "side": "buy", "time_in_force": "ioc", "created_at": "2020-01-01T00:00:00Z"},
            {"order_id": "1122", "status": "open", "client_order_id": "same123", "product_id": "BTC-USD-PERP", "type": "market", "side": "buy", "time_in_force": "ioc", "created_at": "2099-01-01T00:00:00Z"}
        ]
    }
    def list_orders(**params):
        return matching
    a.client.list_orders = list_orders  # type: ignore[attr-defined]

    # Capture client id to assert matching
    got = None
    # Inject a predictable client id
    order = a.place_order(
        symbol="BTC-USD-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.1"),
        client_id="same123",
    )
    assert order.id == "1122"


def test_preview_order_builds_same_payload(monkeypatch):
    a = _make_adapter()
    captured = {}
    def preview(payload):
        captured['payload'] = payload
        return {'success': True}
    a.client.preview_order = preview  # type: ignore[attr-defined]

    _ = a.preview_order(
        symbol='BTC-USD-PERP',
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal('0.25'),
        tif=TimeInForce.IOC,
    )
    payload = captured['payload']
    assert payload['order_configuration']['market_market_ioc']['base_size'] == '0.250'
    assert payload['side'] == 'BUY'


def test_edit_order_preview_wraps_configuration(monkeypatch):
    a = _make_adapter()
    captured = {}
    def edit_preview(payload):
        captured['payload'] = payload
        return {'preview_id': 'prev123'}
    a.client.edit_order_preview = edit_preview  # type: ignore[attr-defined]

    resp = a.edit_order_preview(
        order_id='order-1',
        symbol='ETH-USD-PERP',
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        qty=Decimal('1.5'),
        price=Decimal('2000'),
        tif=TimeInForce.GTC,
        new_client_id='client-xyz',
        reduce_only=True,
    )
    payload = captured['payload']
    assert payload['order_id'] == 'order-1'
    assert 'order_configuration' in payload
    assert payload['reduce_only'] is True
    assert resp['preview_id'] == 'prev123'


def test_edit_order_returns_order(monkeypatch):
    a = _make_adapter()
    def edit(payload):
        return {
            'order_id': 'ord123',
            'product_id': 'BTC-USD-PERP',
            'client_order_id': payload.get('preview_id'),
            'status': 'open',
            'side': 'buy',
            'type': 'market',
            'time_in_force': 'ioc',
            'size': '0.1',
            'created_at': '2024-01-01T00:00:00Z'
        }
    a.client.edit_order = edit  # type: ignore[attr-defined]
    order = a.edit_order('order-1', 'preview-1')
    assert order.id == 'ord123'
    assert order.client_id == 'preview-1'


def test_payment_methods_and_portfolios_wrappers(monkeypatch):
    a = _make_adapter()
    a.client.list_payment_methods = lambda: {'payment_methods': [{'id': 'pm-1'}]}  # type: ignore[attr-defined]
    a.client.get_payment_method = lambda pm_id: {'payment_method': {'id': pm_id}}  # type: ignore[attr-defined]
    a.client.list_portfolios = lambda: {'portfolios': [{'uuid': 'pf-1'}]}  # type: ignore[attr-defined]
    a.client.get_portfolio = lambda uuid: {'portfolio': {'uuid': uuid}}  # type: ignore[attr-defined]
    a.client.get_portfolio_breakdown = lambda uuid: {'breakdown': {'uuid': uuid}}  # type: ignore[attr-defined]
    a.client.move_funds = lambda payload: {'status': 'ok', **payload}  # type: ignore[attr-defined]

    assert a.list_payment_methods() == [{'id': 'pm-1'}]
    assert a.get_payment_method('pm-9')['id'] == 'pm-9'
    assert a.list_portfolios() == [{'uuid': 'pf-1'}]
    assert a.get_portfolio('pf-1')['uuid'] == 'pf-1'
    assert a.get_portfolio_breakdown('pf-1')['uuid'] == 'pf-1'
    move_resp = a.move_portfolio_funds({'from': 'pf-1', 'to': 'pf-2', 'amount': '10'})
    assert move_resp['status'] == 'ok'


def test_convert_wrappers(monkeypatch):
    a = _make_adapter()
    captured = {}
    a.client.convert_quote = lambda payload: captured.setdefault('quote', payload) or {'quote_id': 'q1'}  # type: ignore[attr-defined]
    a.client.commit_convert_trade = lambda trade_id, payload: {'trade_id': trade_id, **(payload or {})}  # type: ignore[attr-defined]
    a.client.get_convert_trade = lambda trade_id: {'trade_id': trade_id, 'status': 'settled'}  # type: ignore[attr-defined]

    quote = a.create_convert_quote({'from': 'USDC', 'to': 'USD', 'amount': '100'})
    assert captured['quote']['amount'] == '100'
    commit = a.commit_convert_trade('trade-1', {'amount': '100'})
    assert commit['trade_id'] == 'trade-1'
    status = a.get_convert_trade('trade-1')
    assert status['status'] == 'settled'

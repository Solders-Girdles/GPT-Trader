from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timedelta

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.core.interfaces import Product, MarketType


def _adapter(api_mode="advanced"):
    cfg = APIConfig(
        api_key="k",
        api_secret="s",
        passphrase=None,
        base_url="https://api.coinbase.com",
        api_mode=api_mode,
        sandbox=False,
    )
    return CoinbaseBrokerage(cfg)


def test_list_balances_advanced_shape(monkeypatch):
    a = _adapter()
    payload = {
        "accounts": [
            {"currency": "USD", "available_balance": {"value": "100.25"}, "hold": {"value": "10.75"}},
            {"currency": "USDC", "available_balance": {"value": "50"}, "hold": "0"}
        ]
    }
    a.client.get_accounts = lambda: payload  # type: ignore[attr-defined]
    balances = a.list_balances()
    d = {b.asset: b for b in balances}
    assert d['USD'].available == Decimal("100.25")
    assert d['USD'].hold == Decimal("10.75")
    assert d['USD'].total == Decimal("111.00")
    assert d['USDC'].total == Decimal("50")


def test_list_balances_exchange_shape(monkeypatch):
    a = _adapter(api_mode="exchange")
    payload = [
        {"currency": "USD", "balance": "200", "available": "150", "hold": "50"},
        {"currency": "BTC", "balance": "0.1", "available": "0.1", "hold": "0"}
    ]
    a.client.get_accounts = lambda: payload  # type: ignore[attr-defined]
    balances = a.list_balances()
    d = {b.asset: b for b in balances}
    assert d['USD'].total == Decimal("200")
    assert d['USD'].available == Decimal("150")
    assert d['USD'].hold == Decimal("50")


def test_funding_enrichment_uses_product_catalog(monkeypatch):
    a = _adapter()

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
            return Decimal("0.0005"), datetime.utcnow() + timedelta(hours=8)

    a.product_catalog = Cat()
    # Stub client.get_product to avoid network and mark as perpetual
    a.client.get_product = lambda pid: {
        "product_id": "BTC-USD-PERP",
        "base_increment": "0.001",
        "quote_increment": "0.01",
        "base_min_size": "0.001",
        "contract_type": "perpetual",
    }  # type: ignore[attr-defined]
    p = a.get_product("BTC-USD-PERP")
    assert p.funding_rate == Decimal("0.0005")
    assert isinstance(p.next_funding_time, datetime)

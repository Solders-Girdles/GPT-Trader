"""Coinbase account balance and product enrichment tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.core import MarketType, Product
from gpt_trader.utilities.datetime_helpers import utc_now
from tests.unit.gpt_trader.features.brokerages.coinbase.helpers import make_adapter

pytestmark = pytest.mark.endpoints


class TestCoinbaseAccountBalances:
    def test_list_balances_advanced_shape(self) -> None:
        adapter = make_adapter()
        payload = {
            "accounts": [
                {
                    "currency": "USD",
                    "available_balance": {"value": "100.25"},
                    "hold": {"value": "10.75"},
                },
                {"currency": "USDC", "available_balance": {"value": "50"}, "hold": "0"},
            ]
        }
        adapter.client.get_accounts = lambda: payload  # type: ignore[attr-defined]
        balances = adapter.list_balances()
        data = {balance.asset: balance for balance in balances}
        assert data["USD"].available == Decimal("100.25")
        assert data["USD"].hold == Decimal("10.75")
        assert data["USD"].total == Decimal("111.00")
        assert data["USDC"].total == Decimal("50")

    def test_list_balances_exchange_shape(self) -> None:
        adapter = make_adapter(api_mode="exchange")
        payload = [
            {"currency": "USD", "balance": "200", "available": "150", "hold": "50"},
            {"currency": "BTC", "balance": "0.1", "available": "0.1", "hold": "0"},
        ]
        adapter.client.get_accounts = lambda: payload  # type: ignore[attr-defined]
        balances = adapter.list_balances()
        data = {balance.asset: balance for balance in balances}
        assert data["USD"].total == Decimal("200")
        assert data["USD"].available == Decimal("150")
        assert data["USD"].hold == Decimal("50")


class TestCoinbaseProductFundingEnrichment:
    def test_funding_enrichment_uses_product_catalog(self) -> None:
        adapter = make_adapter()

        class Catalog:
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
                    leverage_max=None,
                )

            def get_funding(self, client, symbol):
                return Decimal("0.0005"), utc_now() + timedelta(hours=8)

        adapter.product_catalog = Catalog()
        adapter.client.get_product = lambda pid: {
            "product_id": "BTC-USD-PERP",
            "base_increment": "0.001",
            "quote_increment": "0.01",
            "base_min_size": "0.001",
            "contract_type": "perpetual",
        }  # type: ignore[attr-defined]

        product = adapter.get_product("BTC-USD-PERP")
        assert product.funding_rate == Decimal("0.0005")
        assert isinstance(product.next_funding_time, datetime)

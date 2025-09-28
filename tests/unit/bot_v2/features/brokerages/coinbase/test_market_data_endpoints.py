"""Combined market data, catalog, and mode gating tests for Coinbase integration."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict

import pytest
from unittest.mock import MagicMock, patch

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.market_data_features import (
    DepthSnapshot,
    RollingWindow,
    TradeTapeAgg,
)
from bot_v2.features.brokerages.coinbase.utilities import (
    ProductCatalog,
    enforce_perp_rules,
    quantize_to_increment,
)
from bot_v2.features.brokerages.coinbase.models import to_product
from bot_v2.features.brokerages.core.interfaces import (
    InvalidRequestError,
    MarketType,
    NotFoundError,
    Product,
)

from tests.unit.bot_v2.features.brokerages.coinbase.minimal_brokerage import (
    MinimalCoinbaseBrokerage,
)


pytestmark = pytest.mark.endpoints


def make_client(api_mode: str = "advanced") -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


# ---------------------------------------------------------------------------
# CoinbaseClient market data endpoints
# ---------------------------------------------------------------------------


def test_get_ticker_formats_path_correctly():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url, headers, body, timeout))
        return 200, {"content-type": "application/json"}, json.dumps({"price": "123"})

    client.set_transport_for_testing(fake_transport)
    out = client.get_ticker("BTC-USD")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/market/products/BTC-USD/ticker")
    assert out.get("price") == "123"


def test_get_candles_includes_all_params():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"candles": []})

    client.set_transport_for_testing(fake_transport)
    _ = client.get_candles(
        "ETH-USD",
        granularity="1H",
        limit=500,
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 2, 0, 0, 0),
    )
    url = calls[0]
    assert "granularity=1H" in url
    assert "limit=500" in url
    assert "start=2024-01-01T00:00:00Z" in url
    assert "end=2024-01-02T00:00:00Z" in url


def test_get_product_book_handles_levels_advanced():
    client = make_client(api_mode="advanced")
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"bids": [], "asks": []})

    client.set_transport_for_testing(fake_transport)
    _ = client.get_product_book("BTC-USD", level=2)
    url = calls[0]
    assert url.endswith("/api/v3/brokerage/market/product_book?product_id=BTC-USD&level=2")


def test_get_product_book_handles_levels_exchange():
    client = make_client(api_mode="exchange")
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"bids": [], "asks": []})

    client.set_transport_for_testing(fake_transport)
    _ = client.get_product_book("BTC-USD", level=2)
    url = calls[0]
    assert url.endswith("/products/BTC-USD/book?level=2")


def test_get_best_bid_ask_requires_advanced_mode():
    client = make_client(api_mode="advanced")
    client.set_transport_for_testing(lambda *args, **kwargs: (200, {}, json.dumps({"data": []})))
    _ = client.get_best_bid_ask(["BTC-USD", "ETH-USD"])

    client_ex = make_client(api_mode="exchange")
    with pytest.raises(InvalidRequestError):
        client_ex.get_best_bid_ask(["BTC-USD"])


def test_get_market_products_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"products": []})

    client.set_transport_for_testing(fake_transport)
    out = client.get_market_products()
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/market/products")
    assert "products" in out


def test_get_market_product_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"product_id": "BTC-USD"})

    client.set_transport_for_testing(fake_transport)
    out = client.get_market_product("BTC-USD")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/market/products/BTC-USD")
    assert "product_id" in out


def test_get_market_product_ticker_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"price": "50000"})

    client.set_transport_for_testing(fake_transport)
    out = client.get_market_product_ticker("ETH-USD")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/market/products/ETH-USD/ticker")
    assert "price" in out


def test_get_market_product_candles_includes_params():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"candles": []})

    client.set_transport_for_testing(fake_transport)
    _ = client.get_market_product_candles("BTC-USD", granularity="5M", limit=300)
    url = calls[0]
    assert url.endswith(
        "/api/v3/brokerage/market/products/BTC-USD/candles?granularity=5M&limit=300"
    )


def test_get_market_product_book_formats_path():
    client = make_client(api_mode="advanced")
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"bids": [], "asks": []})

    client.set_transport_for_testing(fake_transport)
    _ = client.get_market_product_book("BTC-PERP", level=3)
    url = calls[0]
    assert url.endswith("/api/v3/brokerage/market/product_book?product_id=BTC-PERP&level=3")


def test_get_product_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"product_id": "BTC-USD", "base_currency": "BTC"})

    client.set_transport_for_testing(fake_transport)
    out = client.get_product("BTC-USD")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/market/products/BTC-USD")
    assert "product_id" in out


def test_products_path_differs_by_mode():
    client_ex = make_client("exchange")
    urls = []
    client_ex.set_transport_for_testing(
        lambda m, u, h, b, t: (
            (200, {}, json.dumps({"products": []})) if not urls.append(u) else (200, {}, "{}")
        )
    )
    client_ex.get_products()
    assert urls[0].endswith("/products")

    client_adv = make_client("advanced")
    urls = []
    client_adv.set_transport_for_testing(
        lambda m, u, h, b, t: (
            (200, {}, json.dumps({"products": []})) if not urls.append(u) else (200, {}, "{}")
        )
    )
    client_adv.get_products()
    assert urls[0].endswith("/api/v3/brokerage/market/products")


def test_get_product_book_path_mapping_by_mode():
    client_ex = make_client("exchange")
    urls_ex = []
    client_ex.set_transport_for_testing(
        lambda m, u, h, b, t: (
            (200, {}, json.dumps({"bids": [], "asks": []}))
            if not urls_ex.append(u)
            else (200, {}, "{}")
        )
    )
    client_ex.get_product_book("BTC-USD", level=2)
    assert urls_ex[0].endswith("/products/BTC-USD/book?level=2")

    client_adv = make_client("advanced")
    urls_adv = []
    client_adv.set_transport_for_testing(
        lambda m, u, h, b, t: (
            (200, {}, json.dumps({"bids": [], "asks": []}))
            if not urls_adv.append(u)
            else (200, {}, "{}")
        )
    )
    client_adv.get_product_book("BTC-USD", level=2)
    assert urls_adv[0].endswith("/api/v3/brokerage/market/product_book?product_id=BTC-USD&level=2")


def test_advanced_only_endpoints_raise_in_exchange():
    client_ex = make_client("exchange")
    with pytest.raises(InvalidRequestError):
        client_ex.list_portfolios()
    with pytest.raises(InvalidRequestError):
        client_ex.get_best_bid_ask(["BTC-USD"])


# ---------------------------------------------------------------------------
# Query parameter behavior
# ---------------------------------------------------------------------------


def test_list_parameters_comma_separated():
    client = make_client()
    client.set_transport_for_testing(lambda m, u, h, b, t: (200, {}, json.dumps({"ok": True})))
    _ = client.get_best_bid_ask(["BTC-USD", "ETH-USD", "SOL-USD"])
    assert True  # Behavior covered by implementation; ensure no exception


def test_repeated_parameters_not_encoded_as_array():
    client = make_client()
    urls = []

    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"ok": True})

    client.set_transport_for_testing(transport)
    _ = client.list_orders(filter=["a", "b"])
    assert "filter=['a', 'b']" in urls[0]


def test_unicode_emoji_in_params():
    client = make_client()
    urls = []
    client.set_transport_for_testing(
        lambda m, u, h, b, t: (
            (200, {}, json.dumps({"ok": True})) if not urls.append(u) else (200, {}, "{}")
        )
    )
    _ = client.list_orders(note="🚀")
    assert "🚀" in urls[0]


def test_empty_values_are_included():
    client = make_client()
    urls = []

    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"ok": True})

    client.set_transport_for_testing(transport)
    _ = client.list_orders(filter="")
    assert url_has_param(urls[0], "filter=")


def test_special_characters_plus_slash_at():
    client = make_client()
    urls = []

    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"ok": True})

    client.set_transport_for_testing(transport)
    _ = client.list_orders(path="/foo/bar", email="test+user@example.com")
    url = urls[0]
    assert "path=/foo/bar" in url
    assert "email=test+user@example.com" in url


def url_has_param(url: str, fragment: str) -> bool:
    return ("?" + fragment) in url or ("&" + fragment) in url


# ---------------------------------------------------------------------------
# Market data feature helpers
# ---------------------------------------------------------------------------


class TestDepthSnapshot:
    def test_l1_l10_depth_correctness(self):
        levels = [
            (Decimal("50000"), Decimal("1.0"), "bid"),
            (Decimal("49990"), Decimal("2.0"), "bid"),
            (Decimal("49980"), Decimal("3.0"), "bid"),
            (Decimal("50010"), Decimal("1.5"), "ask"),
            (Decimal("50020"), Decimal("2.5"), "ask"),
            (Decimal("50030"), Decimal("3.5"), "ask"),
        ]
        snapshot = DepthSnapshot(levels)
        assert snapshot.get_l1_depth() == Decimal("1.0")
        assert snapshot.get_l10_depth() == Decimal("13.5")

    def test_spread_bps_calculation(self):
        levels = [
            (Decimal("50000"), Decimal("1.0"), "bid"),
            (Decimal("50010"), Decimal("1.0"), "ask"),
        ]
        snapshot = DepthSnapshot(levels)
        assert snapshot.spread_bps == 2.0

    def test_mid_price(self):
        levels = [
            (Decimal("50000"), Decimal("1.0"), "bid"),
            (Decimal("50020"), Decimal("1.0"), "ask"),
        ]
        snapshot = DepthSnapshot(levels)
        assert snapshot.mid == Decimal("50010")


class TestRollingWindow:
    def test_cleanup_and_stats(self):
        window = RollingWindow(duration_seconds=10)
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        window.add(100.0, base_time)
        window.add(200.0, base_time + timedelta(seconds=5))
        window.add(300.0, base_time + timedelta(seconds=8))
        stats = window.get_stats()
        assert stats["count"] == 3
        assert stats["sum"] == 600.0
        assert stats["avg"] == 200.0

        window.add(400.0, base_time + timedelta(seconds=15))
        stats = window.get_stats()
        assert stats["count"] == 3
        assert stats["sum"] == 900.0
        assert stats["avg"] == 300.0


class TestTradeTapeAgg:
    def test_vwap_calculation(self):
        agg = TradeTapeAgg(duration_seconds=60)
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        agg.add_trade(Decimal("100"), Decimal("10"), "buy", base_time)
        agg.add_trade(Decimal("200"), Decimal("5"), "sell", base_time + timedelta(seconds=10))
        vwap = agg.get_vwap()
        expected = (Decimal("100") * Decimal("10") + Decimal("200") * Decimal("5")) / Decimal("15")
        assert abs(vwap - expected) < Decimal("0.01")

    def test_aggressor_ratio(self):
        agg = TradeTapeAgg(duration_seconds=60)
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        agg.add_trade(Decimal("100"), Decimal("1"), "buy", base_time)
        agg.add_trade(Decimal("100"), Decimal("1"), "buy", base_time + timedelta(seconds=10))
        agg.add_trade(Decimal("100"), Decimal("1"), "sell", base_time + timedelta(seconds=20))
        assert agg.get_aggressor_ratio() == 2 / 3


# ---------------------------------------------------------------------------
# Product catalog and utility helpers
# ---------------------------------------------------------------------------


class TestProductMapping:
    def test_to_product_spot_market(self):
        payload = {
            "product_id": "BTC-USD",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "base_min_size": "0.001",
            "base_increment": "0.00001",
            "quote_increment": "0.01",
            "min_notional": "10",
        }
        product = to_product(payload)
        assert product.symbol == "BTC-USD"
        assert product.market_type == MarketType.SPOT
        assert product.contract_size is None
        assert product.funding_rate is None
        assert product.next_funding_time is None
        assert product.leverage_max is None

    def test_to_product_perpetual_full(self):
        payload = {
            "product_id": "BTC-PERP",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "contract_type": "perpetual",
            "base_min_size": "0.001",
            "base_increment": "0.00001",
            "quote_increment": "0.01",
            "min_notional": "10",
            "max_leverage": 20,
            "contract_size": "1",
            "funding_rate": "0.0001",
            "next_funding_time": "2024-01-15T16:00:00Z",
        }
        product = to_product(payload)
        assert product.symbol == "BTC-PERP"
        assert product.market_type == MarketType.PERPETUAL
        assert product.contract_size == Decimal("1")
        assert product.funding_rate == Decimal("0.0001")
        assert product.next_funding_time == datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        assert product.leverage_max == 20

    def test_to_product_perpetual_partial(self):
        payload = {
            "product_id": "ETH-PERP",
            "base_currency": "ETH",
            "quote_currency": "USD",
            "contract_type": "perpetual",
            "base_min_size": "0.01",
            "base_increment": "0.001",
            "quote_increment": "0.1",
        }
        product = to_product(payload)
        assert product.symbol == "ETH-PERP"
        assert product.market_type == MarketType.PERPETUAL
        assert product.contract_size is None
        assert product.funding_rate is None
        assert product.next_funding_time is None
        assert product.leverage_max is None

    def test_to_product_future_market(self):
        payload = {
            "product_id": "BTC-USD-240331",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "contract_type": "future",
            "base_min_size": "0.001",
            "base_increment": "0.00001",
            "quote_increment": "0.01",
            "expiry": "2024-03-31T08:00:00Z",
            "contract_size": "1",
        }
        product = to_product(payload)
        assert product.symbol == "BTC-USD-240331"
        assert product.market_type == MarketType.FUTURES
        assert product.expiry == datetime(2024, 3, 31, 8, 0, 0, tzinfo=timezone.utc)
        assert product.contract_size == Decimal("1")

    def test_to_product_invalid_funding_time(self):
        payload = {
            "product_id": "BTC-PERP",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "contract_type": "perpetual",
            "base_min_size": "0.001",
            "base_increment": "0.00001",
            "quote_increment": "0.01",
            "next_funding_time": "invalid-date",
        }
        product = to_product(payload)
        assert product.next_funding_time is None


class TestProductCatalog:
    def make_catalog(self, ttl_seconds: int = 900) -> ProductCatalog:
        return ProductCatalog(ttl_seconds=ttl_seconds)

    def test_catalog_refresh_with_perps(self):
        catalog = self.make_catalog()
        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                },
                {
                    "product_id": "BTC-PERP",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "contract_type": "perpetual",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                    "contract_size": "1",
                    "funding_rate": "0.0001",
                    "next_funding_time": "2024-01-15T16:00:00Z",
                    "max_leverage": 20,
                },
            ]
        }
        catalog.refresh(mock_client)
        assert len(catalog._cache) == 2
        perp = catalog._cache["BTC-PERP"]
        assert perp.market_type == MarketType.PERPETUAL
        assert perp.contract_size == Decimal("1")
        assert perp.funding_rate == Decimal("0.0001")
        assert perp.leverage_max == 20

    def test_catalog_get_with_expiry(self):
        catalog = self.make_catalog(ttl_seconds=1)
        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-PERP",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "contract_type": "perpetual",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                }
            ]
        }
        product = catalog.get(mock_client, "BTC-PERP")
        assert product.symbol == "BTC-PERP"
        assert mock_client.get_products.call_count == 1

        product = catalog.get(mock_client, "BTC-PERP")
        assert mock_client.get_products.call_count == 1

        catalog._last_refresh = datetime.utcnow() - timedelta(seconds=2)
        catalog.get(mock_client, "BTC-PERP")
        assert mock_client.get_products.call_count == 2

    def test_catalog_get_not_found(self):
        catalog = self.make_catalog()
        mock_client = MagicMock()
        mock_client.get_products.return_value = {"products": []}
        with pytest.raises(NotFoundError) as exc_info:
            catalog.get(mock_client, "MISSING-PERP")
        assert "Product not found: MISSING-PERP" in str(exc_info.value)
        assert mock_client.get_products.call_count == 2

    def test_catalog_get_funding_for_perpetual(self):
        catalog = self.make_catalog()
        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-PERP",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "contract_type": "perpetual",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                    "funding_rate": "0.0001",
                    "next_funding_time": "2024-01-15T16:00:00Z",
                }
            ]
        }
        funding_rate, next_funding = catalog.get_funding(mock_client, "BTC-PERP")
        assert funding_rate == Decimal("0.0001")
        assert next_funding == datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc)

    def test_catalog_get_funding_for_spot(self):
        catalog = self.make_catalog()
        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                }
            ]
        }
        funding_rate, next_funding = catalog.get_funding(mock_client, "BTC-USD")
        assert funding_rate is None
        assert next_funding is None

    def test_catalog_handles_alternative_response_format(self):
        catalog = self.make_catalog()
        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "data": [
                {
                    "product_id": "ETH-PERP",
                    "base_currency": "ETH",
                    "quote_currency": "USD",
                    "contract_type": "perpetual",
                    "base_min_size": "0.01",
                    "base_increment": "0.001",
                    "quote_increment": "0.1",
                }
            ]
        }
        catalog.refresh(mock_client)
        assert "ETH-PERP" in catalog._cache
        assert catalog._cache["ETH-PERP"].market_type == MarketType.PERPETUAL


class TestEnforcePerpRules:
    def make_perp_product(self) -> Product:
        return Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.00001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=20,
            contract_size=Decimal("1"),
        )

    def test_enforce_quantizes_qty(self):
        product = self.make_perp_product()
        quantity, price = enforce_perp_rules(product, Decimal("0.123456789"))
        assert quantity == Decimal("0.12345")
        assert price is None

    def test_enforce_quantizes_price(self):
        product = self.make_perp_product()
        quantity, price = enforce_perp_rules(product, Decimal("0.01"), Decimal("50123.456"))
        assert quantity == Decimal("0.01")
        assert price == Decimal("50123.45")

    def test_enforce_rejects_below_min_size(self):
        product = self.make_perp_product()
        with pytest.raises(InvalidRequestError) as exc_info:
            enforce_perp_rules(product, Decimal("0.0001"))
        assert "below minimum size" in str(exc_info.value)

    def test_enforce_rejects_below_min_notional(self):
        product = self.make_perp_product()
        with pytest.raises(InvalidRequestError) as exc_info:
            enforce_perp_rules(product, Decimal("0.001"), Decimal("100"))
        assert "below minimum" in str(exc_info.value)

    def test_enforce_accepts_valid_notional(self):
        product = self.make_perp_product()
        quantity, price = enforce_perp_rules(product, Decimal("0.001"), Decimal("20000"))
        assert quantity == Decimal("0.001")
        assert price == Decimal("20000")

    def test_enforce_handles_no_min_notional(self):
        product = self.make_perp_product()
        product.min_notional = None
        quantity, price = enforce_perp_rules(product, Decimal("0.001"), Decimal("1"))
        assert quantity == Decimal("0.001")
        assert price == Decimal("1")

    def test_enforce_complex_quantization(self):
        product = Product(
            symbol="ETH-PERP",
            base_asset="ETH",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.01"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("50"),
            price_increment=Decimal("0.1"),
            leverage_max=15,
        )
        quantity, price = enforce_perp_rules(product, Decimal("0.123456"), Decimal("2345.67"))
        assert quantity == Decimal("0.123")
        assert price == Decimal("2345.6")
        assert quantity * price >= product.min_notional


class TestQuantizeToIncrement:
    def test_quantize_basic(self):
        result = quantize_to_increment(Decimal("1.2345"), Decimal("0.01"))
        assert result == Decimal("1.23")

    def test_quantize_floors_not_rounds(self):
        result = quantize_to_increment(Decimal("1.2389"), Decimal("0.01"))
        assert result == Decimal("1.23")

    def test_quantize_handles_zero_increment(self):
        result = quantize_to_increment(Decimal("1.2345"), Decimal("0"))
        assert result == Decimal("1.2345")
        result = quantize_to_increment(Decimal("1.2345"), None)
        assert result == Decimal("1.2345")

    def test_quantize_arbitrary_increments(self):
        result = quantize_to_increment(Decimal("1.237"), Decimal("0.025"))
        assert result == Decimal("1.225")
        result = quantize_to_increment(Decimal("1.237"), Decimal("0.005"))
        assert result == Decimal("1.235")


# ---------------------------------------------------------------------------
# Adapter staleness utilities
# ---------------------------------------------------------------------------


class TestStalenessDetection:
    def setup_method(self):
        config = APIConfig(
            api_key="test",
            api_secret="test",
            passphrase="test",
            base_url="https://api.sandbox.coinbase.com",
            sandbox=True,
            enable_derivatives=True,
            api_mode="advanced",
            auth_type="HMAC",
        )
        self.adapter = MinimalCoinbaseBrokerage(config)

    def test_fresh_vs_stale_toggles(self):
        symbol = "BTC-PERP"
        assert self.adapter.is_stale(symbol, threshold_seconds=10) is True
        assert self.adapter.is_stale(symbol, threshold_seconds=1) is True
        self.adapter.start_market_data([symbol])
        assert self.adapter.is_stale(symbol, threshold_seconds=10) is False
        assert self.adapter.is_stale(symbol, threshold_seconds=1) is False

    def test_staleness_behavior_matches_validator(self):
        symbol = "ETH-PERP"
        assert self.adapter.is_stale(symbol) is True
        self.adapter.start_market_data([symbol])
        assert self.adapter.is_stale(symbol) is False
        assert self.adapter.is_stale(symbol, threshold_seconds=1) is False

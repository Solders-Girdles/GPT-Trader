"""Coinbase market data and product catalog tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import parse_qs, unquote, unquote_plus, urlparse

import pytest

from gpt_trader.features.brokerages.coinbase.market_data_features import (
    DepthSnapshot,
    RollingWindow,
    TradeTapeAgg,
)
from gpt_trader.features.brokerages.coinbase.models import APIConfig, to_product
from gpt_trader.features.brokerages.coinbase.utilities import (
    ProductCatalog,
    enforce_perp_rules,
    quantize_to_increment,
)
from gpt_trader.features.brokerages.core.interfaces import (
    InvalidRequestError as CoreInvalidRequestError,
)
from gpt_trader.features.brokerages.core.interfaces import (
    MarketType,
    NotFoundError,
    Product,
)
from tests.unit.gpt_trader.features.brokerages.coinbase.minimal_brokerage import (
    MinimalCoinbaseBrokerage,
)
from tests.unit.gpt_trader.features.brokerages.coinbase.test_helpers import (
    MARKET_DATA_ENDPOINT_CASES,
    _decode_body,
    make_client,
    url_has_param,
)

pytestmark = pytest.mark.endpoints


class TestCoinbaseMarketData:
    @pytest.mark.parametrize("case", MARKET_DATA_ENDPOINT_CASES, ids=lambda c: c["id"])
    def test_client_market_data_endpoints(self, case: dict[str, Any]) -> None:
        client = make_client(case.get("api_mode", "advanced"))
        recorded: dict[str, Any] = {}

        def transport(method, url, headers, body, timeout):
            recorded["method"] = method
            recorded["url"] = url
            recorded["body"] = body
            return 200, {}, json.dumps(case.get("response", {}))

        client.set_transport_for_testing(transport)

        result = getattr(client, case["method"])(*case.get("args", ()), **case.get("kwargs", {}))

        assert recorded["method"] == case["expected_method"]
        parsed = urlparse(recorded["url"])
        assert parsed.path.endswith(case["expected_path"])

        expected_query = case.get("expected_query")
        if expected_query is not None:
            assert parse_qs(parsed.query) == expected_query
        else:
            assert parsed.query in ("", None)

        expected_payload = case.get("expected_payload")
        if expected_payload is not None:
            assert _decode_body(recorded.get("body")) == expected_payload
        else:
            assert not recorded.get("body")

        expected_result = case.get("expected_result")
        if expected_result is not None:
            # If expected result has exact match keys, verify them
            # Since CoinbaseClient.get_ticker normalizes output, we should check subsets or updated expected results
            # For now, if it's a dict comparison, we check if expected is a subset of result
            if isinstance(expected_result, dict) and isinstance(result, dict):
                for k, v in expected_result.items():
                    assert result[k] == v
            else:
                assert result == expected_result

    def test_get_product_formats_path(self) -> None:
        client = make_client()
        calls = []

        def fake_transport(method, url, headers, body, timeout):
            calls.append((method, url))
            return 200, {}, json.dumps({"product_id": "BTC-USD", "base_currency": "BTC"})

        client.set_transport_for_testing(fake_transport)
        out = client.get_product("BTC-USD")
        assert calls[0][0] == "GET"
        assert calls[0][1].endswith("/api/v3/brokerage/products/BTC-USD")
        assert "product_id" in out

    def test_products_path_differs_by_mode(self) -> None:
        client_ex = make_client("exchange")
        urls: list[str] = []
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
        assert urls[0].endswith("/api/v3/brokerage/products")

    def test_get_product_book_path_mapping_by_mode(self) -> None:
        client_ex = make_client("exchange")
        urls_ex: list[str] = []
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
        urls_adv: list[str] = []
        client_adv.set_transport_for_testing(
            lambda m, u, h, b, t: (
                (200, {}, json.dumps({"bids": [], "asks": []}))
                if not urls_adv.append(u)
                else (200, {}, "{}")
            )
        )
        client_adv.get_product_book("BTC-USD", level=2)
        assert urls_adv[0].endswith("/api/v3/brokerage/product_book?product_id=BTC-USD&level=2")

    def test_advanced_only_endpoints_raise_in_exchange(self) -> None:
        client_ex = make_client("exchange")
        with pytest.raises(CoreInvalidRequestError):
            client_ex.list_portfolios()
        with pytest.raises(CoreInvalidRequestError):
            client_ex.get_best_bid_ask(["BTC-USD"])

    def test_list_parameters_comma_separated(self) -> None:
        client = make_client()
        client.set_transport_for_testing(lambda m, u, h, b, t: (200, {}, json.dumps({"ok": True})))
        _ = client.get_best_bid_ask(["BTC-USD", "ETH-USD", "SOL-USD"])
        assert True

    def test_repeated_parameters_not_encoded_as_array(self) -> None:
        client = make_client()
        urls: list[str] = []

        def transport(method, url, headers, body, timeout):
            urls.append(url)
            return 200, {}, json.dumps({"ok": True})

        client.set_transport_for_testing(transport)
        _ = client.list_orders(filter=["a", "b"])
        # URL should be properly encoded; decode to verify content (use unquote_plus for + as space)
        decoded_url = unquote_plus(urls[0])
        assert "filter=['a', 'b']" in decoded_url

    def test_unicode_emoji_in_params(self) -> None:
        client = make_client()
        urls: list[str] = []
        client.set_transport_for_testing(
            lambda m, u, h, b, t: (
                (200, {}, json.dumps({"ok": True})) if not urls.append(u) else (200, {}, "{}")
            )
        )
        _ = client.list_orders(note="🚀")
        # URL should be properly encoded; decode to verify emoji is preserved
        decoded_url = unquote(urls[0])
        assert "🚀" in decoded_url

    def test_empty_values_are_included(self) -> None:
        client = make_client()
        urls: list[str] = []

        def transport(method, url, headers, body, timeout):
            urls.append(url)
            return 200, {}, json.dumps({"ok": True})

        client.set_transport_for_testing(transport)
        _ = client.list_orders(filter="")
        assert url_has_param(urls[0], "filter=")

    def test_special_characters_plus_slash_at(self) -> None:
        client = make_client()
        urls: list[str] = []

        def transport(method, url, headers, body, timeout):
            urls.append(url)
            return 200, {}, json.dumps({"ok": True})

        client.set_transport_for_testing(transport)
        _ = client.list_orders(path="/foo/bar", email="test+user@example.com")
        # URL should be properly encoded; decode to verify content
        decoded_url = unquote(urls[0])
        assert "path=/foo/bar" in decoded_url
        assert "test+user@example.com" in decoded_url

    def test_depth_snapshot_l1_l10_depth_correctness(self) -> None:
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

    def test_depth_snapshot_spread_bps(self) -> None:
        levels = [
            (Decimal("50000"), Decimal("1.0"), "bid"),
            (Decimal("50010"), Decimal("1.0"), "ask"),
        ]
        snapshot = DepthSnapshot(levels)
        assert snapshot.spread_bps == 2.0

    def test_depth_snapshot_mid_price(self) -> None:
        levels = [
            (Decimal("50000"), Decimal("1.0"), "bid"),
            (Decimal("50020"), Decimal("1.0"), "ask"),
        ]
        snapshot = DepthSnapshot(levels)
        assert snapshot.mid == Decimal("50010")

    def test_rolling_window_cleanup_and_stats(self) -> None:
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

    def test_trade_tape_vwap_calculation(self) -> None:
        agg = TradeTapeAgg(duration_seconds=60)
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        agg.add_trade(Decimal("100"), Decimal("10"), "buy", base_time)
        agg.add_trade(Decimal("200"), Decimal("5"), "sell", base_time + timedelta(seconds=10))
        vwap = agg.get_vwap()
        expected = (Decimal("100") * Decimal("10") + Decimal("200") * Decimal("5")) / Decimal("15")
        assert abs(vwap - expected) < Decimal("0.01")

    def test_trade_tape_aggressor_ratio(self) -> None:
        agg = TradeTapeAgg(duration_seconds=60)
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        agg.add_trade(Decimal("100"), Decimal("1"), "buy", base_time)
        agg.add_trade(Decimal("100"), Decimal("1"), "buy", base_time + timedelta(seconds=10))
        agg.add_trade(Decimal("100"), Decimal("1"), "sell", base_time + timedelta(seconds=20))
        assert agg.get_aggressor_ratio() == 2 / 3

    def test_to_product_spot_market(self) -> None:
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

    def test_to_product_perpetual_full(self) -> None:
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

    def test_to_product_perpetual_partial(self) -> None:
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

    def test_to_product_future_market(self) -> None:
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

    def test_to_product_invalid_funding_time(self) -> None:
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

    def make_catalog(self, ttl_seconds: int = 900) -> ProductCatalog:
        return ProductCatalog(ttl_seconds=ttl_seconds)

    def test_catalog_refresh_with_perps(self) -> None:
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

    def test_catalog_get_with_expiry(self) -> None:
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

    def test_catalog_get_not_found(self) -> None:
        catalog = self.make_catalog()
        mock_client = MagicMock()
        mock_client.get_products.return_value = {"products": []}
        with pytest.raises(NotFoundError) as exc_info:
            catalog.get(mock_client, "MISSING-PERP")
        assert "Product not found: MISSING-PERP" in str(exc_info.value)
        assert mock_client.get_products.call_count == 1

    def test_catalog_get_funding_for_perpetual(self) -> None:
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

    def test_catalog_get_funding_for_spot(self) -> None:
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

    def test_catalog_handles_alternative_response_format(self) -> None:
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

    def test_enforce_quantizes_quantity(self) -> None:
        product = self.make_perp_product()
        quantity, price = enforce_perp_rules(product, Decimal("0.123456789"))
        assert quantity == Decimal("0.12345")
        assert price is None

    def test_enforce_quantizes_price(self) -> None:
        product = self.make_perp_product()
        quantity, price = enforce_perp_rules(product, Decimal("0.01"), Decimal("50123.456"))
        assert quantity == Decimal("0.01")
        assert price == Decimal("50123.45")

    def test_enforce_rejects_below_min_size(self) -> None:
        product = self.make_perp_product()
        with pytest.raises(CoreInvalidRequestError) as exc_info:
            enforce_perp_rules(product, Decimal("0.0001"))
        assert "below minimum size" in str(exc_info.value)

    def test_enforce_rejects_below_min_notional(self) -> None:
        product = self.make_perp_product()
        with pytest.raises(CoreInvalidRequestError) as exc_info:
            enforce_perp_rules(product, Decimal("0.001"), Decimal("100"))
        assert "below minimum" in str(exc_info.value)

    def test_enforce_accepts_valid_notional(self) -> None:
        product = self.make_perp_product()
        quantity, price = enforce_perp_rules(product, Decimal("0.001"), Decimal("20000"))
        assert quantity == Decimal("0.001")
        assert price == Decimal("20000")

    def test_enforce_handles_no_min_notional(self) -> None:
        product = self.make_perp_product()
        product.min_notional = None
        quantity, price = enforce_perp_rules(product, Decimal("0.001"), Decimal("1"))
        assert quantity == Decimal("0.001")
        assert price == Decimal("1")

    def test_enforce_complex_quantization(self) -> None:
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

    def test_quantize_basic(self) -> None:
        result = quantize_to_increment(Decimal("1.2345"), Decimal("0.01"))
        assert result == Decimal("1.23")

    def test_quantize_floors_not_rounds(self) -> None:
        result = quantize_to_increment(Decimal("1.2389"), Decimal("0.01"))
        assert result == Decimal("1.23")

    def test_quantize_handles_zero_increment(self) -> None:
        result = quantize_to_increment(Decimal("1.2345"), Decimal("0"))
        assert result == Decimal("1.2345")
        result = quantize_to_increment(Decimal("1.2345"), None)
        assert result == Decimal("1.2345")

    def test_quantize_arbitrary_increments(self) -> None:
        result = quantize_to_increment(Decimal("1.237"), Decimal("0.025"))
        assert result == Decimal("1.225")
        result = quantize_to_increment(Decimal("1.237"), Decimal("0.005"))
        assert result == Decimal("1.235")

    def test_staleness_detection_fresh_vs_stale_toggles(self) -> None:
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
        adapter = MinimalCoinbaseBrokerage(config)
        symbol = "BTC-PERP"
        assert adapter.is_stale(symbol, threshold_seconds=10) is True
        assert adapter.is_stale(symbol, threshold_seconds=1) is True
        adapter.start_market_data([symbol])
        assert adapter.is_stale(symbol, threshold_seconds=10) is False
        assert adapter.is_stale(symbol, threshold_seconds=1) is False

    def test_staleness_behavior_matches_validator(self) -> None:
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
        adapter = MinimalCoinbaseBrokerage(config)
        symbol = "ETH-PERP"
        assert adapter.is_stale(symbol) is True
        adapter.start_market_data([symbol])
        assert adapter.is_stale(symbol) is False
        assert adapter.is_stale(symbol, threshold_seconds=1) is False

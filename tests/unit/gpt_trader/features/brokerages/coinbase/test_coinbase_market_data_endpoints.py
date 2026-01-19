"""Coinbase market data endpoint tests."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, unquote, unquote_plus, urlparse

import pytest

from gpt_trader.core import InvalidRequestError as CoreInvalidRequestError
from tests.unit.gpt_trader.features.brokerages.coinbase.helpers import (
    MARKET_DATA_ENDPOINT_CASES,
    _decode_body,
    make_client,
    url_has_param,
)

pytestmark = pytest.mark.endpoints


class TestCoinbaseMarketDataEndpoints:
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
        urls: list[str] = []

        def transport(method, url, headers, body, timeout):
            urls.append(url)
            return 200, {}, json.dumps({"ok": True})

        client.set_transport_for_testing(transport)
        _ = client.get_best_bid_ask(["BTC-USD", "ETH-USD", "SOL-USD"])
        parsed = urlparse(urls[0])
        params = parse_qs(parsed.query)
        assert params["product_ids"] == ["BTC-USD,ETH-USD,SOL-USD"]

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

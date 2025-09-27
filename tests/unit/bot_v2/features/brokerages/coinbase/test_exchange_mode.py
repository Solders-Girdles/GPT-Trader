"""Tests for Exchange mode vs Advanced mode endpoint path mapping and gating."""

import json
import pytest

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.core.interfaces import InvalidRequestError


def make_client(mode: str) -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api-public.sandbox.exchange.coinbase.com" if mode == "exchange" else "https://api.coinbase.com", auth=None, api_mode=mode)


def test_products_path_differs_by_mode():
    # Exchange
    client_ex = make_client("exchange")
    urls = []
    client_ex.set_transport_for_testing(lambda m, u, h, b, t: (200, {}, json.dumps({"products": []})) if not urls.append(u) else (200, {}, '{}'))
    client_ex.get_products()
    assert urls[0].endswith("/products")

    # Advanced
    client_adv = make_client("advanced")
    urls = []
    client_adv.set_transport_for_testing(lambda m, u, h, b, t: (200, {}, json.dumps({"products": []})) if not urls.append(u) else (200, {}, '{}'))
    client_adv.get_products()
    assert urls[0].endswith("/api/v3/brokerage/market/products")


def test_get_product_book_path_mapping():
    # Exchange: level query param only
    client_ex = make_client("exchange")
    urls_ex = []
    client_ex.set_transport_for_testing(lambda m, u, h, b, t: (200, {}, json.dumps({"bids": [], "asks": []})) if not urls_ex.append(u) else (200, {}, '{}'))
    client_ex.get_product_book("BTC-USD", level=2)
    assert urls_ex[0].endswith("/products/BTC-USD/book?level=2")

    # Advanced: includes product_id in query
    client_adv = make_client("advanced")
    urls_adv = []
    client_adv.set_transport_for_testing(lambda m, u, h, b, t: (200, {}, json.dumps({"bids": [], "asks": []})) if not urls_adv.append(u) else (200, {}, '{}'))
    client_adv.get_product_book("BTC-USD", level=2)
    assert urls_adv[0].endswith("/api/v3/brokerage/market/product_book?product_id=BTC-USD&level=2")


def test_advanced_only_endpoints_raise_in_exchange():
    client_ex = make_client("exchange")
    with pytest.raises(InvalidRequestError):
        client_ex.list_portfolios()
    with pytest.raises(InvalidRequestError):
        client_ex.get_best_bid_ask(["BTC-USD"])  # not supported in exchange mode


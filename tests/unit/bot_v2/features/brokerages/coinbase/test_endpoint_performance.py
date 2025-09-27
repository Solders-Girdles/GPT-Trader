"""Lightweight performance baselines for CoinbaseClient.

Opt-in with -m perf; ensures mocked endpoints return within a small budget.
"""

import json
import time
import pytest

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient

pytestmark = pytest.mark.perf


def make_client() -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode="advanced")


def test_get_products_perf_budget():
    client = make_client()
    client.set_transport_for_testing(lambda m, u, h, b, t: (200, {}, json.dumps({"products": []})))
    start = time.perf_counter()
    _ = client.get_products()
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 10  # Mocked path should be very fast


def test_place_order_perf_budget():
    client = make_client()
    client.set_transport_for_testing(lambda m, u, h, b, t: (200, {}, json.dumps({"order_id": "ord"})))
    start = time.perf_counter()
    _ = client.place_order({"product_id": "BTC-USD", "side": "BUY", "size": "0.01"})
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 10


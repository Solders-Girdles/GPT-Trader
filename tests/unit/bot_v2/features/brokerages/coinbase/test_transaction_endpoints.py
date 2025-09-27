"""Unit tests for CoinbaseClient transaction summary endpoint."""

import json
import pytest

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.core.interfaces import InvalidRequestError


pytestmark = pytest.mark.endpoints


def make_client(api_mode: str = "advanced") -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


def test_get_transaction_summary_path_and_method():
    client = make_client("advanced")
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"summary": {"fees": "10", "volume": "1000"}})

    client.set_transport_for_testing(transport)
    out = client.get_transaction_summary()
    method, url = calls[0]
    assert method == "GET"
    assert url.endswith("/api/v3/brokerage/transaction_summary")
    assert "summary" in out


def test_get_transaction_summary_blocked_in_exchange_mode():
    client = make_client("exchange")
    with pytest.raises(InvalidRequestError):
        client.get_transaction_summary()

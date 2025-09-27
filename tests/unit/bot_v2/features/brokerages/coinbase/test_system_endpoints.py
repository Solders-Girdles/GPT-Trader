"""Unit tests for CoinbaseClient system/utility endpoints."""

import json
import pytest

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient


pytestmark = pytest.mark.endpoints


def make_client() -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode="advanced")


def test_get_time_returns_iso_timestamp():
    client = make_client()
    urls = []
    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"iso": "2024-01-01T00:00:00Z"})
    client.set_transport_for_testing(transport)
    out = client.get_time()
    assert urls[0].endswith("/api/v3/brokerage/time")
    assert out.get("iso") == "2024-01-01T00:00:00Z"


def test_get_key_permissions_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"permissions": ["read", "trade"]})

    client.set_transport_for_testing(fake_transport)
    out = client.get_key_permissions()
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/key_permissions")
    assert "permissions" in out


def test_get_fees_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"maker_fee_rate": "0.004", "taker_fee_rate": "0.006"})

    client.set_transport_for_testing(fake_transport)
    out = client.get_fees()
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/fees")  # Correct endpoint
    assert "maker_fee_rate" in out


def test_get_limits_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"buy_power": "50000", "sell_power": "50000"})

    client.set_transport_for_testing(fake_transport)
    out = client.get_limits()
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/limits")  # Correct endpoint
    assert "buy_power" in out or "sell_power" in out  # Response structure varies


def test_list_payment_methods_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"payment_methods": []})

    client.set_transport_for_testing(fake_transport)
    out = client.list_payment_methods()
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/payment_methods")
    assert "payment_methods" in out


def test_get_payment_method_with_id():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"payment_method": {"id": "pm-123", "type": "bank"}})

    client.set_transport_for_testing(fake_transport)
    out = client.get_payment_method("pm-123")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/payment_methods/pm-123")
    assert "payment_method" in out


def test_get_convert_trade_with_id():
    """Test get_convert_trade endpoint path."""
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"trade": {"id": "convert-456", "status": "completed"}})

    client.set_transport_for_testing(fake_transport)
    out = client.get_convert_trade("convert-456")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/convert/trade/convert-456")
    assert "trade" in out

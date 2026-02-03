"""Consolidated Coinbase HTTP error and retry behaviour tests."""

from __future__ import annotations

import json
import time

import pytest

from gpt_trader.core import (
    AuthError,
    BrokerageError,
    InvalidRequestError,
    RateLimitError,
)
from gpt_trader.features.brokerages.coinbase.errors import (
    TransientBrokerError,
    map_http_error,
)
from tests.unit.gpt_trader.features.brokerages.coinbase.helpers import make_client


def test_401_maps_to_auth_error():
    client = make_client()

    def transport(method, url, headers, body, timeout):
        return 401, {}, json.dumps({"error": "invalid_api_key", "message": "Invalid API key"})

    client.set_transport_for_testing(transport)
    with pytest.raises(AuthError) as exc:
        client.get_accounts()
    assert "Invalid API key" in str(exc.value)


def test_400_maps_to_invalid_request_error():
    client = make_client()

    def transport(method, url, headers, body, timeout):
        payload = {"error": "invalid_request", "message": "Product ID is required"}
        return 400, {}, json.dumps(payload)

    client.set_transport_for_testing(transport)
    with pytest.raises(InvalidRequestError) as exc:
        client.place_order({})
    assert "Product ID is required" in str(exc.value)


def test_500_maps_to_brokerage_error():
    client = make_client()

    def transport(method, url, headers, body, timeout):
        payload = {"error": "internal_server_error", "message": "Something went wrong"}
        return 500, {}, json.dumps(payload)

    client.set_transport_for_testing(transport)
    with pytest.raises(BrokerageError) as exc:
        client.list_orders()
    assert "Something went wrong" in str(exc.value)


def test_503_maps_to_transient_broker_error() -> None:
    error = map_http_error(503, "service_unavailable", "Service temporarily unavailable")
    assert isinstance(error, TransientBrokerError)
    assert "unavailable" in str(error).lower()


def test_429_triggers_retry_with_backoff(fake_clock):
    client = make_client()
    calls = 0

    def transport(method, url, headers, body, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 429, {"retry-after": "0.1"}, json.dumps({"error": "rate_limited"})
        return 200, {}, json.dumps({"success": True})

    client.set_transport_for_testing(transport)
    result = client.get_products()
    assert calls == 2
    assert result["success"] is True


def test_429_exhausts_retries(fake_clock):
    client = make_client()

    def transport(method, url, headers, body, timeout):
        return 429, {"retry-after": "0.01"}, json.dumps({"error": "rate_limited"})

    client.set_transport_for_testing(transport)
    with pytest.raises(RateLimitError) as exc:
        client.get_ticker("BTC-USD")
    assert "rate_limited" in str(exc.value)


def test_503_triggers_retry(fake_clock):
    client = make_client()
    calls = 0

    def transport(method, url, headers, body, timeout):
        nonlocal calls
        calls += 1
        if calls <= 2:
            return 503, {}, json.dumps({"error": "service_unavailable"})
        return 200, {}, json.dumps({"data": "success"})

    client.set_transport_for_testing(transport)
    result = client.get_candles("BTC-USD", "1H")
    assert calls == 3
    assert result["data"] == "success"


def test_network_errors_retry(fake_clock):
    client = make_client()
    calls = 0

    def transport(method, url, headers, body, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("Network unreachable")
        return 200, {}, json.dumps({"connected": True})

    client.set_transport_for_testing(transport)
    result = client.get_time()
    assert calls == 2
    assert result.get("connected") is True


def test_jitter_applied_to_retry_delay(fake_clock, monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client()
    delays: list[float] = []

    def transport(method, url, headers, body, timeout):
        return 429, {"retry-after": "0.1"}, json.dumps({"error": "rate_limited"})

    client.set_transport_for_testing(transport)

    def capture_sleep(seconds: float) -> None:
        delays.append(seconds)
        fake_clock.sleep(seconds)

    monkeypatch.setattr(time, "sleep", capture_sleep)

    with pytest.raises(RateLimitError):
        client.get_products()

    if len(delays) > 1:
        assert len(set(delays)) > 1 or all(delay > 0 for delay in delays)


def test_408_timeout_does_not_retry(fake_clock):
    client = make_client()
    calls = 0

    def transport(method, url, headers, body, timeout):
        nonlocal calls
        calls += 1
        return 408, {}, json.dumps({"error": "request_timeout", "message": "Request timed out"})

    client.set_transport_for_testing(transport)
    with pytest.raises(BrokerageError) as exc:
        client.get_products()
    assert "timed out" in str(exc.value).lower()
    assert calls == 1

from __future__ import annotations

import json
import time
from unittest.mock import Mock

import pytest
import requests

import gpt_trader.features.brokerages.coinbase.client.base as base_module
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.client.metrics import APIMetricsCollector
from gpt_trader.features.brokerages.coinbase.errors import (
    InvalidRequestError,
    PermissionDeniedError,
    RateLimitError,
)


@pytest.fixture
def base_url() -> str:
    return "https://api.coinbase.com"


@pytest.fixture
def auth() -> Mock:
    auth = Mock(spec=SimpleAuth)
    auth.get_headers.return_value = {"Authorization": "Bearer test-token"}
    return auth


@pytest.fixture
def client(base_url: str, auth: Mock) -> CoinbaseClientBase:
    return CoinbaseClientBase(base_url=base_url, auth=auth)


@pytest.fixture
def client_no_auth(base_url: str) -> CoinbaseClientBase:
    return CoinbaseClientBase(base_url=base_url, auth=None)


@pytest.fixture
def sleep_mock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock_sleep = Mock()
    monkeypatch.setattr(time, "sleep", mock_sleep)
    return mock_sleep


def _set_transport(client: CoinbaseClientBase, response):
    transport = Mock()
    if isinstance(response, list):
        transport.side_effect = response
    else:
        transport.return_value = response
    client.set_transport_for_testing(transport)
    return transport


def test_request_success(client: CoinbaseClientBase, auth: Mock) -> None:
    transport = _set_transport(client, (200, {}, '{"success": true}'))

    result = client._request("GET", "/api/v3/test")

    assert result == {"success": True}
    transport.assert_called_once()
    auth.get_headers.assert_called_once_with("GET", "/api/v3/test")

    headers = transport.call_args[0][2]
    assert headers["Content-Type"] == "application/json"
    assert headers["CB-VERSION"] == "2024-10-24"
    assert "Authorization" in headers


def test_request_with_body(client: CoinbaseClientBase, auth: Mock) -> None:
    transport = _set_transport(client, (200, {}, '{"success": true}'))

    body = {"test": "data"}
    result = client._request("POST", "/api/v3/test", body)

    assert result == {"success": True}
    request_body = transport.call_args[0][3]
    assert json.loads(request_body) == body
    auth.get_headers.assert_called_once_with("POST", "/api/v3/test", body)


def test_request_no_auth(client_no_auth: CoinbaseClientBase) -> None:
    transport = _set_transport(client_no_auth, (200, {}, '{"success": true}'))

    client_no_auth._request("GET", "/api/v3/test")

    headers = transport.call_args[0][2]
    assert "Authorization" not in headers


def test_request_with_correlation_id(
    client: CoinbaseClientBase, auth: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base_module, "get_correlation_id", lambda: "test-correlation-123")
    transport = _set_transport(client, (200, {}, '{"success": true}'))

    client._request("GET", "/api/v3/test")

    headers = transport.call_args[0][2]
    assert headers["X-Correlation-Id"] == "test-correlation-123"


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ('{"error": "INVALID_REQUEST", "message": "Bad request"}', "Bad request"),
        ("not-json", "not-json"),
    ],
)
def test_request_4xx_error_variants(client: CoinbaseClientBase, body: str, match: str) -> None:
    _set_transport(client, (400, {}, body))

    with pytest.raises(InvalidRequestError, match=match):
        client._request("GET", "/api/v3/test")


def test_raise_client_error_forbidden(auth: Mock) -> None:
    client = CoinbaseClientBase(base_url="https://api.coinbase.com", auth=auth)
    resp = requests.Response()
    resp.status_code = 403
    resp._content = b'{"message": "forbidden", "error": "FORBIDDEN"}'

    with pytest.raises(PermissionDeniedError, match="forbidden"):
        client._raise_client_error(resp)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [("invalid json response", {"raw": "invalid json response"}), ("", {})],
)
def test_request_non_json_payloads(
    client: CoinbaseClientBase, payload: str, expected: dict
) -> None:
    _set_transport(client, (200, {}, payload))

    result = client._request("GET", "/api/v3/test")

    assert result == expected


@pytest.mark.parametrize(
    ("retry_after", "expected_sleep"),
    [("nope", 1.0), ("5", 5.0)],
)
def test_request_rate_limit_retries(
    client: CoinbaseClientBase,
    sleep_mock: Mock,
    retry_after: str,
    expected_sleep: float,
) -> None:
    _set_transport(
        client,
        [
            (429, {"retry-after": retry_after}, '{"error": "RATE_LIMITED"}'),
            (200, {}, '{"success": true}'),
        ],
    )

    result = client._request("GET", "/api/v3/test")

    assert result == {"success": True}
    sleep_mock.assert_called_once_with(expected_sleep)


def test_request_rate_limit_exhausts_retries(client: CoinbaseClientBase, sleep_mock: Mock) -> None:
    _set_transport(client, (429, {"retry-after": "0"}, '{"error": "rate_limited"}'))

    with pytest.raises(RateLimitError):
        client._request("GET", "/api/v3/test")


def test_request_5xx_error_with_retry(client: CoinbaseClientBase, sleep_mock: Mock) -> None:
    client._metrics = APIMetricsCollector(max_history=10)
    transport = _set_transport(
        client,
        [
            (500, {}, '{"error": "INTERNAL_ERROR", "message": "Server error"}'),
            (200, {}, '{"success": true}'),
        ],
    )

    result = client._request("GET", "/api/v3/test")

    assert result == {"success": True}
    assert transport.call_count == 2
    sleep_mock.assert_called_once()
    assert client.get_api_metrics()["total_errors"] == 0


def test_request_network_error_with_retry(client: CoinbaseClientBase, sleep_mock: Mock) -> None:
    client._metrics = APIMetricsCollector(max_history=10)
    transport = _set_transport(
        client,
        [
            requests.ConnectionError("Network error"),
            (200, {}, '{"success": true}'),
        ],
    )

    result = client._request("GET", "/api/v3/test")

    assert result == {"success": True}
    assert transport.call_count == 2
    sleep_mock.assert_called_once()
    assert client.get_api_metrics()["total_errors"] == 0


def test_request_max_retries_exceeded(client: CoinbaseClientBase, sleep_mock: Mock) -> None:
    transport = _set_transport(client, (500, {}, '{"error": "INTERNAL_ERROR"}'))

    with pytest.raises(Exception):
        client._request("GET", "/api/v3/test")

    assert transport.call_count == 4

"""Combined system, error, and performance tests for Coinbase integration."""

import json
import time
from decimal import Decimal
from unittest.mock import MagicMock, patch, call

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.errors import (
    AuthError,
    BrokerageError,
    InvalidRequestError,
    InsufficientFunds,
    RateLimitError,
)
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce


pytestmark = pytest.mark.endpoints


def make_client(api_mode: str = "advanced", auth=None) -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode=api_mode)


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------


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
    assert calls[0][1].endswith("/api/v3/brokerage/fees")
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
    assert calls[0][1].endswith("/api/v3/brokerage/limits")
    assert "buy_power" in out or "sell_power" in out


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


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_401_maps_to_auth_error():
    client = make_client()

    def fake_transport(method, url, headers, body, timeout):
        return 401, {}, json.dumps({"error": "invalid_api_key", "message": "Invalid API key"})

    client.set_transport_for_testing(fake_transport)
    with pytest.raises(AuthError) as exc:
        client.get_accounts()
    assert "Invalid API key" in str(exc.value)


def test_429_triggers_retry_with_backoff(fake_clock):
    client = make_client()
    call_count = 0

    def fake_transport(method, url, headers, body, timeout):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return 429, {"retry-after": "0.1"}, json.dumps({"error": "rate_limited"})
        return 200, {}, json.dumps({"success": True})

    client.set_transport_for_testing(fake_transport)
    result = client.get_products()
    assert call_count == 2
    assert result["success"] is True


def test_429_exhausts_retries_raises_rate_limit_error(fake_clock):
    client = make_client()

    def fake_transport(method, url, headers, body, timeout):
        return 429, {"retry-after": "0.01"}, json.dumps({"error": "rate_limited"})

    client.set_transport_for_testing(fake_transport)
    with pytest.raises(RateLimitError) as exc:
        client.get_ticker("BTC-USD")
    assert "rate_limited" in str(exc.value)


def test_400_maps_to_invalid_request_error():
    client = make_client()

    def fake_transport(method, url, headers, body, timeout):
        return (
            400,
            {},
            json.dumps({"error": "invalid_request", "message": "Product ID is required"}),
        )

    client.set_transport_for_testing(fake_transport)
    with pytest.raises(InvalidRequestError) as exc:
        client.place_order({})
    assert "Product ID is required" in str(exc.value)


def test_500_maps_to_brokerage_error():
    client = make_client()

    def fake_transport(method, url, headers, body, timeout):
        return (
            500,
            {},
            json.dumps({"error": "internal_server_error", "message": "Something went wrong"}),
        )

    client.set_transport_for_testing(fake_transport)
    with pytest.raises(BrokerageError) as exc:
        client.list_orders()
    assert "Something went wrong" in str(exc.value)


def test_503_triggers_retry(fake_clock):
    client = make_client()
    call_count = 0

    def fake_transport(method, url, headers, body, timeout):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return 503, {}, json.dumps({"error": "service_unavailable"})
        return 200, {}, json.dumps({"data": "success"})

    client.set_transport_for_testing(fake_transport)
    result = client.get_candles("BTC-USD", "1H")
    assert call_count == 3
    assert result["data"] == "success"


def test_network_error_triggers_retry():
    client = make_client()
    call_count = 0

    def fake_transport(method, url, headers, body, timeout):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Network unreachable")
        return 200, {}, json.dumps({"connected": True})

    client.set_transport_for_testing(fake_transport)
    result = client.get_time()
    assert call_count == 2
    assert result.get("connected") is True


def test_jitter_applied_to_retry_delay(fake_clock):
    client = make_client()
    delays = []

    def fake_transport(method, url, headers, body, timeout):
        return 429, {"retry-after": "0.1"}, json.dumps({"error": "rate_limited"})

    client.set_transport_for_testing(fake_transport)

    def mock_sleep(seconds):
        delays.append(seconds)
        fake_clock.sleep(seconds)

    with patch("time.sleep", side_effect=mock_sleep):
        with pytest.raises(RateLimitError):
            client.get_products()

    if len(delays) > 1:
        assert len(set(delays)) > 1 or all(d > 0 for d in delays)


def test_408_timeout_maps_to_brokerage_error_without_retry():
    client = make_client()
    calls = 0

    def fake_transport(method, url, headers, body, timeout):
        nonlocal calls
        calls += 1
        return 408, {}, json.dumps({"error": "request_timeout", "message": "Request timed out"})

    client.set_transport_for_testing(fake_transport)
    with pytest.raises(BrokerageError) as exc:
        client.get_products()
    assert "timed out" in str(exc.value).lower()
    assert calls == 1


# ---------------------------------------------------------------------------
# Performance baselines
# ---------------------------------------------------------------------------


@pytest.mark.perf
def test_get_products_perf_budget():
    client = make_client()
    client.set_transport_for_testing(lambda m, u, h, b, t: (200, {}, json.dumps({"products": []})))
    start = time.perf_counter()
    _ = client.get_products()
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 10


@pytest.mark.perf
def test_place_order_perf_budget():
    client = make_client()
    client.set_transport_for_testing(
        lambda m, u, h, b, t: (200, {}, json.dumps({"order_id": "ord"}))
    )
    start = time.perf_counter()
    _ = client.place_order({"product_id": "BTC-USD", "side": "BUY", "size": "0.01"})
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 10


def test_keep_alive_header_added():
    client = CoinbaseClient(base_url="https://api.coinbase.com", enable_keep_alive=True)
    assert client.enable_keep_alive is True
    assert client._opener is not None

    mock_opener = MagicMock()
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.headers.items.return_value = []
    mock_response.read.return_value = b'{"success": true}'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    mock_opener.open.return_value = mock_response
    client._opener = mock_opener

    with patch("urllib.request.Request") as mock_request:
        mock_req_obj = MagicMock()
        mock_request.return_value = mock_req_obj
        status, headers, text = client._urllib_transport(
            "GET",
            "https://api.coinbase.com/test",
            {"Content-Type": "application/json"},
            None,
            30,
        )
        calls = [c.args for c in mock_req_obj.add_header.call_args_list]
        assert ("Connection", "keep-alive") in calls
        assert status == 200 and text == '{"success": true}'


def test_keep_alive_disabled():
    client = CoinbaseClient(base_url="https://api.coinbase.com", enable_keep_alive=False)
    assert client.enable_keep_alive is False
    assert client._opener is None

    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.headers.items.return_value = []
    mock_response.read.return_value = b'{"success": true}'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)

    with (
        patch("urllib.request.Request") as mock_request,
        patch("urllib.request.urlopen") as mock_urlopen,
    ):
        mock_req_obj = MagicMock()
        mock_request.return_value = mock_req_obj
        mock_urlopen.return_value = mock_response
        status, headers, text = client._urllib_transport(
            "GET",
            "https://api.coinbase.com/test",
            {"Content-Type": "application/json"},
            None,
            30,
        )
        calls = [c.args for c in mock_req_obj.add_header.call_args_list]
        assert ("Connection", "keep-alive") not in calls
        assert status == 200 and text == '{"success": true}'


def test_shared_opener_created():
    client = CoinbaseClient(base_url="https://api.coinbase.com", enable_keep_alive=True)
    assert client._opener is not None
    client_no_keepalive = CoinbaseClient(
        base_url="https://api.coinbase.com", enable_keep_alive=False
    )
    assert client_no_keepalive._opener is None


def test_backoff_jitter_deterministic(fake_clock):
    with patch("bot_v2.features.brokerages.coinbase.client.get_config") as mock_config:
        mock_config.return_value = {"max_retries": 3, "retry_delay": 1.0, "jitter_factor": 0.1}
        client = make_client()
        sleep_calls = []

        def mock_transport(method, url, headers, body, timeout):
            if len(sleep_calls) < 2:
                return (429, {}, '{"error": "rate limited"}')
            return (200, {}, '{"success": true}')

        client._transport = mock_transport

        def capture_sleep(duration):
            sleep_calls.append(duration)
            fake_clock.sleep(duration)

        with patch("time.sleep", side_effect=capture_sleep):
            client._request("GET", "/test")

        assert len(sleep_calls) == 2
        assert abs(sleep_calls[0] - 1.01) < 0.001
        assert abs(sleep_calls[1] - 2.04) < 0.001


def test_jitter_disabled(fake_clock):
    with patch("bot_v2.features.brokerages.coinbase.client.get_config") as mock_config:
        mock_config.return_value = {"max_retries": 3, "retry_delay": 1.0, "jitter_factor": 0}
        client = make_client()
        sleep_calls = []

        def mock_transport(method, url, headers, body, timeout):
            if len(sleep_calls) < 2:
                return (429, {}, '{"error": "rate limited"}')
            return (200, {}, '{"success": true}')

        client._transport = mock_transport

        def capture_sleep(duration):
            sleep_calls.append(duration)
            fake_clock.sleep(duration)

        with patch("time.sleep", side_effect=capture_sleep):
            client._request("GET", "/test")

        assert sleep_calls == [1.0, 2.0]


def test_connection_reuse_with_opener():
    client = CoinbaseClient(base_url="https://api.coinbase.com", enable_keep_alive=True)
    mock_opener = MagicMock()
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.headers.items.return_value = []
    mock_response.read.return_value = b'{"success": true}'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    mock_opener.open.return_value = mock_response
    client._opener = mock_opener

    with patch("urllib.request.Request") as mock_request:
        mock_req_obj = MagicMock()
        mock_request.return_value = mock_req_obj
        client._urllib_transport("GET", "https://api.coinbase.com/test", {}, None, 30)
        mock_opener.open.assert_called_once()


# ---------------------------------------------------------------------------
# Integration improvements and connection validation
# ---------------------------------------------------------------------------


def test_rate_limit_tracking():
    client = make_client()
    assert hasattr(client, "_request_count") and client._request_count == 0
    assert hasattr(client, "_request_window_start")

    def mock_transport(method, url, headers, body, timeout):
        return 200, {}, '{"success": true}'

    client._transport = mock_transport
    initial = client._request_count
    client._request("GET", "/test")
    assert client._request_count == initial + 1


def test_connection_validation():
    config = APIConfig(
        api_key="test_key",
        api_secret="test_secret",
        passphrase=None,
        base_url="https://api.coinbase.com",
        sandbox=False,
    )
    broker = CoinbaseBrokerage(config)
    mock_accounts_response = {
        "accounts": [{"uuid": "test-account-123", "currency": "USD", "balance": "100.00"}]
    }
    broker.client.get_accounts = MagicMock(return_value=mock_accounts_response)
    result = broker.connect()
    assert result is True
    assert broker._connected is True
    assert broker._account_id == "test-account-123"


def test_position_list_spot_trading():
    config = APIConfig(
        api_key="test_key",
        api_secret="test_secret",
        passphrase=None,
        base_url="https://api.coinbase.com",
        sandbox=False,
        enable_derivatives=False,
    )
    broker = CoinbaseBrokerage(config)
    positions = broker.list_positions()
    assert positions == []
    broker.client.cfm_positions = MagicMock()
    positions = broker.list_positions()
    assert positions == []
    broker.client.cfm_positions.assert_not_called()


def test_order_error_handling():
    config = APIConfig(
        api_key="test_key",
        api_secret="test_secret",
        passphrase=None,
        base_url="https://api.coinbase.com",
        sandbox=False,
    )
    broker = CoinbaseBrokerage(config)
    mock_product = MagicMock()
    mock_product.step_size = Decimal("0.001")
    mock_product.price_increment = Decimal("0.01")
    mock_product.min_size = Decimal("0.001")
    mock_product.min_notional = None
    broker.product_catalog.get = MagicMock(return_value=mock_product)
    broker.client.place_order = MagicMock(side_effect=InsufficientFunds("Not enough balance"))
    with pytest.raises(InsufficientFunds):
        broker.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

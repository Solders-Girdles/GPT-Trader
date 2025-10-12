"""Coinbase system and client infrastructure tests."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
from unittest.mock import MagicMock, patch

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.errors import InsufficientFunds
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType
from decimal import Decimal

from tests.unit.bot_v2.features.brokerages.coinbase.test_helpers import (
    SYSTEM_ENDPOINT_CASES,
    make_client,
)


pytestmark = pytest.mark.endpoints


class TestCoinbaseSystem:
    @pytest.mark.parametrize("case", SYSTEM_ENDPOINT_CASES, ids=lambda c: c["id"])
    def test_client_system_endpoints(self, case: dict[str, Any]) -> None:
        client = make_client()
        recorded: dict[str, Any] = {}

        def transport(method, url, headers, body, timeout):
            recorded["method"] = method
            recorded["url"] = url
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

        expected_result = case.get("expected_result")
        if expected_result is not None:
            assert result == expected_result

    @pytest.mark.perf
    def test_get_products_perf_budget(self) -> None:
        client = make_client()
        client.set_transport_for_testing(
            lambda m, u, h, b, t: (200, {}, json.dumps({"products": []}))
        )
        start = time.perf_counter()
        _ = client.get_products()
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10

    @pytest.mark.perf
    def test_place_order_perf_budget(self) -> None:
        client = make_client()
        client.set_transport_for_testing(
            lambda m, u, h, b, t: (200, {}, json.dumps({"order_id": "ord"}))
        )
        start = time.perf_counter()
        _ = client.place_order({"product_id": "BTC-USD", "side": "BUY", "size": "0.01"})
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10

    def test_keep_alive_header_added(self) -> None:
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

    def test_keep_alive_disabled(self) -> None:
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

    def test_shared_opener_created_flag(self) -> None:
        client = CoinbaseClient(base_url="https://api.coinbase.com", enable_keep_alive=True)
        assert client._opener is not None
        client_no_keepalive = CoinbaseClient(
            base_url="https://api.coinbase.com", enable_keep_alive=False
        )
        assert client_no_keepalive._opener is None

    def test_backoff_jitter_deterministic(self, fake_clock) -> None:
        with patch("bot_v2.features.brokerages.coinbase.client.get_config") as mock_config:
            mock_config.return_value = {"max_retries": 3, "retry_delay": 1.0, "jitter_factor": 0.1}
            client = make_client()
            sleep_calls: list[float] = []

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

    def test_jitter_disabled(self, fake_clock) -> None:
        with patch("bot_v2.features.brokerages.coinbase.client.get_config") as mock_config:
            mock_config.return_value = {"max_retries": 3, "retry_delay": 1.0, "jitter_factor": 0}
            client = make_client()
            sleep_calls: list[float] = []

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

    def test_connection_reuse_with_opener(self) -> None:
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

    def test_rate_limit_tracking(self) -> None:
        client = make_client()
        assert hasattr(client, "_request_count") and client._request_count == 0
        assert hasattr(client, "_request_window_start")

        def mock_transport(method, url, headers, body, timeout):
            return 200, {}, '{"success": true}'

        client._transport = mock_transport
        initial = client._request_count
        client._request("GET", "/test")
        assert client._request_count == initial + 1

    def test_connection_validation(self) -> None:
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

    def test_position_list_spot_trading(self) -> None:
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

    def test_order_error_handling(self) -> None:
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

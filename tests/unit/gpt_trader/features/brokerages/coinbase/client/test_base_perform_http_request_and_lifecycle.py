"""Tests for CoinbaseClientBase low-level HTTP + lifecycle helpers."""

import time
from unittest.mock import Mock, patch

import pytest
import requests

import gpt_trader.features.brokerages.coinbase.client.base as base_module
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.errors import BrokerageError


class TestCoinbaseClientBasePerformHttpRequestAndLifecycle:
    """Test CoinbaseClientBase low-level request helpers and session lifecycle."""

    def setup_method(self) -> None:
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

    def test_perform_http_request_http_error_invalid_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test HTTPError path when error response is not JSON."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        response = requests.Response()
        response.status_code = 500
        response._content = b"not-json"
        client._perform_request = Mock(return_value=response)
        client._check_rate_limit = Mock()
        monkeypatch.setattr(base_module, "MAX_HTTP_RETRIES", 0)

        with pytest.raises(BrokerageError):
            client._perform_http_request(
                "GET",
                "/api/v3/test",
                None,
                time.perf_counter(),
                None,
            )

    def test_perform_http_request_network_error_final_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test final network error attempt raises without retry."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._perform_request = Mock(side_effect=requests.ConnectionError("boom"))
        client._check_rate_limit = Mock()
        monkeypatch.setattr(base_module, "MAX_HTTP_RETRIES", 0)

        with pytest.raises(requests.ConnectionError):
            client._perform_http_request(
                "GET",
                "/api/v3/test",
                None,
                time.perf_counter(),
                None,
            )

    def test_perform_http_request_records_failure_and_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test failures record circuit breaker and span attributes."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._perform_request = Mock(side_effect=RuntimeError("boom"))
        client._check_rate_limit = Mock()
        client._circuit_breaker = Mock()
        span = Mock()
        monkeypatch.setattr(base_module, "MAX_HTTP_RETRIES", 0)

        with pytest.raises(RuntimeError):
            client._perform_http_request(
                "GET",
                "/api/v3/test",
                None,
                time.perf_counter(),
                span,
            )

        client._circuit_breaker.record_failure.assert_called_once()
        span.set_attribute.assert_any_call("error", True)
        span.set_attribute.assert_any_call("error.type", "RuntimeError")

    def test_context_manager_closes(self) -> None:
        """Test context manager closes session."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client.close = Mock()

        with client as ctx:
            assert ctx is client

        client.close.assert_called_once()

    def test_close_handles_session_error(self) -> None:
        """Test close logs warning on session close error."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client.session.close = Mock(side_effect=RuntimeError("boom"))

        with patch("gpt_trader.features.brokerages.coinbase.client.base.logger") as mock_logger:
            client.close()

        mock_logger.warning.assert_called_once()

    def test_del_suppresses_close_errors(self) -> None:
        """Test destructor suppresses close errors."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client.close = Mock(side_effect=RuntimeError("boom"))

        client.__del__()

"""Tests for CoinbaseClientBase metrics recording and resilience status."""

from unittest.mock import Mock, patch

import requests

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase


class TestCoinbaseClientBaseMetricsAndResilienceStatus:
    """Test CoinbaseClientBase observability helpers."""

    def setup_method(self) -> None:
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

    def test_record_success_sets_span_and_breaker(self) -> None:
        """Test success recording updates circuit breaker and span."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._circuit_breaker = Mock()
        resp = requests.Response()
        resp.status_code = 200
        span = Mock()

        client._record_success("/api/v3/test", resp, span)

        client._circuit_breaker.record_success.assert_called_once_with("/api/v3/test")
        span.set_attribute.assert_called_once_with("http.status_code", 200)

    def test_record_request_metrics_records_metrics_and_span(self) -> None:
        """Test request metrics recording updates metrics and span attributes."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._metrics = Mock()
        span = Mock()

        with (
            patch(
                "gpt_trader.features.brokerages.coinbase.client.base.record_histogram"
            ) as mock_hist,
            patch(
                "gpt_trader.features.brokerages.coinbase.client.base.record_counter"
            ) as mock_counter,
        ):
            client._record_request_metrics("/api/v3/test", 0.25, True, False, span)

        client._metrics.record_request.assert_called_once_with(
            "/api/v3/test",
            250.0,
            error=True,
            rate_limited=False,
        )
        mock_hist.assert_called_once()
        mock_counter.assert_called_once()
        span.set_attribute.assert_any_call("http.latency_ms", 250.0)
        span.set_attribute.assert_any_call("http.rate_limited", False)

    def test_get_resilience_status_records_metrics(self) -> None:
        """Test resilience status includes metrics and circuit breaker info."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client.get_rate_limit_usage = Mock(return_value=0.5)
        client._metrics = Mock()
        client._metrics.get_summary.return_value = {"error_rate": 0.2}
        client._circuit_breaker = Mock()
        client._circuit_breaker.get_all_status.return_value = {"orders": {"state": "open"}}
        client._response_cache = Mock()
        client._response_cache.get_stats.return_value = {"entries": 1}
        client._priority_manager = Mock()
        client._priority_manager.get_stats.return_value = {"deferred": 2}

        with patch(
            "gpt_trader.features.brokerages.coinbase.client.base.record_gauge"
        ) as mock_gauge:
            status = client.get_resilience_status()

        assert status["metrics"] == {"error_rate": 0.2}
        assert status["circuit_breakers"] == {"orders": {"state": "open"}}
        assert status["cache"] == {"entries": 1}
        assert status["priority"] == {"deferred": 2}
        mock_gauge.assert_any_call("gpt_trader_rate_limit_usage_ratio", 0.5)
        mock_gauge.assert_any_call("gpt_trader_api_error_rate", 0.2)
        mock_gauge.assert_any_call(
            "gpt_trader_circuit_breaker_state",
            2.0,
            labels={"category": "orders"},
        )

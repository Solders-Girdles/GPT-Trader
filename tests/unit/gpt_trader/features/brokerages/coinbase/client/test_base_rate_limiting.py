"""Tests for CoinbaseClientBase throttling/rate limit helpers."""

from unittest.mock import Mock, patch

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase


class TestCoinbaseClientBaseRateLimiting:
    """Test CoinbaseClientBase rate limit behavior."""

    def setup_method(self) -> None:
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.time")
    def test_check_rate_limit_disabled(self, mock_time: Mock) -> None:
        """Test rate limit checking when disabled."""
        mock_time.return_value = 1000.0
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, enable_throttle=False)

        # Should not raise any exceptions or sleep
        client._check_rate_limit()

        assert len(client._request_times) == 0

    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.time")
    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.sleep")
    @patch("gpt_trader.features.brokerages.coinbase.client.base.logger")
    def test_check_rate_limit_normal_usage(
        self, mock_logger: Mock, mock_sleep: Mock, mock_time: Mock
    ) -> None:
        """Test rate limit checking with normal usage."""
        mock_time.return_value = 1000.0
        client = CoinbaseClientBase(
            base_url=self.base_url, auth=self.auth, rate_limit_per_minute=10
        )

        # Add some requests within the limit
        for i in range(5):
            client._check_rate_limit()

        assert len(client._request_times) == 5
        mock_sleep.assert_not_called()

    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.time")
    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.sleep")
    @patch("gpt_trader.features.brokerages.coinbase.client.base.logger")
    def test_check_rate_limit_warning_threshold(
        self, mock_logger: Mock, mock_sleep: Mock, mock_time: Mock
    ) -> None:
        """Test rate limit warning threshold."""
        mock_time.return_value = 1000.0
        client = CoinbaseClientBase(
            base_url=self.base_url, auth=self.auth, rate_limit_per_minute=10
        )

        # Add requests up to 80% of limit
        # Need to trigger the warning, which happens when len >= 8
        for i in range(9):
            client._check_rate_limit()

        mock_logger.warning.assert_called_once_with(
            "Approaching rate limit: %d/%d requests in last minute",
            8,
            10,
        )

    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.time")
    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.sleep")
    @patch("gpt_trader.features.brokerages.coinbase.client.base.logger")
    def test_check_rate_limit_exceeded(
        self, mock_logger: Mock, mock_sleep: Mock, mock_time: Mock
    ) -> None:
        """Test rate limit exceeded behavior."""
        # Start at time 1000
        mock_time.return_value = 1000.0
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, rate_limit_per_minute=5)

        # Add requests up to the limit
        for i in range(5):
            client._check_rate_limit()

        # Next request should trigger rate limiting
        # Advance time by 30 seconds (still within 1-minute window)
        mock_time.return_value = 1030.0
        client._check_rate_limit()

        # Should have slept to respect rate limit
        assert mock_sleep.call_count >= 1
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list if c.args]
        assert any(s > 30 for s in sleep_calls)  # remaining time + buffer

        mock_logger.info.assert_called_once()
        assert "Rate limit reached" in mock_logger.info.call_args[0][0]

    @patch("gpt_trader.features.brokerages.coinbase.client.base.time.time")
    def test_check_rate_limit_window_cleanup(self, mock_time: Mock) -> None:
        """Test rate limit window cleanup."""
        client = CoinbaseClientBase(
            base_url=self.base_url, auth=self.auth, rate_limit_per_minute=10
        )

        # Add requests at time 1000
        mock_time.return_value = 1000.0
        for i in range(5):
            client._check_rate_limit()

        assert len(client._request_times) == 5

        # Advance time beyond 1 minute window
        mock_time.return_value = 2000.0
        client._check_rate_limit()

        # Old requests should be cleaned up
        assert len(client._request_times) == 1
        assert client._request_times[0] == 2000.0

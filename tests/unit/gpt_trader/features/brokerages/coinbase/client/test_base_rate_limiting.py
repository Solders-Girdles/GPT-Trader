"""Tests for CoinbaseClientBase throttling/rate limit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, Mock

import pytest

import gpt_trader.features.brokerages.coinbase.client.base as base_module
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase


@dataclass(slots=True)
class RateLimitHarness:
    now: float
    sleep: MagicMock
    logger: MagicMock


@pytest.fixture
def rate_limit_harness(monkeypatch: pytest.MonkeyPatch) -> RateLimitHarness:
    sleep = MagicMock(name="sleep")
    logger = MagicMock(name="logger")
    harness = RateLimitHarness(now=0.0, sleep=sleep, logger=logger)

    monkeypatch.setattr(base_module.time, "time", lambda: harness.now)
    monkeypatch.setattr(base_module.time, "sleep", sleep)
    monkeypatch.setattr(base_module, "logger", logger)

    return harness


class TestCoinbaseClientBaseRateLimiting:
    """Test CoinbaseClientBase rate limit behavior."""

    def setup_method(self) -> None:
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

    def test_check_rate_limit_disabled(self, rate_limit_harness: RateLimitHarness) -> None:
        """Test rate limit checking when disabled."""
        rate_limit_harness.now = 1000.0
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, enable_throttle=False)

        client._check_rate_limit()

        assert len(client._request_times) == 0

    def test_check_rate_limit_normal_usage(self, rate_limit_harness: RateLimitHarness) -> None:
        """Test rate limit checking with normal usage."""
        rate_limit_harness.now = 1000.0
        client = CoinbaseClientBase(
            base_url=self.base_url, auth=self.auth, rate_limit_per_minute=10
        )
        client._adaptive_throttle_enabled = False

        for _ in range(5):
            client._check_rate_limit()

        assert len(client._request_times) == 5
        rate_limit_harness.sleep.assert_not_called()

    def test_check_rate_limit_warning_threshold(self, rate_limit_harness: RateLimitHarness) -> None:
        """Test rate limit warning threshold."""
        rate_limit_harness.now = 1000.0
        client = CoinbaseClientBase(
            base_url=self.base_url, auth=self.auth, rate_limit_per_minute=10
        )
        client._adaptive_throttle_enabled = False

        for _ in range(9):
            client._check_rate_limit()

        rate_limit_harness.logger.warning.assert_called_once_with(
            "Approaching rate limit: %d/%d requests in last minute",
            8,
            10,
        )

    def test_check_rate_limit_exceeded(self, rate_limit_harness: RateLimitHarness) -> None:
        """Test rate limit exceeded behavior."""
        rate_limit_harness.now = 1000.0
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth, rate_limit_per_minute=5)
        client._adaptive_throttle_enabled = False

        for _ in range(5):
            client._check_rate_limit()

        rate_limit_harness.now = 1030.0
        client._check_rate_limit()

        assert rate_limit_harness.sleep.call_count >= 1
        sleep_calls = [c.args[0] for c in rate_limit_harness.sleep.call_args_list if c.args]
        assert any(s > 30 for s in sleep_calls)  # remaining time + buffer

        rate_limit_harness.logger.info.assert_called_once()
        assert "Rate limit reached" in rate_limit_harness.logger.info.call_args[0][0]

    def test_check_rate_limit_window_cleanup(self, rate_limit_harness: RateLimitHarness) -> None:
        """Test rate limit window cleanup."""
        client = CoinbaseClientBase(
            base_url=self.base_url, auth=self.auth, rate_limit_per_minute=10
        )
        client._adaptive_throttle_enabled = False

        rate_limit_harness.now = 1000.0
        for _ in range(5):
            client._check_rate_limit()

        assert len(client._request_times) == 5

        rate_limit_harness.now = 2000.0
        client._check_rate_limit()

        assert len(client._request_times) == 1
        assert client._request_times[0] == 2000.0

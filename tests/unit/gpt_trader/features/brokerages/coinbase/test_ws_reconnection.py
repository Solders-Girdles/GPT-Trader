"""Tests for WebSocket reconnection with exponential backoff and jitter."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from gpt_trader.features.brokerages.coinbase.ws import (
    CoinbaseWebSocket,
    calculate_backoff_with_jitter,
)


class TestCalculateBackoffWithJitter:
    """Tests for the backoff calculation function."""

    def test_backoff_grows_exponentially(self) -> None:
        """Test that backoff delay grows with each attempt."""
        # Disable jitter for deterministic testing
        delays = [calculate_backoff_with_jitter(attempt, jitter_pct=0.0) for attempt in range(5)]

        # Each delay should be greater than the previous (exponential growth)
        for i in range(1, len(delays)):
            assert delays[i] > delays[i - 1], f"Delay {i} should be > delay {i - 1}"

    def test_backoff_respects_max(self) -> None:
        """Test that backoff delay is capped at max_seconds."""
        max_seconds = 30.0

        # High attempt count should hit the max
        delay = calculate_backoff_with_jitter(attempt=20, max_seconds=max_seconds, jitter_pct=0.0)

        assert delay == max_seconds

    def test_backoff_uses_base_at_attempt_zero(self) -> None:
        """Test that first attempt uses base delay."""
        base = 2.0
        delay = calculate_backoff_with_jitter(attempt=0, base_seconds=base, jitter_pct=0.0)
        assert delay == base

    def test_backoff_uses_multiplier(self) -> None:
        """Test that multiplier is applied correctly."""
        base = 2.0
        multiplier = 3.0

        delay0 = calculate_backoff_with_jitter(
            attempt=0, base_seconds=base, multiplier=multiplier, jitter_pct=0.0
        )
        delay1 = calculate_backoff_with_jitter(
            attempt=1, base_seconds=base, multiplier=multiplier, jitter_pct=0.0
        )
        delay2 = calculate_backoff_with_jitter(
            attempt=2, base_seconds=base, multiplier=multiplier, jitter_pct=0.0
        )

        assert delay0 == base  # 2.0
        assert delay1 == base * multiplier  # 6.0
        assert delay2 == base * multiplier * multiplier  # 18.0

    def test_jitter_within_bounds(self) -> None:
        """Test that jitter stays within ±jitter_pct bounds."""
        base = 10.0
        jitter_pct = 0.25  # ±25%

        # Run many iterations to test randomness bounds
        for _ in range(100):
            delay = calculate_backoff_with_jitter(
                attempt=0, base_seconds=base, jitter_pct=jitter_pct, max_seconds=100.0
            )

            # With 25% jitter, delay should be between 7.5 and 12.5
            min_expected = base * (1 - jitter_pct)
            max_expected = base * (1 + jitter_pct)

            assert (
                min_expected <= delay <= max_expected
            ), f"Delay {delay} out of bounds [{min_expected}, {max_expected}]"

    def test_jitter_produces_different_values(self) -> None:
        """Test that jitter produces variation across calls."""
        # Collect multiple values
        delays = [calculate_backoff_with_jitter(attempt=0, jitter_pct=0.25) for _ in range(20)]

        # Should have at least some variation (not all identical)
        unique_delays = set(delays)
        assert len(unique_delays) > 1, "Jitter should produce varying delays"

    @patch("gpt_trader.features.brokerages.coinbase.ws.random.uniform")
    def test_jitter_uses_random_uniform(self, mock_uniform: MagicMock) -> None:
        """Test that jitter uses random.uniform for deterministic testing."""
        mock_uniform.return_value = 0.5  # Return positive jitter

        base = 10.0
        jitter_pct = 0.25
        delay = calculate_backoff_with_jitter(
            attempt=0, base_seconds=base, jitter_pct=jitter_pct, max_seconds=100.0
        )

        # Should be base + jitter (0.5)
        assert delay == base + 0.5
        mock_uniform.assert_called_once()


class TestWebSocketReconnectionReset:
    """Tests for reconnection attempt counter reset after stable connection."""

    def test_attempts_reset_after_stable_period(self) -> None:
        """Test that reconnect counter resets after stable connection."""
        ws = CoinbaseWebSocket()

        # Simulate some failed reconnection attempts
        ws._reconnect_count = 5
        ws._running.set()

        # Simulate a connection that was stable for 120 seconds
        ws._connected_since = time.time() - 120.0

        # Patch WS_RECONNECT_RESET_SECONDS to 60 for test
        with patch(
            "gpt_trader.features.brokerages.coinbase.ws.WS_RECONNECT_RESET_SECONDS",
            60.0,
        ):
            # Simulate _on_close being called (connection dropped)
            # Don't actually reconnect - just test the reset logic
            ws._shutdown.set()  # Prevent actual reconnection
            ws._on_close(None, 1000, "Normal closure")

        # Counter should be reset because connection was stable
        assert ws._reconnect_count == 0

    def test_attempts_not_reset_if_unstable(self) -> None:
        """Test that reconnect counter is NOT reset if connection was short-lived."""
        ws = CoinbaseWebSocket()

        # Simulate some failed reconnection attempts
        ws._reconnect_count = 5
        ws._running.set()

        # Simulate a connection that was only stable for 10 seconds
        ws._connected_since = time.time() - 10.0

        # Patch WS_RECONNECT_RESET_SECONDS to 60 for test
        with patch(
            "gpt_trader.features.brokerages.coinbase.ws.WS_RECONNECT_RESET_SECONDS",
            60.0,
        ):
            # Simulate _on_close being called
            ws._shutdown.set()  # Prevent actual reconnection
            ws._on_close(None, 1000, "Normal closure")

        # Counter should NOT be reset (connection wasn't stable long enough)
        # Note: counter increments by 1 because of the close event
        assert ws._reconnect_count == 5  # Stays at 5 (no increment since shutdown)


class TestWebSocketDegradationCallback:
    """Tests for degradation callback when max attempts exceeded."""

    def test_callback_triggered_on_max_attempts(self) -> None:
        """Test that degradation callback is triggered when max attempts exceeded."""
        callback_called = []

        def on_max_exceeded(pause_seconds: int) -> None:
            callback_called.append(pause_seconds)

        ws = CoinbaseWebSocket(on_max_attempts_exceeded=on_max_exceeded)

        # Set up to exceed max attempts
        with (
            patch(
                "gpt_trader.features.brokerages.coinbase.ws.WS_RECONNECT_MAX_ATTEMPTS",
                3,
            ),
            patch(
                "gpt_trader.features.brokerages.coinbase.ws.WS_RECONNECT_PAUSE_SECONDS",
                300,
            ),
        ):
            ws._reconnect_count = 3  # At max
            ws._running.set()

            # Trigger close (will exceed max on increment)
            ws._on_close(None, 1000, "Connection lost")

        # Callback should have been called
        assert len(callback_called) == 1
        assert callback_called[0] == 300

    def test_callback_only_triggered_once(self) -> None:
        """Test that degradation callback is only triggered once per max-exceeded event."""
        callback_count = []

        def on_max_exceeded(pause_seconds: int) -> None:
            callback_count.append(1)

        ws = CoinbaseWebSocket(on_max_attempts_exceeded=on_max_exceeded)

        with patch(
            "gpt_trader.features.brokerages.coinbase.ws.WS_RECONNECT_MAX_ATTEMPTS",
            3,
        ):
            ws._reconnect_count = 3
            ws._running.set()

            # First close - should trigger callback
            ws._on_close(None, 1000, "Connection lost")

            # Second close - should NOT trigger callback again
            # Note: After max attempts triggered, _running is cleared
            # so we need to set it again to simulate another close
            ws._running.set()
            ws._shutdown.set()  # Prevent actual reconnection attempt
            ws._on_close(None, 1000, "Connection lost again")

        # Callback should only be called once
        assert len(callback_count) == 1

    def test_no_callback_when_none_provided(self) -> None:
        """Test that missing callback doesn't cause errors."""
        ws = CoinbaseWebSocket(on_max_attempts_exceeded=None)

        with patch(
            "gpt_trader.features.brokerages.coinbase.ws.WS_RECONNECT_MAX_ATTEMPTS",
            3,
        ):
            ws._reconnect_count = 3
            ws._running.set()

            # Should not raise even without callback
            ws._on_close(None, 1000, "Connection lost")

        # Verify max_attempts_triggered flag is set
        assert ws._max_attempts_triggered is True

    def test_callback_exception_does_not_crash(self) -> None:
        """Test that exception in callback is handled gracefully."""

        def bad_callback(pause_seconds: int) -> None:
            raise ValueError("Intentional test error")

        ws = CoinbaseWebSocket(on_max_attempts_exceeded=bad_callback)

        with patch(
            "gpt_trader.features.brokerages.coinbase.ws.WS_RECONNECT_MAX_ATTEMPTS",
            3,
        ):
            ws._reconnect_count = 3
            ws._running.set()

            # Should not raise even with bad callback
            ws._on_close(None, 1000, "Connection lost")

        # Should still set the triggered flag
        assert ws._max_attempts_triggered is True


class TestWebSocketHealthIncludesNewFields:
    """Tests that get_health includes the new reconnection fields."""

    def test_health_includes_connected_since(self) -> None:
        """Test that health dict includes connected_since field."""
        ws = CoinbaseWebSocket()
        ws._connected_since = 12345.0

        health = ws.get_health()

        assert "connected_since" in health
        assert health["connected_since"] == 12345.0

    def test_health_includes_max_attempts_triggered(self) -> None:
        """Test that health dict includes max_attempts_triggered field."""
        ws = CoinbaseWebSocket()
        ws._max_attempts_triggered = True

        health = ws.get_health()

        assert "max_attempts_triggered" in health
        assert health["max_attempts_triggered"] is True

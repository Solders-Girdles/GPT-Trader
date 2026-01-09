"""Tests for graceful degradation state management."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from gpt_trader.features.live_trade.degradation import DegradationState, PauseRecord


class TestPauseRecord:
    """Test PauseRecord dataclass."""

    def test_pause_record_creation(self) -> None:
        record = PauseRecord(until=time.time() + 60, reason="test", allow_reduce_only=True)
        assert record.reason == "test" and record.allow_reduce_only is True


class TestDegradationStatePauseAll:
    """Test global pause functionality."""

    def test_pause_all_sets_global_pause(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="test_reason")
        assert state._global_pause is not None
        assert state._global_pause.reason == "test_reason"
        assert state._global_pause.until > time.time()

    def test_is_paused_returns_true_during_global_pause(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="test")
        assert state.is_paused() is True and state.is_paused(symbol="BTC-USD") is True

    def test_is_paused_returns_false_after_expiry(self) -> None:
        state = DegradationState()
        state._global_pause = PauseRecord(until=time.time() - 1, reason="expired")
        assert state.is_paused() is False

    def test_global_pause_allows_reduce_only_when_configured(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="test", allow_reduce_only=True)
        assert state.is_paused(is_reduce_only=True) is False  # Allowed through
        assert state.is_paused(is_reduce_only=False) is True  # Blocked

    def test_global_pause_blocks_all_when_reduce_only_disabled(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="test", allow_reduce_only=False)
        assert (
            state.is_paused(is_reduce_only=True) is True
            and state.is_paused(is_reduce_only=False) is True
        )


class TestDegradationStatePauseSymbol:
    """Test symbol-specific pause functionality."""

    def test_pause_symbol_sets_symbol_pause(self) -> None:
        state = DegradationState()
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="test")
        assert (
            "BTC-USD" in state._symbol_pauses and state._symbol_pauses["BTC-USD"].reason == "test"
        )

    def test_symbol_pause_only_affects_that_symbol(self) -> None:
        state = DegradationState()
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="test")
        assert state.is_paused(symbol="BTC-USD") is True
        assert state.is_paused(symbol="ETH-USD") is False
        assert state.is_paused() is False  # Global is not paused

    def test_symbol_pause_allows_reduce_only_when_configured(self) -> None:
        state = DegradationState()
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="test", allow_reduce_only=True)
        assert state.is_paused(symbol="BTC-USD", is_reduce_only=True) is False
        assert state.is_paused(symbol="BTC-USD", is_reduce_only=False) is True

    def test_symbol_pause_expires(self) -> None:
        state = DegradationState()
        state._symbol_pauses["BTC-USD"] = PauseRecord(until=time.time() - 1, reason="expired")
        assert state.is_paused(symbol="BTC-USD") is False
        assert "BTC-USD" not in state._symbol_pauses  # Cleaned up


class TestGetPauseReason:
    """Test pause reason retrieval."""

    def test_returns_global_reason_when_globally_paused(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="global_test")
        assert state.get_pause_reason() == "global_test"

    def test_returns_symbol_reason_when_symbol_paused(self) -> None:
        state = DegradationState()
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="symbol_test")
        assert state.get_pause_reason(symbol="BTC-USD") == "symbol_test"

    def test_returns_none_when_not_paused(self) -> None:
        state = DegradationState()
        assert state.get_pause_reason() is None and state.get_pause_reason(symbol="BTC-USD") is None


class TestSlippageFailureTracking:
    """Test slippage failure tracking and pause logic."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        config = MagicMock()
        config.slippage_failure_pause_after = 3
        config.slippage_pause_seconds = 60
        return config

    def test_record_slippage_failure_increments_counter(self, mock_config: MagicMock) -> None:
        state = DegradationState()
        state.record_slippage_failure("BTC-USD", mock_config)
        assert state._slippage_failures["BTC-USD"] == 1

    def test_record_slippage_failure_triggers_pause_at_threshold(
        self, mock_config: MagicMock
    ) -> None:
        state = DegradationState()
        for _ in range(2):
            assert state.record_slippage_failure("BTC-USD", mock_config) is False
        assert state.record_slippage_failure("BTC-USD", mock_config) is True
        assert state.is_paused(symbol="BTC-USD") is True

    def test_slippage_counter_resets_after_pause(self, mock_config: MagicMock) -> None:
        state = DegradationState()
        for _ in range(3):
            state.record_slippage_failure("BTC-USD", mock_config)
        assert state._slippage_failures["BTC-USD"] == 0

    def test_reset_slippage_failures_clears_counter(self) -> None:
        state = DegradationState()
        state._slippage_failures["BTC-USD"] = 2
        state.reset_slippage_failures("BTC-USD")
        assert state._slippage_failures["BTC-USD"] == 0


class TestBrokerFailureTracking:
    """Test broker failure tracking and pause logic."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        config = MagicMock()
        config.broker_outage_max_failures = 3
        config.broker_outage_cooldown_seconds = 120
        return config

    def test_record_broker_failure_increments_counter(self, mock_config: MagicMock) -> None:
        state = DegradationState()
        state.record_broker_failure(mock_config)
        assert state._broker_failures == 1

    def test_record_broker_failure_triggers_global_pause(self, mock_config: MagicMock) -> None:
        state = DegradationState()
        for _ in range(2):
            assert state.record_broker_failure(mock_config) is False
        assert state.record_broker_failure(mock_config) is True
        assert state.is_paused() is True

    def test_broker_counter_resets_after_pause(self, mock_config: MagicMock) -> None:
        state = DegradationState()
        for _ in range(3):
            state.record_broker_failure(mock_config)
        assert state._broker_failures == 0

    def test_reset_broker_failures_clears_counter(self) -> None:
        state = DegradationState()
        state._broker_failures = 2
        state.reset_broker_failures()
        assert state._broker_failures == 0


class TestClearAllAndStatus:
    """Test clear_all and get_status functionality."""

    def test_clear_all_resets_state(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="global")
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="symbol")
        state._slippage_failures["BTC-USD"] = 2
        state._broker_failures = 1
        state.clear_all()
        assert state._global_pause is None
        assert len(state._symbol_pauses) == 0 and len(state._slippage_failures) == 0
        assert state._broker_failures == 0

    def test_get_status_returns_complete_state(self) -> None:
        state = DegradationState()
        state.pause_all(seconds=60, reason="global_test")
        state.pause_symbol(symbol="BTC-USD", seconds=30, reason="btc_test")
        state._slippage_failures["ETH-USD"] = 2
        state._broker_failures = 1
        status = state.get_status()
        assert status["global_paused"] is True and status["global_reason"] == "global_test"
        assert status["global_remaining_seconds"] > 0 and "BTC-USD" in status["paused_symbols"]
        assert status["slippage_failures"]["ETH-USD"] == 2 and status["broker_failures"] == 1

    def test_get_status_when_not_paused(self) -> None:
        state = DegradationState()
        status = state.get_status()
        assert status["global_paused"] is False and status["global_reason"] is None
        assert status["global_remaining_seconds"] == 0 and len(status["paused_symbols"]) == 0


class TestPauseMonotonicity:
    """Test that pauses are monotonic (only extend, never shorten)."""

    def test_global_pause_does_not_shorten_existing_pause(self) -> None:
        """A shorter pause request should not shorten an existing longer pause."""
        state = DegradationState()
        # Set a long pause
        state.pause_all(seconds=300, reason="long_pause")
        original_until = state._global_pause.until

        # Try to set a shorter pause
        state.pause_all(seconds=60, reason="short_pause")

        # Original pause should remain unchanged
        assert state._global_pause.until == original_until
        assert state._global_pause.reason == "long_pause"

    def test_global_pause_extends_existing_pause(self) -> None:
        """A longer pause request should extend an existing shorter pause."""
        state = DegradationState()
        # Set a short pause
        state.pause_all(seconds=60, reason="short_pause")
        original_until = state._global_pause.until

        # Set a longer pause
        state.pause_all(seconds=300, reason="long_pause")

        # Pause should be extended
        assert state._global_pause.until > original_until
        assert state._global_pause.reason == "long_pause"

    def test_symbol_pause_does_not_shorten_existing_pause(self) -> None:
        """A shorter symbol pause should not shorten an existing longer pause."""
        state = DegradationState()
        # Set a long pause for BTC-USD
        state.pause_symbol(symbol="BTC-USD", seconds=300, reason="long_pause")
        original_until = state._symbol_pauses["BTC-USD"].until

        # Try to set a shorter pause
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="short_pause")

        # Original pause should remain unchanged
        assert state._symbol_pauses["BTC-USD"].until == original_until
        assert state._symbol_pauses["BTC-USD"].reason == "long_pause"

    def test_symbol_pause_extends_existing_pause(self) -> None:
        """A longer symbol pause request should extend an existing shorter pause."""
        state = DegradationState()
        # Set a short pause
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="short_pause")
        original_until = state._symbol_pauses["BTC-USD"].until

        # Set a longer pause
        state.pause_symbol(symbol="BTC-USD", seconds=300, reason="long_pause")

        # Pause should be extended
        assert state._symbol_pauses["BTC-USD"].until > original_until
        assert state._symbol_pauses["BTC-USD"].reason == "long_pause"

    def test_symbol_pause_monotonicity_independent_per_symbol(self) -> None:
        """Monotonicity is enforced independently for each symbol."""
        state = DegradationState()
        # Set different pauses for different symbols
        state.pause_symbol(symbol="BTC-USD", seconds=300, reason="btc_long")
        state.pause_symbol(symbol="ETH-USD", seconds=60, reason="eth_short")

        btc_original = state._symbol_pauses["BTC-USD"].until
        eth_original = state._symbol_pauses["ETH-USD"].until

        # Try to shorten BTC but extend ETH
        state.pause_symbol(symbol="BTC-USD", seconds=60, reason="btc_shorter")
        state.pause_symbol(symbol="ETH-USD", seconds=300, reason="eth_longer")

        # BTC unchanged, ETH extended
        assert state._symbol_pauses["BTC-USD"].until == btc_original
        assert state._symbol_pauses["BTC-USD"].reason == "btc_long"
        assert state._symbol_pauses["ETH-USD"].until > eth_original
        assert state._symbol_pauses["ETH-USD"].reason == "eth_longer"


class TestGuardFailureTelemetry:
    """Test that guard failures emit telemetry correctly."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        config = MagicMock()
        config.slippage_failure_pause_after = 3
        config.slippage_pause_seconds = 60
        config.broker_outage_max_failures = 3
        config.broker_outage_cooldown_seconds = 120
        return config

    def test_slippage_failure_increments_once_per_call(self, mock_config: MagicMock) -> None:
        """Each slippage failure call increments counter exactly once."""
        state = DegradationState()
        state.record_slippage_failure("BTC-USD", mock_config)
        assert state._slippage_failures["BTC-USD"] == 1
        state.record_slippage_failure("BTC-USD", mock_config)
        assert state._slippage_failures["BTC-USD"] == 2

    def test_broker_failure_increments_once_per_call(self, mock_config: MagicMock) -> None:
        """Each broker failure call increments counter exactly once."""
        state = DegradationState()
        state.record_broker_failure(mock_config)
        assert state._broker_failures == 1
        state.record_broker_failure(mock_config)
        assert state._broker_failures == 2

    def test_slippage_pause_triggers_exactly_at_threshold(self, mock_config: MagicMock) -> None:
        """Slippage pause triggers exactly when threshold is reached, not before or after."""
        state = DegradationState()
        # First two should not trigger
        result1 = state.record_slippage_failure("BTC-USD", mock_config)
        result2 = state.record_slippage_failure("BTC-USD", mock_config)
        assert result1 is False and result2 is False
        assert state.is_paused(symbol="BTC-USD") is False

        # Third should trigger exactly once
        result3 = state.record_slippage_failure("BTC-USD", mock_config)
        assert result3 is True
        assert state.is_paused(symbol="BTC-USD") is True

        # Counter should reset after pause trigger
        assert state._slippage_failures["BTC-USD"] == 0

    def test_broker_pause_triggers_exactly_at_threshold(self, mock_config: MagicMock) -> None:
        """Broker pause triggers exactly when threshold is reached."""
        state = DegradationState()
        # First two should not trigger
        result1 = state.record_broker_failure(mock_config)
        result2 = state.record_broker_failure(mock_config)
        assert result1 is False and result2 is False
        assert state.is_paused() is False

        # Third should trigger exactly once
        result3 = state.record_broker_failure(mock_config)
        assert result3 is True
        assert state.is_paused() is True

        # Counter should reset after pause trigger
        assert state._broker_failures == 0

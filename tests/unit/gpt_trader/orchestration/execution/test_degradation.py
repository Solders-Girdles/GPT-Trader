"""Tests for graceful degradation state management."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from gpt_trader.orchestration.execution.degradation import DegradationState, PauseRecord


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

"""Tests for degradation pause expiry and recovery sequences."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from gpt_trader.features.live_trade.degradation import DegradationState, PauseRecord


class TestPauseExpiryRecovery:
    """Test that pauses expire correctly and trading resumes."""

    def test_global_pause_expires_and_trading_resumes(self) -> None:
        """Test that global pause expires and is_paused returns False."""
        state = DegradationState()
        state._global_pause = PauseRecord(
            until=time.time() - 1.0,
            reason="expired_test",
            allow_reduce_only=False,
        )

        assert state.is_paused() is False
        assert state.is_paused(symbol="BTC-USD") is False

        assert state._global_pause is None

    def test_symbol_pause_expires_and_trading_resumes(self) -> None:
        """Test that symbol-specific pause expires correctly."""
        state = DegradationState()
        state._symbol_pauses["BTC-USD"] = PauseRecord(
            until=time.time() - 1.0,
            reason="expired_test",
            allow_reduce_only=False,
        )

        assert state.is_paused(symbol="BTC-USD") is False
        assert state.is_paused(symbol="ETH-USD") is False

        assert "BTC-USD" not in state._symbol_pauses

    def test_reduce_only_allowed_during_pause(self) -> None:
        """Test that reduce-only orders are allowed when configured."""
        state = DegradationState()
        state.pause_all(
            seconds=60,
            reason="allow_reduce_only_test",
            allow_reduce_only=True,
        )

        assert state.is_paused(is_reduce_only=False) is True
        assert state.is_paused(is_reduce_only=True) is False

    def test_pause_then_recovery_sequence(self) -> None:
        """Test a complete pause -> recovery sequence."""
        state = DegradationState()

        assert state.is_paused() is False

        state.pause_all(seconds=10, reason="test_pause")
        assert state.is_paused() is True
        assert state.get_pause_reason() == "test_pause"

        state._global_pause = PauseRecord(
            until=time.time() - 0.1,
            reason="test_pause",
            allow_reduce_only=False,
        )

        assert state.is_paused() is False
        assert state.get_pause_reason() is None

    def test_multiple_symbol_pauses_expire_independently(self) -> None:
        """Test that each symbol's pause expires independently."""
        state = DegradationState()

        state._symbol_pauses["BTC-USD"] = PauseRecord(
            until=time.time() - 1.0,
            reason="btc_expired",
            allow_reduce_only=False,
        )
        state._symbol_pauses["ETH-USD"] = PauseRecord(
            until=time.time() + 60.0,
            reason="eth_active",
            allow_reduce_only=False,
        )

        assert state.is_paused(symbol="BTC-USD") is False
        assert "BTC-USD" not in state._symbol_pauses

        assert state.is_paused(symbol="ETH-USD") is True
        assert state.get_pause_reason(symbol="ETH-USD") == "eth_active"

    def test_guard_error_triggers_pause_via_record_broker_failure(self) -> None:
        """Test that guard/broker errors trigger pause when threshold reached."""
        state = DegradationState()
        mock_config = MagicMock()
        mock_config.broker_outage_max_failures = 2
        mock_config.broker_outage_cooldown_seconds = 60

        triggered = state.record_broker_failure(mock_config)
        assert triggered is False
        assert state.is_paused() is False

        triggered = state.record_broker_failure(mock_config)
        assert triggered is True
        assert state.is_paused() is True
        assert state._broker_failures == 0

    def test_pause_allows_trading_after_cooldown(self) -> None:
        """Test that trading is allowed after cooldown period."""
        state = DegradationState()
        mock_config = MagicMock()
        mock_config.broker_outage_max_failures = 1
        mock_config.broker_outage_cooldown_seconds = 30

        state.record_broker_failure(mock_config)
        assert state.is_paused() is True

        state._global_pause = PauseRecord(
            until=time.time() - 1.0,
            reason="broker_outage",
            allow_reduce_only=True,
        )

        assert state.is_paused() is False

        state.record_broker_failure(mock_config)
        assert state.is_paused() is True

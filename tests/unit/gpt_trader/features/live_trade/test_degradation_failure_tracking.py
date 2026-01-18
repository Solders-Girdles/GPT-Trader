"""Tests for degradation failure tracking (slippage + broker outage)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.features.live_trade.degradation import DegradationState


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
        result1 = state.record_slippage_failure("BTC-USD", mock_config)
        result2 = state.record_slippage_failure("BTC-USD", mock_config)
        assert result1 is False and result2 is False
        assert state.is_paused(symbol="BTC-USD") is False

        result3 = state.record_slippage_failure("BTC-USD", mock_config)
        assert result3 is True
        assert state.is_paused(symbol="BTC-USD") is True

        assert state._slippage_failures["BTC-USD"] == 0

    def test_broker_pause_triggers_exactly_at_threshold(self, mock_config: MagicMock) -> None:
        """Broker pause triggers exactly when threshold is reached."""
        state = DegradationState()
        result1 = state.record_broker_failure(mock_config)
        result2 = state.record_broker_failure(mock_config)
        assert result1 is False and result2 is False
        assert state.is_paused() is False

        result3 = state.record_broker_failure(mock_config)
        assert result3 is True
        assert state.is_paused() is True

        assert state._broker_failures == 0

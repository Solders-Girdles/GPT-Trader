"""Tests for BrokerExecutor initialization."""

from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor


class TestBrokerExecutorInit:
    """Tests for BrokerExecutor initialization."""

    def test_init_stores_broker(self, mock_broker: MagicMock) -> None:
        """Test that broker is stored correctly."""
        executor = BrokerExecutor(broker=mock_broker)
        assert executor._broker is mock_broker

    def test_init_defaults_integration_mode_false(self, mock_broker: MagicMock) -> None:
        """Test that integration_mode defaults to False."""
        executor = BrokerExecutor(broker=mock_broker)
        assert executor._integration_mode is False

    def test_init_accepts_integration_mode_true(self, mock_broker: MagicMock) -> None:
        """Test that integration_mode can be set to True."""
        executor = BrokerExecutor(broker=mock_broker, integration_mode=True)
        assert executor._integration_mode is True

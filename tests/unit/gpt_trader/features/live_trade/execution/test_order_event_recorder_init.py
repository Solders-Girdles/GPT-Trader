"""Tests for OrderEventRecorder initialization."""

from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


class TestOrderEventRecorderInit:
    """Tests for OrderEventRecorder initialization."""

    def test_init_stores_event_store(self, mock_event_store: MagicMock) -> None:
        """Test that event_store is stored correctly."""
        recorder = OrderEventRecorder(event_store=mock_event_store, bot_id="test-bot")
        assert recorder._event_store is mock_event_store

    def test_init_stores_bot_id(self, mock_event_store: MagicMock) -> None:
        """Test that bot_id is stored correctly."""
        recorder = OrderEventRecorder(event_store=mock_event_store, bot_id="my-bot-id")
        assert recorder._bot_id == "my-bot-id"

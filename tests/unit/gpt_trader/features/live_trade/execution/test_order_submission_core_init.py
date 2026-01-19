"""Core unit tests for OrderSubmitter initialization."""

from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


class TestOrderSubmitterInit:
    """Tests for OrderSubmitter initialization."""

    def test_init_stores_dependencies(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that dependencies are stored correctly."""
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            integration_mode=True,
        )

        assert submitter.broker is mock_broker
        assert submitter.event_store is mock_event_store
        assert submitter.bot_id == "test-bot"
        assert submitter.open_orders is open_orders
        assert submitter.integration_mode is True

    def test_init_defaults_integration_mode_to_false(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that integration_mode defaults to False."""
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        assert submitter.integration_mode is False

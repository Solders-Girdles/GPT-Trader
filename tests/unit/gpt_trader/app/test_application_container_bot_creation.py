"""Unit tests for ApplicationContainer bot creation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import ApplicationContainer


class TestApplicationContainerBotCreation:
    """Test cases for ApplicationContainer.create_bot()."""

    @patch("gpt_trader.features.live_trade.bot.TradingBot")
    def test_create_bot(self, mock_bot_class: MagicMock, mock_config: BotConfig) -> None:
        """Test that TradingBot is created correctly from container."""
        from gpt_trader.app.containers.brokerage import BrokerageContainer

        mock_broker = MagicMock()
        mock_bot = MagicMock()
        mock_create_brokerage = MagicMock()
        mock_create_brokerage.return_value = (
            mock_broker,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_bot_class.return_value = mock_bot

        container = ApplicationContainer(mock_config)
        container._brokerage = BrokerageContainer(
            config=mock_config,
            event_store_provider=lambda: container.event_store,
            broker_factory=mock_create_brokerage,
        )

        bot = container.create_bot()

        mock_bot_class.assert_called_once()
        call_args = mock_bot_class.call_args

        assert call_args.kwargs["config"] == mock_config
        assert call_args.kwargs["container"] == container
        assert call_args.kwargs["event_store"] == container.event_store
        assert call_args.kwargs["orders_store"] == container.orders_store

        assert bot == mock_bot

    @patch("gpt_trader.features.live_trade.bot.TradingBot")
    def test_create_bot_includes_notification_service(
        self, mock_bot_class: MagicMock, mock_config: BotConfig
    ) -> None:
        """Test that TradingBot is created with notification service."""
        from gpt_trader.app.containers.brokerage import BrokerageContainer

        mock_broker = MagicMock()
        mock_bot = MagicMock()
        mock_create_brokerage = MagicMock()
        mock_create_brokerage.return_value = (
            mock_broker,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_bot_class.return_value = mock_bot

        container = ApplicationContainer(mock_config)
        container._brokerage = BrokerageContainer(
            config=mock_config,
            event_store_provider=lambda: container.event_store,
            broker_factory=mock_create_brokerage,
        )

        _ = container.create_bot()

        mock_bot_class.assert_called_once()
        call_args = mock_bot_class.call_args

        assert call_args.kwargs["notification_service"] is not None
        assert call_args.kwargs["notification_service"] == container.notification_service

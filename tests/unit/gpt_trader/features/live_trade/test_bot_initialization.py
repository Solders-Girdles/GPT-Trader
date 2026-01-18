"""Tests for TradingBot initialization."""

from __future__ import annotations

from unittest.mock import ANY, Mock, patch

import pytest

from gpt_trader.features.live_trade.bot import TradingBot


class TestTradingBotInitialization:
    """Test TradingBot initialization."""

    @pytest.fixture
    def mock_config(self) -> Mock:
        """Create a mock BotConfig."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC", "ETH-PERP-USDC"]
        config.interval = 60
        return config

    def test_init_with_config_only(self, mock_config: Mock) -> None:
        """Test initialization with only config."""
        mock_container = Mock()
        # Set container attributes to None for this test
        mock_container.broker = None
        mock_container.risk_manager = None
        mock_container.event_store = Mock()
        mock_container.orders_store = Mock()
        mock_container.notification_service = Mock()

        mock_container.account_manager = Mock()
        mock_container.account_telemetry = Mock()
        mock_container.runtime_state = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config, container=mock_container)

        assert bot.broker is mock_container.broker
        assert bot.account_manager is mock_container.account_manager
        assert bot.account_telemetry is mock_container.account_telemetry
        assert bot.risk_manager is mock_container.risk_manager
        assert bot.runtime_state is mock_container.runtime_state

    def test_init_with_container(self, mock_config: Mock) -> None:
        """Test initialization with container."""
        mock_container = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config, container=mock_container)

        assert bot.container is mock_container

    def test_init_creates_context(self, mock_config: Mock) -> None:
        """Test that initialization creates CoordinatorContext."""
        mock_container = Mock()
        mock_container.broker = Mock()
        mock_container.risk_manager = Mock()
        mock_container.event_store = Mock()
        mock_container.orders_store = Mock()
        mock_container.notification_service = Mock()

        with (
            patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine,
            patch("gpt_trader.features.live_trade.bot.CoordinatorContext") as mock_context,
        ):
            mock_engine.return_value = Mock()
            mock_context.return_value = Mock()

            bot = TradingBot(config=mock_config, container=mock_container)

            mock_context.assert_called_once_with(
                config=mock_config,
                container=mock_container,
                broker=mock_container.broker,
                broker_calls=ANY,
                symbols=tuple(mock_config.symbols),
                risk_manager=mock_container.risk_manager,
                event_store=mock_container.event_store,
                orders_store=mock_container.orders_store,
                notification_service=mock_container.notification_service,
            )
            assert bot.context is not None

    def test_init_creates_engine(self, mock_config: Mock) -> None:
        """Test that initialization creates TradingEngine."""
        mock_container = Mock()
        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config, container=mock_container)

            assert bot.engine is not None
            mock_engine.assert_called_once()

    def test_init_container_missing_optional_attributes(self, mock_config: Mock) -> None:
        """Test initialization handles container with missing optional attributes."""
        from types import SimpleNamespace

        # Use SimpleNamespace with only required attributes
        mock_container = SimpleNamespace(
            broker=Mock(),
            risk_manager=Mock(),
            event_store=Mock(),
            orders_store=None,
            notification_service=Mock(),
            # Optional attributes not set - getattr will return None
        )

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config, container=mock_container)

        assert bot.broker is mock_container.broker
        assert bot.account_manager is None  # Not set in SimpleNamespace
        assert bot.account_telemetry is None  # Not set in SimpleNamespace
        assert bot.runtime_state is None  # Not set in SimpleNamespace

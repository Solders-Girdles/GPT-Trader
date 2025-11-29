"""
Unit tests for the application container.
"""

from __future__ import annotations

from unittest.mock import ANY, MagicMock, patch

import pytest

from gpt_trader.app.container import ApplicationContainer, create_application_container
from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import OrdersStore


class TestApplicationContainer:
    """Test cases for ApplicationContainer."""

    def test_container_initialization(self, mock_config: BotConfig) -> None:
        """Test that container initializes correctly with config."""
        container = ApplicationContainer(mock_config)

        assert container.config == mock_config
        assert container._config_controller is None
        assert container._runtime_paths is None
        assert container._broker is None
        assert container._event_store is None
        assert container._orders_store is None
        assert container._market_data_service is None
        assert container._product_catalog is None
        assert container._risk_manager is None
        assert container._notification_service is None

    def test_config_controller_creation(self, mock_config: BotConfig) -> None:
        """Test that config controller is created correctly."""
        container = ApplicationContainer(mock_config)

        # First access should create the controller
        config_controller = container.config_controller

        assert config_controller is not None
        assert config_controller.current == mock_config
        assert container._config_controller == config_controller

        # Second access should return the same instance
        config_controller2 = container.config_controller
        assert config_controller is config_controller2

    def test_event_store_creation(self, mock_config: BotConfig) -> None:
        """Test that event store is created correctly."""
        container = ApplicationContainer(mock_config)

        # First access should create the event store
        event_store = container.event_store

        assert isinstance(event_store, EventStore)
        assert container._event_store == event_store

        # Second access should return the same instance
        event_store2 = container.event_store
        assert event_store is event_store2

    def test_orders_store_creation(self, mock_config: BotConfig) -> None:
        """Test that orders store is created correctly."""
        container = ApplicationContainer(mock_config)

        # First access should create the orders store
        orders_store = container.orders_store

        assert isinstance(orders_store, OrdersStore)
        assert container._orders_store == orders_store

        # Second access should return the same instance
        orders_store2 = container.orders_store
        assert orders_store is orders_store2

    @patch("gpt_trader.app.container.MarketDataService")
    def test_market_data_service_creation(
        self, mock_market_data_service: MagicMock, mock_config: BotConfig
    ) -> None:
        """Test that market data service is created correctly."""
        mock_instance = MagicMock()
        mock_market_data_service.return_value = mock_instance

        container = ApplicationContainer(mock_config)

        # First access should create the market data service
        market_data_service = container.market_data_service

        assert market_data_service == mock_instance
        assert container._market_data_service == market_data_service
        mock_market_data_service.assert_called_once()

        # Second access should return the same instance
        market_data_service2 = container.market_data_service
        assert market_data_service is market_data_service2

    @patch("gpt_trader.app.container.ProductCatalog")
    def test_product_catalog_creation(
        self, mock_product_catalog: MagicMock, mock_config: BotConfig
    ) -> None:
        """Test that product catalog is created correctly."""
        mock_instance = MagicMock()
        mock_product_catalog.return_value = mock_instance

        container = ApplicationContainer(mock_config)

        # First access should create the product catalog
        product_catalog = container.product_catalog

        assert product_catalog == mock_instance
        assert container._product_catalog == product_catalog
        mock_product_catalog.assert_called_once_with()

        # Second access should return the same instance
        product_catalog2 = container.product_catalog
        assert product_catalog is product_catalog2

    @patch("gpt_trader.app.container.create_brokerage")
    @patch("gpt_trader.app.container.MarketDataService")
    @patch("gpt_trader.app.container.ProductCatalog")
    def test_broker_creation(
        self,
        mock_product_catalog: MagicMock,
        mock_market_data_service: MagicMock,
        mock_create_brokerage: MagicMock,
        mock_config: BotConfig,
    ) -> None:
        """Test that broker is created correctly with dependencies."""
        # Setup mocks
        mock_broker = MagicMock()
        mock_market_data_instance = MagicMock()
        mock_product_catalog_instance = MagicMock()

        mock_create_brokerage.return_value = (
            mock_broker,
            MagicMock(),  # event_store return (ignored by container)
            MagicMock(),  # market_data return (ignored by container)
            MagicMock(),  # product_catalog return (ignored by container)
        )
        mock_market_data_service.return_value = mock_market_data_instance
        mock_product_catalog.return_value = mock_product_catalog_instance

        container = ApplicationContainer(mock_config)

        # Access broker to trigger creation
        broker = container.broker

        assert broker == mock_broker
        assert container._broker == broker

        # Verify create_brokerage was called with correct dependencies
        mock_create_brokerage.assert_called_once_with(
            event_store=ANY,
            market_data=mock_market_data_instance,
            product_catalog=mock_product_catalog_instance,
            config=mock_config,
        )

    @patch("gpt_trader.app.container.create_brokerage")
    @patch("gpt_trader.app.container.MarketDataService")
    @patch("gpt_trader.app.container.ProductCatalog")
    def test_create_service_registry(
        self,
        mock_product_catalog: MagicMock,
        mock_market_data_service: MagicMock,
        mock_create_brokerage: MagicMock,
        mock_config: BotConfig,
    ) -> None:
        """Test that service registry is created correctly."""
        # Setup mocks
        mock_broker = MagicMock()
        mock_market_data_instance = MagicMock()
        mock_product_catalog_instance = MagicMock()

        mock_create_brokerage.return_value = (
            mock_broker,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_market_data_service.return_value = mock_market_data_instance
        mock_product_catalog.return_value = mock_product_catalog_instance

        container = ApplicationContainer(mock_config)

        # Create service registry
        registry = container.create_service_registry()

        # Verify registry contains correct services
        assert registry.config == mock_config
        assert registry.broker == mock_broker
        assert registry.event_store == container.event_store
        assert registry.orders_store == container.orders_store
        assert registry.market_data_service == mock_market_data_instance
        assert registry.product_catalog == mock_product_catalog_instance

    @patch("gpt_trader.orchestration.trading_bot.bot.TradingBot")
    @patch("gpt_trader.app.container.create_brokerage")
    @patch("gpt_trader.app.container.MarketDataService")
    @patch("gpt_trader.app.container.ProductCatalog")
    def test_create_bot(
        self,
        mock_product_catalog: MagicMock,
        mock_market_data_service: MagicMock,
        mock_create_brokerage: MagicMock,
        mock_bot_class: MagicMock,
        mock_config: BotConfig,
    ) -> None:
        """Test that TradingBot is created correctly from container."""
        # Setup mocks
        mock_broker = MagicMock()
        mock_bot = MagicMock()

        mock_create_brokerage.return_value = (
            mock_broker,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_bot_class.return_value = mock_bot

        container = ApplicationContainer(mock_config)

        # Create bot
        bot = container.create_bot()

        # Verify bot was created with correct dependencies
        mock_bot_class.assert_called_once()
        call_args = mock_bot_class.call_args

        assert call_args.kwargs["config"] == mock_config
        assert call_args.kwargs["container"] == container
        assert call_args.kwargs["registry"] is not None
        assert call_args.kwargs["event_store"] == container.event_store
        assert call_args.kwargs["orders_store"] == container.orders_store

        assert bot == mock_bot

    @patch("gpt_trader.orchestration.trading_bot.bot.TradingBot")
    @patch("gpt_trader.app.container.create_brokerage")
    @patch("gpt_trader.app.container.MarketDataService")
    @patch("gpt_trader.app.container.ProductCatalog")
    def test_create_bot_includes_notification_service(
        self,
        mock_product_catalog: MagicMock,
        mock_market_data_service: MagicMock,
        mock_create_brokerage: MagicMock,
        mock_bot_class: MagicMock,
        mock_config: BotConfig,
    ) -> None:
        """Test that TradingBot is created with notification service."""
        # Setup mocks
        mock_broker = MagicMock()
        mock_bot = MagicMock()

        mock_create_brokerage.return_value = (
            mock_broker,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_bot_class.return_value = mock_bot

        container = ApplicationContainer(mock_config)

        # Create bot
        _ = container.create_bot()

        # Verify bot was created with notification service
        mock_bot_class.assert_called_once()
        call_args = mock_bot_class.call_args

        assert call_args.kwargs["notification_service"] is not None
        assert call_args.kwargs["notification_service"] == container.notification_service

    @patch("gpt_trader.app.container.create_brokerage")
    @patch("gpt_trader.app.container.MarketDataService")
    @patch("gpt_trader.app.container.ProductCatalog")
    def test_reset_broker(
        self,
        mock_product_catalog: MagicMock,
        mock_market_data_service: MagicMock,
        mock_create_brokerage: MagicMock,
        mock_config: BotConfig,
    ) -> None:
        """Test that broker can be reset."""
        mock_broker = MagicMock()
        mock_create_brokerage.return_value = (
            mock_broker,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        container = ApplicationContainer(mock_config)

        # Access broker to create it
        broker = container.broker
        assert broker == mock_broker
        assert container._broker == mock_broker

        # Reset broker
        container.reset_broker()
        assert container._broker is None

        # Access again to recreate
        broker2 = container.broker
        assert broker2 == mock_broker  # Same mock instance
        assert container._broker == mock_broker

        # Verify create_brokerage was called twice
        assert mock_create_brokerage.call_count == 2

    def test_reset_config(self, mock_config: BotConfig) -> None:
        """Test that config controller can be reset."""
        container = ApplicationContainer(mock_config)

        # Access config controller to create it
        config_controller = container.config_controller
        assert config_controller is not None
        assert container._config_controller == config_controller

        # Reset config controller
        container.reset_config()
        assert container._config_controller is None

        # Access again to recreate
        config_controller2 = container.config_controller
        assert config_controller2 is not None
        assert config_controller2 is not config_controller  # New instance


class TestCreateApplicationContainer:
    """Test cases for create_application_container function."""

    def test_create_application_container(self, mock_config: BotConfig) -> None:
        """Test that application container is created correctly."""
        container = create_application_container(mock_config)

        assert isinstance(container, ApplicationContainer)
        assert container.config == mock_config


@pytest.fixture
def mock_config() -> BotConfig:
    """Create a mock BotConfig for testing."""
    return BotConfig(
        symbols=["BTC-USD"],
    )

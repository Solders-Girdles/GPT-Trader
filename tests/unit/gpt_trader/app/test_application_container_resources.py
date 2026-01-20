"""Unit tests for ApplicationContainer resource construction."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gpt_trader.app.containers.brokerage as brokerage_module
from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import ApplicationContainer
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import OrdersStore


class TestApplicationContainerResources:
    """Test cases for ApplicationContainer lazy resource wiring."""

    def test_container_initialization(self, mock_config: BotConfig) -> None:
        """Test that container initializes correctly with config."""
        container = ApplicationContainer(mock_config)

        assert container.config == mock_config
        assert container._config_container._config_controller is None
        assert container._persistence._runtime_paths is None
        assert container._brokerage._broker is None
        assert container._persistence._event_store is None
        assert container._persistence._orders_store is None
        assert container._brokerage._market_data_service is None
        assert container._brokerage._product_catalog is None
        assert container._risk_validation._risk_manager is None
        assert container._observability._notification_service is None
        assert container._risk_validation._validation_failure_tracker is None
        assert container._config_container._profile_loader is None

    def test_config_controller_creation(self, mock_config: BotConfig) -> None:
        """Test that config controller is created correctly."""
        container = ApplicationContainer(mock_config)

        config_controller = container.config_controller

        assert config_controller is not None
        assert config_controller.current == mock_config
        assert container._config_container._config_controller == config_controller

        config_controller2 = container.config_controller
        assert config_controller is config_controller2

    def test_event_store_creation(self, mock_config: BotConfig) -> None:
        """Test that event store is created correctly."""
        container = ApplicationContainer(mock_config)

        event_store = container.event_store

        assert isinstance(event_store, EventStore)
        assert container._persistence._event_store == event_store

        event_store2 = container.event_store
        assert event_store is event_store2

    def test_orders_store_creation(self, mock_config: BotConfig) -> None:
        """Test that orders store is created correctly."""
        container = ApplicationContainer(mock_config)

        orders_store = container.orders_store

        assert isinstance(orders_store, OrdersStore)
        assert container._persistence._orders_store == orders_store

        orders_store2 = container.orders_store
        assert orders_store is orders_store2

    def test_market_data_service_creation(
        self, mock_config: BotConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that market data service is created correctly."""
        mock_instance = MagicMock()
        mock_market_data_service = MagicMock(return_value=mock_instance)
        monkeypatch.setattr(brokerage_module, "MarketDataService", mock_market_data_service)

        container = ApplicationContainer(mock_config)

        market_data_service = container.market_data_service

        assert market_data_service == mock_instance
        assert container._brokerage._market_data_service == market_data_service
        mock_market_data_service.assert_called_once()

        market_data_service2 = container.market_data_service
        assert market_data_service is market_data_service2

    def test_product_catalog_creation(
        self, mock_config: BotConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that product catalog is created correctly."""
        mock_instance = MagicMock()
        mock_product_catalog = MagicMock(return_value=mock_instance)
        monkeypatch.setattr(brokerage_module, "ProductCatalog", mock_product_catalog)

        container = ApplicationContainer(mock_config)

        product_catalog = container.product_catalog

        assert product_catalog == mock_instance
        assert container._brokerage._product_catalog == product_catalog
        mock_product_catalog.assert_called_once_with()

        product_catalog2 = container.product_catalog
        assert product_catalog is product_catalog2

    def test_broker_creation(self, mock_config: BotConfig) -> None:
        """Test that broker is created correctly with dependencies."""
        from gpt_trader.app.containers.brokerage import BrokerageContainer

        mock_broker = MagicMock()
        mock_create_brokerage = MagicMock()
        mock_create_brokerage.return_value = (
            mock_broker,
            MagicMock(),  # event_store return (ignored by container)
            MagicMock(),  # market_data return (ignored by container)
            MagicMock(),  # product_catalog return (ignored by container)
        )

        container = ApplicationContainer(mock_config)
        container._brokerage = BrokerageContainer(
            config=mock_config,
            event_store_provider=lambda: container.event_store,
            broker_factory=mock_create_brokerage,
        )

        broker = container.broker

        assert broker == mock_broker
        assert container._brokerage._broker == broker
        mock_create_brokerage.assert_called_once()

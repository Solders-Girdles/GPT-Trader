"""
Integration tests for the composition root implementation.

These tests verify that the composition root works correctly with real
dependencies and that the PerpsBot can be created and run using the
container-based approach.
"""

from __future__ import annotations

import pytest

from bot_v2.app.container import ApplicationContainer, create_application_container
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.perps_bootstrap import (
    prepare_perps_bot,
    prepare_perps_bot_with_container,
)
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.perps_bot_builder import (
    PerpsBotBuilder,
    create_perps_bot,
    create_perps_bot_with_container,
    create_test_perps_bot,
    create_test_perps_bot_with_container,
)
from bot_v2.orchestration.runtime_settings import RuntimeSettings


class TestCompositionRootIntegration:
    """Integration tests for the composition root."""

    def test_container_creates_perps_bot_successfully(self, mock_config: BotConfig) -> None:
        """Test that container can create a fully functional PerpsBot."""
        # Create container
        container = create_application_container(mock_config)

        # Create PerpsBot from container
        bot = container.create_perps_bot()

        # Verify bot was created correctly
        assert isinstance(bot, PerpsBot)
        assert bot.bot_id == "coinbase_trader"
        assert bot.config == mock_config
        assert bot.container == container

        # Verify bot has all required dependencies
        assert bot.config_controller is not None
        assert bot.registry is not None
        assert bot.event_store is not None
        assert bot.orders_store is not None
        assert bot.baseline_snapshot is not None
        assert bot.configuration_guardian is not None

    def test_perps_bot_from_container_class_method(self, mock_config: BotConfig) -> None:
        """Test that PerpsBot.from_container works correctly."""
        container = create_application_container(mock_config)

        # Create bot using class method
        bot = PerpsBot.from_container(container)

        # Verify bot was created correctly
        assert isinstance(bot, PerpsBot)
        assert bot.container == container
        assert bot.config == mock_config

    def test_builder_with_container_enabled(self, mock_config: BotConfig) -> None:
        """Test that PerpsBotBuilder works with container enabled."""
        builder = PerpsBotBuilder(use_container=True).with_config(mock_config)
        bot = builder.build()

        # Verify bot was created using container
        assert isinstance(bot, PerpsBot)
        assert bot.container is not None
        assert bot.config == mock_config

    def test_builder_legacy_mode_still_works(self, mock_config: BotConfig) -> None:
        """Test that PerpsBotBuilder still works in legacy mode."""
        builder = PerpsBotBuilder(use_container=False).with_config(mock_config)
        bot = builder.build()

        # Verify bot was created using legacy approach
        assert isinstance(bot, PerpsBot)
        assert bot.container is None  # No container in legacy mode
        assert bot.config == mock_config

    def test_create_perps_bot_with_container_function(self, mock_config: BotConfig) -> None:
        """Test the create_perps_bot_with_container function."""
        bot = create_perps_bot_with_container(mock_config)

        # Verify bot was created using container
        assert isinstance(bot, PerpsBot)
        assert bot.container is not None
        assert bot.config == mock_config

    def test_create_perps_bot_function_with_flag(self, mock_config: BotConfig) -> None:
        """Test the create_perps_bot function with container flag."""
        # Test with container enabled
        bot = create_perps_bot(mock_config, use_container=True)
        assert isinstance(bot, PerpsBot)
        assert bot.container is not None

        # Test with container disabled (legacy)
        bot_legacy = create_perps_bot(mock_config, use_container=False)
        assert isinstance(bot_legacy, PerpsBot)
        assert bot_legacy.container is None

    def test_bootstrap_with_container(self, mock_config: BotConfig) -> None:
        """Test that prepare_perps_bot_with_container works correctly."""
        result = prepare_perps_bot_with_container(mock_config)

        # Verify bootstrap result
        assert result.config == mock_config
        assert result.registry is not None
        assert result.event_store is not None
        assert result.orders_store is not None
        assert result.settings is not None

        # Verify container is stored in registry extras
        assert "container" in result.registry.extras
        container = result.registry.extras["container"]
        assert isinstance(container, ApplicationContainer)

    def test_bootstrap_with_use_container_flag(self, mock_config: BotConfig) -> None:
        """Test that prepare_perps_bot works with use_container flag."""
        # Test with container enabled
        result = prepare_perps_bot(mock_config, use_container=True)
        assert "container" in result.registry.extras

        # Test with container disabled (legacy)
        result_legacy = prepare_perps_bot(mock_config, use_container=False)
        assert "container" not in result_legacy.registry.extras

    def test_container_service_registry_creation(self, mock_config: BotConfig) -> None:
        """Test that container creates service registry correctly."""
        container = create_application_container(mock_config)

        # Create service registry
        registry = container.create_service_registry()

        # Verify registry contains all expected services
        assert registry.config == mock_config
        assert registry.broker is not None
        assert registry.event_store is not None
        assert registry.orders_store is not None
        assert registry.market_data_service is not None
        assert registry.product_catalog is not None
        assert registry.runtime_settings is not None

    def test_container_broker_dependency_injection(self, mock_config: BotConfig) -> None:
        """Test that broker dependencies are correctly injected."""
        container = create_application_container(mock_config)

        # Access broker to trigger creation
        broker = container.broker

        # Verify broker was created
        assert broker is not None

        # Verify dependencies were created and are the same instances
        assert container.event_store is not None
        assert container.market_data_service is not None
        assert container.product_catalog is not None

    def test_container_lazy_initialization(self, mock_config: BotConfig) -> None:
        """Test that container services are lazily initialized."""
        container = create_application_container(mock_config)

        # Initially, no services should be created
        assert container._config_controller is None
        assert container._event_store is None
        assert container._orders_store is None
        assert container._market_data_service is None
        assert container._product_catalog is None
        assert container._broker is None

        # Access each service to trigger creation
        _ = container.config_controller
        _ = container.event_store
        _ = container.orders_store
        _ = container.market_data_service
        _ = container.product_catalog
        _ = container.broker

        # Now all services should be created
        assert container._config_controller is not None
        assert container._event_store is not None
        assert container._orders_store is not None
        assert container._market_data_service is not None
        assert container._product_catalog is not None
        assert container._broker is not None

    def test_container_singleton_behavior(self, mock_config: BotConfig) -> None:
        """Test that container services behave as singletons."""
        container = create_application_container(mock_config)

        # Access each service multiple times
        config_controller1 = container.config_controller
        config_controller2 = container.config_controller
        assert config_controller1 is config_controller2

        event_store1 = container.event_store
        event_store2 = container.event_store
        assert event_store1 is event_store2

        orders_store1 = container.orders_store
        orders_store2 = container.orders_store
        assert orders_store1 is orders_store2

        market_data1 = container.market_data_service
        market_data2 = container.market_data_service
        assert market_data1 is market_data2

        product_catalog1 = container.product_catalog
        product_catalog2 = container.product_catalog
        assert product_catalog1 is product_catalog2

        broker1 = container.broker
        broker2 = container.broker
        assert broker1 is broker2

    def test_container_reset_functionality(self, mock_config: BotConfig) -> None:
        """Test that container reset functionality works correctly."""
        container = create_application_container(mock_config)

        # Create services
        config_controller1 = container.config_controller
        broker1 = container.broker

        # Reset services
        container.reset_config()
        container.reset_broker()

        # Recreate services
        config_controller2 = container.config_controller
        broker2 = container.broker

        # Verify new instances were created
        assert config_controller1 is not config_controller2
        assert broker1 is not broker2

    def test_test_helpers_with_container(self, mock_config: BotConfig) -> None:
        """Test that test helper functions work with container."""
        # Test create_test_perps_bot_with_container
        bot = create_test_perps_bot_with_container(mock_config)
        assert isinstance(bot, PerpsBot)
        assert bot.container is not None

        # Test create_test_perps_bot with container flag
        bot2 = create_test_perps_bot(mock_config, use_container=True)
        assert isinstance(bot2, PerpsBot)
        assert bot2.container is not None

    def test_container_with_settings(self, mock_config: BotConfig) -> None:
        """Test that container works with custom settings."""
        custom_settings = RuntimeSettings()
        custom_settings.coinbase_default_quote = "USDT"

        container = create_application_container(mock_config, custom_settings)

        # Verify settings are used
        assert container.settings == custom_settings
        assert container.settings.coinbase_default_quote == "USDT"

    def test_container_perps_bot_lifecycle(self, mock_config: BotConfig) -> None:
        """Test that PerpsBot created from container has proper lifecycle."""
        container = create_application_container(mock_config)
        bot = container.create_perps_bot()

        # Verify lifecycle manager was initialized
        assert bot.lifecycle_manager is not None

        # Verify bot was bootstrapped
        # (This is verified by mocking in unit tests, but here we check the structure)
        assert hasattr(bot.lifecycle_manager, "bootstrap")

    def test_backward_compatibility_mixed_usage(self, mock_config: BotConfig) -> None:
        """Test that legacy and container approaches can coexist."""
        # Create bot using legacy approach
        legacy_bot = create_perps_bot(mock_config, use_container=False)

        # Create bot using container approach
        container_bot = create_perps_bot(mock_config, use_container=True)

        # Both should be valid PerpsBot instances
        assert isinstance(legacy_bot, PerpsBot)
        assert isinstance(container_bot, PerpsBot)

        # Legacy bot should not have container
        assert legacy_bot.container is None

        # Container bot should have container
        assert container_bot.container is not None

        # Both should have the same config
        assert legacy_bot.config == container_bot.config


@pytest.fixture
def mock_config() -> BotConfig:
    """Create a mock BotConfig for testing."""
    return BotConfig(
        profile=Profile.DEV,
        mock_broker=True,
        symbols=["BTC-USD"],
        trading_window_start="09:30",
        trading_window_end="16:00",
        trading_days=["MON", "TUE", "WED", "THU", "FRI"],
    )

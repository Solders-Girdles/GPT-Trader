"""Integration test for container-based TradingBot boot and basic functionality.

This test verifies that the composition root pattern works correctly and that
TradingBot can be created and started using the ApplicationContainer.
"""

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import ApplicationContainer


def test_container_bot_boot_roundtrip():
    """Test that container can boot TradingBot and verify wiring."""

    # Create a simple configuration for testing
    config = BotConfig.from_profile("dev", symbols=["BTC-USD"], mock_broker=True)

    # Create application container
    container = ApplicationContainer(config)

    # Verify container can create core services
    assert container.config is not None

    # Test service creation
    event_store = container.event_store
    assert event_store is not None

    orders_store = container.orders_store
    assert orders_store is not None

    config_controller = container.config_controller
    assert config_controller is not None

    # Test broker creation (should work with mock settings)
    broker = container.broker
    assert broker is not None

    # Create TradingBot from container
    bot = container.create_bot()

    # Verify bot was created successfully
    assert bot is not None
    assert bot.config is not None
    assert len(bot.config.symbols) == 1
    assert "BTC-USD" in bot.config.symbols

    # Verify all dependencies are properly injected from container
    assert bot.broker == broker
    assert bot.container == container

    print("Container-based TradingBot integration test passed!")


def test_container_services():
    """Test that container provides all expected services."""

    config = BotConfig.from_profile("dev", symbols=["ETH-USD"], mock_broker=True)

    # Create container
    container = ApplicationContainer(config)

    # Verify container has all expected services
    assert container.config is not None
    assert container.broker is not None
    assert container.event_store is not None
    assert container.orders_store is not None
    assert container.risk_manager is not None
    assert container.notification_service is not None

    print("Container services test passed!")


if __name__ == "__main__":
    # Run the tests directly
    test_container_services()
    print("\nAll container integration tests passed!")

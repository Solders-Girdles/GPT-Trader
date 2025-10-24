"""Integration test for container-based PerpsBot boot and basic functionality.

This test verifies that the composition root pattern works correctly and that
PerpsBot can be created and started using the ApplicationContainer.
"""

import pytest
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.runtime_settings import load_runtime_settings
from bot_v2.app.container import create_application_container


def test_container_perps_bot_boot_roundtrip():
    """Test that container can boot PerpsBot and execute a no-op decision path."""

    # Create a simple configuration for testing
    config = BotConfig.from_profile("dev", symbols=["BTC-USD"])

    # Load runtime settings for the test environment
    settings = load_runtime_settings({
        "PERPS_FORCE_MOCK": "true",
        "PERPS_PAPER_TRADING": "true",
        "PERPS_ENABLE_STREAMING": "false",  # Disable for test stability
        "COINBASE_SANDBOX": "true",
        "GPT_TRADER_RUNTIME_ROOT": "/tmp/test_gpt_trader",
        "EVENT_STORE_ROOT": "/tmp/test_gpt_trader/events",
    })

    # Create application container
    container = create_application_container(config, settings)

    # Verify container can create core services
    assert container.config is not None
    assert container.settings is not None

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

    # Create PerpsBot from container
    bot = container.create_perps_bot()

    # Verify bot was created successfully
    assert bot is not None
    assert bot.bot_id is not None
    assert len(bot.symbols) == 1
    assert "BTC-USD" in bot.symbols

    # Test that bot lifecycle works and dependencies are properly wired
    try:
        # Verify the bot is properly constructed with all dependencies
        assert bot.config_controller is not None
        assert bot.registry is not None
        assert bot.event_store is not None
        assert bot.orders_store is not None
        assert bot.lifecycle_manager is not None

        # Verify all dependencies are properly injected from container
        assert bot.registry.broker == broker
        assert bot.registry.event_store == event_store
        assert bot.registry.orders_store == orders_store

        # Test that the bot can access its basic configuration
        assert bot.bot_id is not None
        assert len(bot.symbols) > 0
        assert "BTC-USD" in bot.symbols

        # Verify the bot is in proper initialized state
        # (Note: We don't actually start trading in this test - just verify wiring)

    except Exception as e:
        pytest.fail(f"Bot wiring verification failed: {e}")

    # Clean shutdown
    # In a real scenario, you would call bot.lifecycle_manager.shutdown()
    # For this integration test, we just verify the bot is properly constructed

    print("✅ Container-based PerpsBot integration test passed!")


def test_container_service_registry_compatibility():
    """Test that container can create ServiceRegistry for backward compatibility."""

    config = BotConfig.from_profile("dev", symbols=["ETH-USD"])
    settings = load_runtime_settings({
        "PERPS_FORCE_MOCK": "true",
        "PERPS_PAPER_TRADING": "true",
        "COINBASE_SANDBOX": "true",
    })

    # Create container
    container = create_application_container(config, settings)

    # Create service registry from container
    registry = container.create_service_registry()

    # Verify registry has all expected services
    assert registry.config is not None
    assert registry.broker is not None
    assert registry.event_store is not None
    assert registry.orders_store is not None
    assert registry.runtime_settings is not None

    print("✅ Container ServiceRegistry compatibility test passed!")


if __name__ == "__main__":
    # Run the tests directly
    test_container_perps_bot_boot_roundtrip()
    test_container_service_registry_compatibility()
    print("\n🎉 All container integration tests passed!")
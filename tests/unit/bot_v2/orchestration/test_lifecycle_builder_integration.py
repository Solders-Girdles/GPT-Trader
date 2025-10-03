"""Integration tests verifying LifecycleService works with PerpsBotBuilder."""

import os
import pytest
from unittest.mock import AsyncMock, Mock, patch

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder


@pytest.fixture
def minimal_config():
    """Create minimal bot configuration."""
    return BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-USD"],
        dry_run=True,
    )


class TestLifecycleServiceBuilderIntegration:
    """Test that LifecycleService integrates correctly with PerpsBotBuilder."""

    def test_builder_creates_lifecycle_service(self, minimal_config):
        """Builder creates lifecycle_service attribute."""
        bot = PerpsBotBuilder(minimal_config).build()

        assert hasattr(bot, "lifecycle_service")
        assert bot.lifecycle_service is not None

    def test_builder_creates_product_map(self, minimal_config):
        """Builder creates _product_map attribute."""
        bot = PerpsBotBuilder(minimal_config).build()

        assert hasattr(bot, "_product_map")
        assert isinstance(bot._product_map, dict)

    @pytest.mark.asyncio
    async def test_run_with_builder_path_single_cycle(self, minimal_config):
        """Bot built with builder can run single cycle without AttributeError."""
        bot = PerpsBotBuilder(minimal_config).build()

        # Mock run_cycle to avoid actual trading logic
        bot.run_cycle = AsyncMock()

        # This should not raise AttributeError on lifecycle_service
        await bot.run(single_cycle=True)

        # Verify lifecycle ran
        bot.run_cycle.assert_called_once()

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_LIFECYCLE_SERVICE": "true"})
    async def test_run_uses_lifecycle_service_when_enabled(self, minimal_config):
        """Bot uses LifecycleService when USE_LIFECYCLE_SERVICE=true."""
        bot = PerpsBotBuilder(minimal_config).build()
        bot.run_cycle = AsyncMock()

        # Mock the lifecycle service's run method
        original_run = bot.lifecycle_service.run
        bot.lifecycle_service.run = AsyncMock()

        await bot.run(single_cycle=True)

        # Should have called lifecycle service
        bot.lifecycle_service.run.assert_called_once()
        # Verify it was called with single_cycle=True (positional or keyword)
        call_args = bot.lifecycle_service.run.call_args
        assert call_args[0][0] is True or call_args[1].get("single_cycle") is True

    @pytest.mark.asyncio
    async def test_lifecycle_service_can_configure_background_tasks(self, minimal_config):
        """LifecycleService can configure background tasks without error."""
        minimal_config.dry_run = False  # Enable background tasks
        bot = PerpsBotBuilder(minimal_config).build()

        # This should not raise any errors
        bot.lifecycle_service.configure_background_tasks(single_cycle=False)

        # Should have registered tasks
        assert len(bot.lifecycle_service._task_registry._factory_functions) > 0

    def test_builder_path_has_all_dependencies_for_lifecycle(self, minimal_config):
        """Builder creates all dependencies required by LifecycleService."""
        bot = PerpsBotBuilder(minimal_config).build()

        # Lifecycle service requires these attributes
        required_attrs = [
            "config",
            "runtime_coordinator",
            "execution_coordinator",
            "system_monitor",
            "account_telemetry",
            "run_cycle",
            "shutdown",
        ]

        for attr in required_attrs:
            assert hasattr(bot, attr), f"Missing required attribute: {attr}"

    def test_lifecycle_service_bound_to_correct_bot_instance(self, minimal_config):
        """LifecycleService._bot points to the actual bot, not the builder temp instance."""
        bot = PerpsBotBuilder(minimal_config).build()

        # The lifecycle service should point to the actual bot instance
        assert (
            bot.lifecycle_service._bot is bot
        ), "lifecycle_service._bot points to wrong instance (builder temp instead of real bot)"

    def test_all_coordinators_bound_to_correct_bot_instance(self, minimal_config):
        """All coordinator services point to the actual bot, not the builder temp instance."""
        bot = PerpsBotBuilder(minimal_config).build()

        # All coordinators should point to the actual bot instance
        assert bot.strategy_orchestrator._bot is bot
        assert bot.execution_coordinator._bot is bot
        assert bot.system_monitor._bot is bot
        assert bot.runtime_coordinator._bot is bot
        assert bot.lifecycle_service._bot is bot

    def test_running_flag_shared_between_bot_and_lifecycle(self, minimal_config):
        """Bot.running and lifecycle_service._bot.running are the same object."""
        bot = PerpsBotBuilder(minimal_config).build()

        # Initially False
        assert bot.running is False
        assert bot.lifecycle_service._bot.running is False

        # Setting on bot should affect lifecycle's view
        bot.running = True
        assert bot.lifecycle_service._bot.running is True

        # Setting on lifecycle's bot should affect the real bot
        bot.lifecycle_service._bot.running = False
        assert bot.running is False

    @pytest.mark.asyncio
    async def test_shutdown_handler_can_stop_lifecycle_loop(self, minimal_config):
        """Simulates ShutdownHandler setting bot.running=False to stop the loop."""
        bot = PerpsBotBuilder(minimal_config).build()
        bot.run_cycle = AsyncMock()

        # Start the bot with a custom sleep that checks the running flag
        async def check_running_and_stop(duration):
            # Simulate ShutdownHandler doing: self.bot.running = False
            bot.running = False

        bot.lifecycle_service._sleep_fn = check_running_and_stop

        await bot.run(single_cycle=False)

        # Lifecycle should have stopped because bot.running was set to False
        # If the binding was wrong, the loop would hang
        assert bot.running is False

    def test_normal_construction_path_rebinds_services(self, minimal_config):
        """PerpsBot(config) via builder also rebinds all coordinator services."""
        # This is the normal way users create a bot
        bot = PerpsBot(minimal_config)

        # All coordinators should point to the actual bot instance, not temp
        assert bot.lifecycle_service._bot is bot
        assert bot.strategy_orchestrator._bot is bot
        assert bot.execution_coordinator._bot is bot
        assert bot.system_monitor._bot is bot
        assert bot.runtime_coordinator._bot is bot

    @pytest.mark.asyncio
    async def test_normal_construction_running_flag_works(self, minimal_config):
        """PerpsBot(config) via builder has working running flag."""
        bot = PerpsBot(minimal_config)
        bot.run_cycle = AsyncMock()

        # Test that setting bot.running stops the lifecycle
        async def stop_after_first_cycle(duration):
            bot.running = False

        bot.lifecycle_service._sleep_fn = stop_after_first_cycle

        # Should not hang
        await bot.run(single_cycle=False)

        assert bot.running is False

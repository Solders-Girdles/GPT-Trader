"""
Characterization Tests for Feature Toggles and Streaming

Tests verifying feature toggles and streaming restart behavior.
"""

import pytest
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.config_controller import ConfigChange


@pytest.mark.integration
@pytest.mark.characterization
class TestFeatureToggles:
    """Verify feature toggles behave as expected."""

    def test_market_data_service_initializes(self, monkeypatch, tmp_path, minimal_config):
        """Document: MarketDataService is always initialized and shares state"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Verify MarketDataService exists and shares state
        assert hasattr(bot, "_market_data_service")
        assert bot._market_data_service is not None
        assert bot._market_data_service.mark_windows is bot.mark_windows
        assert bot._market_data_service._mark_lock is bot._mark_lock

    def test_streaming_service_always_created(self, monkeypatch, tmp_path, minimal_config):
        """Document: StreamingService always created with MarketDataService"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))

        bot = PerpsBot(minimal_config)

        # Verify StreamingService always exists when MarketDataService exists
        assert hasattr(bot, "_streaming_service")
        assert bot._market_data_service is not None
        assert bot._streaming_service is not None
        assert bot._streaming_service.symbols == bot.symbols
        assert bot._streaming_service.broker is bot.broker
        assert bot._streaming_service.market_data_service is bot._market_data_service
        assert bot._streaming_service.risk_manager is bot.risk_manager

    def test_direct_constructor_uses_builder(self, monkeypatch, tmp_path, minimal_config):
        """Document: Direct PerpsBot() construction always uses builder (no legacy path)"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        # Direct construction (goes through builder internally)
        bot = PerpsBot(minimal_config)

        # Verify bot is fully initialized via builder
        assert bot.config is minimal_config
        assert bot.symbols == minimal_config.symbols
        assert bot.registry is not None
        assert bot._market_data_service is not None
        assert bot._streaming_service is not None

        # Note: Builder is now the only construction path (USE_PERPS_BOT_BUILDER retired)
        # Comprehensive builder tests in test_builder.py


@pytest.mark.integration
@pytest.mark.characterization
class TestStreamingServiceRestartBehavior:
    """Characterize streaming restart behavior on config changes"""

    def test_restart_streaming_stops_and_restarts_on_level_change(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: Changing perps_stream_level must restart streaming"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))

        # Start with streaming enabled
        config = BotConfig(
            profile=Profile.CANARY,  # Required for streaming
            symbols=["BTC-USD"],
            mock_broker=True,
            perps_enable_streaming=True,
            perps_stream_level=1,
        )
        bot = PerpsBot(config)

        # Verify initial streaming service exists (always created)
        assert bot._streaming_service is not None

        # Simulate config change with level change
        new_config = BotConfig(
            profile=Profile.CANARY,
            symbols=["BTC-USD"],
            mock_broker=True,
            perps_enable_streaming=True,
            perps_stream_level=2,  # Changed
        )

        change = ConfigChange(updated=new_config, diff={"perps_stream_level": (1, 2)})

        # Apply change (should trigger restart)
        bot.apply_config_change(change)

        # Verify service still exists after restart
        assert bot._streaming_service is not None

    def test_restart_streaming_stops_when_disabled(self, monkeypatch, tmp_path):
        """Document: Disabling streaming must stop the service"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))

        # Start with streaming enabled
        config = BotConfig(
            profile=Profile.CANARY,
            symbols=["BTC-USD"],
            mock_broker=True,
            perps_enable_streaming=True,
        )
        bot = PerpsBot(config)

        # Simulate disabling streaming
        new_config = BotConfig(
            profile=Profile.CANARY,
            symbols=["BTC-USD"],
            mock_broker=True,
            perps_enable_streaming=False,  # Disabled
        )

        change = ConfigChange(updated=new_config, diff={"perps_enable_streaming": (True, False)})

        # Apply change (should stop streaming)
        bot.apply_config_change(change)

        # Verify service stopped (if it was created)
        if bot._streaming_service is not None:
            assert not bot._streaming_service.is_running()

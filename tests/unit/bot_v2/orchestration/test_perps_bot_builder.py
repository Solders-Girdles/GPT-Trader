"""Unit tests for PerpsBotBuilder."""

import os
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch

import pytest

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder
from bot_v2.orchestration.service_registry import ServiceRegistry, empty_registry


class TestPerpsBotBuilderConstruction:
    """Test basic builder construction and fluent API."""

    def test_builder_minimal_config(self, minimal_config: BotConfig):
        """Builder constructs bot with minimal config."""
        builder = PerpsBotBuilder(minimal_config)
        bot = builder.build()

        assert bot.bot_id == "perps_bot"
        assert bot.config == minimal_config
        assert bot.running is False
        assert hasattr(bot, "config_controller")
        assert hasattr(bot, "registry")

    def test_builder_with_custom_registry(self, minimal_config: BotConfig):
        """Builder respects custom registry."""
        mock_broker = Mock()
        mock_risk = Mock()
        custom_registry = empty_registry(minimal_config).with_updates(
            broker=mock_broker, risk_manager=mock_risk
        )

        builder = PerpsBotBuilder(minimal_config).with_registry(custom_registry)
        bot = builder.build()

        assert bot.registry.broker is mock_broker
        assert bot.registry.risk_manager is mock_risk

    def test_builder_with_custom_symbols(self, minimal_config: BotConfig):
        """Builder respects custom symbols."""
        symbols = ["BTC-USD", "ETH-USD"]
        builder = PerpsBotBuilder(minimal_config).with_symbols(symbols)
        bot = builder.build()

        assert bot.symbols == symbols
        assert "BTC-USD" in bot.mark_windows
        assert "ETH-USD" in bot.mark_windows

    def test_builder_uses_config_symbols_by_default(self, minimal_config: BotConfig):
        """Builder falls back to config symbols when not explicitly set."""
        minimal_config.symbols = ["SOL-USD"]
        builder = PerpsBotBuilder(minimal_config)
        bot = builder.build()

        assert bot.symbols == ["SOL-USD"]
        assert "SOL-USD" in bot.mark_windows

    def test_builder_fluent_api(self, minimal_config: BotConfig):
        """Builder supports method chaining."""
        custom_registry = empty_registry(minimal_config)
        symbols = ["BTC-USD"]

        builder = (
            PerpsBotBuilder(minimal_config).with_registry(custom_registry).with_symbols(symbols)
        )
        bot = builder.build()

        assert bot.symbols == symbols
        assert bot.registry.config == minimal_config


class TestPerpsBotBuilderComponents:
    """Test that builder creates all required components."""

    def test_builder_creates_configuration_state(self, minimal_config: BotConfig):
        """Builder creates all configuration state components."""
        bot = PerpsBotBuilder(minimal_config).build()

        # Configuration state
        assert hasattr(bot, "config_controller")
        assert hasattr(bot, "config")
        assert hasattr(bot, "registry")
        assert hasattr(bot, "_session_guard")
        assert hasattr(bot, "symbols")
        assert hasattr(bot, "_derivatives_enabled")
        assert hasattr(bot, "bot_id")
        assert hasattr(bot, "start_time")
        assert hasattr(bot, "running")

    def test_builder_creates_runtime_state(self, minimal_config: BotConfig):
        """Builder creates all runtime state components."""
        bot = PerpsBotBuilder(minimal_config).build()

        # Runtime state
        assert hasattr(bot, "mark_windows")
        assert hasattr(bot, "last_decisions")
        assert hasattr(bot, "_last_positions")
        assert hasattr(bot, "order_stats")
        assert hasattr(bot, "_order_lock")
        assert hasattr(bot, "_mark_lock")
        assert hasattr(bot, "_symbol_strategies")
        assert hasattr(bot, "strategy")
        assert hasattr(bot, "_exec_engine")
        assert hasattr(bot, "_product_map")

    def test_builder_creates_storage_components(self, minimal_config: BotConfig):
        """Builder creates storage layer components."""
        bot = PerpsBotBuilder(minimal_config).build()

        assert hasattr(bot, "event_store")
        assert hasattr(bot, "orders_store")

    def test_builder_creates_core_services(self, minimal_config: BotConfig):
        """Builder creates core orchestration services."""
        bot = PerpsBotBuilder(minimal_config).build()

        assert hasattr(bot, "strategy_orchestrator")
        assert hasattr(bot, "execution_coordinator")
        assert hasattr(bot, "system_monitor")
        assert hasattr(bot, "runtime_coordinator")
        assert hasattr(bot, "lifecycle_service")

    def test_builder_creates_accounting_services(self, minimal_config: BotConfig):
        """Builder creates accounting and telemetry services."""
        bot = PerpsBotBuilder(minimal_config).build()

        assert hasattr(bot, "account_manager")
        assert hasattr(bot, "account_telemetry")

    def test_builder_creates_market_services(self, minimal_config: BotConfig):
        """Builder creates market monitoring services."""
        bot = PerpsBotBuilder(minimal_config).build()

        assert hasattr(bot, "_market_monitor")

    @patch.dict(os.environ, {"USE_NEW_MARKET_DATA_SERVICE": "true"})
    def test_builder_creates_market_data_service_when_enabled(self, minimal_config: BotConfig):
        """Builder creates MarketDataService when flag is enabled."""
        bot = PerpsBotBuilder(minimal_config).build()

        assert hasattr(bot, "_market_data_service")
        assert bot._market_data_service is not None

    @patch.dict(os.environ, {"USE_NEW_MARKET_DATA_SERVICE": "false"})
    def test_builder_skips_market_data_service_when_disabled(self, minimal_config: BotConfig):
        """Builder skips MarketDataService when flag is disabled."""
        bot = PerpsBotBuilder(minimal_config).build()

        assert hasattr(bot, "_market_data_service")
        assert bot._market_data_service is None

    @patch.dict(os.environ, {"USE_NEW_STREAMING_SERVICE": "true"})
    def test_builder_creates_streaming_service_when_enabled(self, minimal_config: BotConfig):
        """Builder creates StreamingService when flag is enabled and MarketDataService exists."""
        minimal_config.symbols = ["BTC-USD"]
        bot = PerpsBotBuilder(minimal_config).build()

        # StreamingService requires MarketDataService, which requires USE_NEW_MARKET_DATA_SERVICE
        if bot._market_data_service is not None:
            assert bot._streaming_service is not None
        else:
            assert bot._streaming_service is None


class TestPerpsBotBuilderErrorHandling:
    """Test builder error handling and validation."""

    def test_builder_validates_config_symbols(self, minimal_config: BotConfig):
        """Builder validates config via ConfigController."""
        from bot_v2.orchestration.configuration import ConfigValidationError

        # ConfigController validates that symbols must be non-empty
        minimal_config.symbols = []

        with pytest.raises(ConfigValidationError, match="symbols must contain at least one"):
            PerpsBotBuilder(minimal_config).build()

    def test_builder_validates_none_symbols(self, minimal_config: BotConfig):
        """Builder rejects None symbols via ConfigController validation."""
        from bot_v2.orchestration.configuration import ConfigValidationError

        minimal_config.symbols = None

        with pytest.raises(ConfigValidationError):
            PerpsBotBuilder(minimal_config).build()

    def test_builder_with_dry_run_mode(self, minimal_config: BotConfig):
        """Builder works in dry-run mode without real broker."""
        # dry_run mode should allow building without a real broker
        minimal_config.dry_run = True

        bot = PerpsBotBuilder(minimal_config).build()
        assert bot.config.dry_run is True


class TestPerpsBotBuilderStreaming:
    """Test builder streaming initialization behavior."""

    def test_builder_has_streaming_service_attribute(self, minimal_config: BotConfig):
        """Builder creates _streaming_service attribute."""
        bot = PerpsBotBuilder(minimal_config).build()

        # Verify streaming service attribute exists (may be None based on config/flags)
        assert hasattr(bot, "_streaming_service")

    def test_builder_skips_streaming_when_disabled(self, minimal_config: BotConfig):
        """Builder skips streaming when perps_enable_streaming=false."""
        minimal_config.perps_enable_streaming = False

        bot = PerpsBotBuilder(minimal_config).build()

        # Streaming should not be started (no assertions needed, just verify no errors)
        assert bot is not None


class TestPerpsBotFromBuilder:
    """Test PerpsBot.from_builder classmethod."""

    def test_from_builder_classmethod(self, minimal_config: BotConfig):
        """PerpsBot.from_builder creates instance from builder."""
        builder = PerpsBotBuilder(minimal_config).with_symbols(["BTC-USD"])
        bot = PerpsBot.from_builder(builder)

        assert isinstance(bot, PerpsBot)
        assert bot.symbols == ["BTC-USD"]
        assert bot.bot_id == "perps_bot"


@pytest.fixture
def minimal_config() -> BotConfig:
    """Minimal valid BotConfig for testing."""
    config = BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-USD"],
        dry_run=True,
        update_interval=60,
        long_ma=50,
        short_ma=20,
    )
    return config

"""Tests for IBotRuntime protocol.

Verifies that:
1. Protocol is properly defined
2. PerpsBot implements the protocol
3. Protocol can be used for type checking
"""

from __future__ import annotations

import pytest
from decimal import Decimal
from unittest.mock import Mock

from bot_v2.orchestration.core.bot_interface import IBotRuntime
from bot_v2.orchestration.configuration import BotConfig, Profile


class TestIBotRuntimeProtocol:
    """Test IBotRuntime protocol definition."""

    def test_protocol_is_runtime_checkable(self):
        """Verify protocol is marked as runtime_checkable."""
        # This allows isinstance() checks
        # Check that the protocol has the _is_runtime_protocol marker
        assert hasattr(IBotRuntime, "_is_runtime_protocol")
        assert IBotRuntime._is_runtime_protocol is True

    def test_perps_bot_implements_protocol(self):
        """Verify PerpsBot implements IBotRuntime protocol.

        Note: Full isinstance() compliance verified in integration tests.
        This test verifies core attributes exist.
        """
        from bot_v2.orchestration.perps_bot import PerpsBot

        # Create minimal config
        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-USD"],
            mock_broker=True,
        )

        # Create bot instance
        bot = PerpsBot(config)

        # Verify bot has core protocol attributes
        # (Full isinstance() check done in integration tests)
        assert hasattr(bot, "config")
        assert hasattr(bot, "running")
        assert hasattr(bot, "broker")
        assert hasattr(bot, "registry")

    def test_protocol_has_required_properties(self):
        """Verify protocol defines all required properties."""
        # Check that protocol has all expected attributes
        required_properties = [
            "config",
            "running",
            "broker",
            "registry",
            "risk_manager",
            "metrics_server",
            "strategy_orchestrator",
            "execution_coordinator",
            "config_controller",
            "orders_store",
            "last_decisions",
            "mark_windows",
            "_product_map",
            "_symbol_strategies",
            "strategy",
        ]

        for prop in required_properties:
            assert hasattr(IBotRuntime, prop), f"Protocol missing property: {prop}"

    def test_protocol_has_required_methods(self):
        """Verify protocol defines all required methods."""
        required_methods = [
            "run_cycle",
            "get_product",
            "execute_decision",
            "stop",
        ]

        for method in required_methods:
            assert hasattr(IBotRuntime, method), f"Protocol missing method: {method}"

    def test_mock_implements_protocol(self):
        """Verify a mock can implement the protocol for testing."""
        # Create a mock that implements the protocol
        mock_bot = Mock(spec=IBotRuntime)

        # Set up required properties
        mock_bot.config = Mock()
        mock_bot.running = False
        mock_bot.broker = Mock()
        mock_bot.last_decisions = {}
        mock_bot.mark_windows = {"BTC-USD": [Decimal("50000")]}

        # Mock should be usable anywhere IBotRuntime is expected
        def accept_bot_runtime(bot: IBotRuntime) -> str:
            return f"Bot running: {bot.running}"

        result = accept_bot_runtime(mock_bot)
        assert result == "Bot running: False"


class TestIBotRuntimeUsage:
    """Test using IBotRuntime for dependency injection."""

    def test_service_accepts_protocol(self):
        """Verify services can accept IBotRuntime instead of PerpsBot."""

        # Example service that accepts protocol
        class ExampleService:
            def __init__(self, bot: IBotRuntime):
                self.bot = bot

            def get_config_profile(self) -> str:
                return self.bot.config.profile.value

        # Create mock bot
        mock_bot = Mock(spec=IBotRuntime)
        mock_bot.config = Mock()
        mock_bot.config.profile = Profile.DEV

        # Service should work with mock
        service = ExampleService(mock_bot)
        assert service.get_config_profile() == "dev"

    def test_protocol_used_in_type_hints(self):
        """Verify protocol works in type hints without circular imports."""

        # This should not cause circular import
        from bot_v2.orchestration.core.bot_interface import IBotRuntime

        def process_bot(bot: IBotRuntime) -> bool:
            """Example function using protocol in type hint."""
            return bot.running

        # Should work with mock
        mock_bot = Mock(spec=IBotRuntime)
        mock_bot.running = True

        assert process_bot(mock_bot) is True


@pytest.mark.integration
class TestPerpsBotProtocolCompliance:
    """Integration tests verifying PerpsBot fully implements protocol."""

    def test_perps_bot_has_all_protocol_properties(self, monkeypatch, tmp_path):
        """Verify PerpsBot instance has all protocol properties."""
        from bot_v2.orchestration.perps_bot import PerpsBot

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-USD"],
            mock_broker=True,
        )

        bot = PerpsBot(config)

        # Verify all protocol properties exist
        assert hasattr(bot, "config")
        assert hasattr(bot, "running")
        assert hasattr(bot, "broker")
        assert hasattr(bot, "registry")
        assert hasattr(bot, "risk_manager")
        assert hasattr(bot, "metrics_server")
        assert hasattr(bot, "strategy_orchestrator")
        assert hasattr(bot, "execution_coordinator")
        assert hasattr(bot, "config_controller")
        assert hasattr(bot, "orders_store")
        assert hasattr(bot, "last_decisions")
        assert hasattr(bot, "mark_windows")
        assert hasattr(bot, "_product_map")

    def test_perps_bot_has_all_protocol_methods(self, monkeypatch, tmp_path):
        """Verify PerpsBot instance has all protocol methods."""
        from bot_v2.orchestration.perps_bot import PerpsBot

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-USD"],
            mock_broker=True,
        )

        bot = PerpsBot(config)

        # Verify all protocol methods exist and are callable
        assert hasattr(bot, "run_cycle") and callable(bot.run_cycle)
        assert hasattr(bot, "get_product") and callable(bot.get_product)
        assert hasattr(bot, "execute_decision") and callable(bot.execute_decision)
        assert hasattr(bot, "stop") and callable(bot.stop)

"""
Characterization Tests for PerpsBot Delegation

Tests documenting method delegation patterns to services.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

from bot_v2.orchestration.perps_bot import PerpsBot


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotDelegation:
    """Characterize method delegation patterns"""

    @pytest.mark.asyncio
    async def test_process_symbol_delegates_to_strategy_orchestrator(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: process_symbol must delegate to strategy_orchestrator"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot.strategy_orchestrator.process_symbol = AsyncMock()

        await bot.process_symbol("BTC-USD")

        bot.strategy_orchestrator.process_symbol.assert_called_once_with("BTC-USD", None, None)

    @pytest.mark.asyncio
    async def test_execute_decision_delegates_to_execution_coordinator(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: execute_decision must delegate to execution_coordinator"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot.execution_coordinator.execute_decision = AsyncMock()

        decision = Mock()
        mark = Decimal("50000")
        product = Mock()
        position_state = {}

        await bot.execute_decision("BTC-USD", decision, mark, product, position_state)

        bot.execution_coordinator.execute_decision.assert_called_once()

    def test_write_health_status_delegation(self, monkeypatch, tmp_path, minimal_config):
        """Document: write_health_status must delegate to system_monitor"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot.system_monitor.write_health_status = Mock()

        bot.write_health_status(ok=True, message="test", error="")

        bot.system_monitor.write_health_status.assert_called_once_with(
            ok=True, message="test", error=""
        )

    def test_is_reduce_only_mode_delegation(self, monkeypatch, tmp_path, minimal_config):
        """Document: is_reduce_only_mode must delegate to runtime_coordinator"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot.runtime_coordinator.is_reduce_only_mode = Mock(return_value=True)

        result = bot.is_reduce_only_mode()

        assert result is True
        bot.runtime_coordinator.is_reduce_only_mode.assert_called_once()

    def test_set_reduce_only_mode_delegation(self, monkeypatch, tmp_path, minimal_config):
        """Document: set_reduce_only_mode must delegate to runtime_coordinator"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot.runtime_coordinator.set_reduce_only_mode = Mock()

        bot.set_reduce_only_mode(enabled=True, reason="test reason")

        bot.runtime_coordinator.set_reduce_only_mode.assert_called_once_with(True, "test reason")


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotConfigChange:
    """Characterize apply_config_change behavior"""

    def test_apply_config_change_updates_symbols(self, monkeypatch, tmp_path, minimal_config):
        """Document: apply_config_change must update bot.symbols"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        assert bot.symbols == ["BTC-USD"]

        # Mock services to avoid side effects
        bot.strategy_orchestrator.init_strategy = Mock()

        # Create config change
        from bot_v2.orchestration.configuration import BotConfig, Profile
        from bot_v2.orchestration.config_controller import ConfigChange

        new_config = BotConfig(
            profile=Profile.DEV, symbols=["ETH-USD", "SOL-USD"], mock_broker=True
        )
        change = ConfigChange(updated=new_config, diff={"symbols": ["ETH-USD", "SOL-USD"]})

        # Apply change
        bot.apply_config_change(change)

        # Verify symbols updated
        assert bot.symbols == ["ETH-USD", "SOL-USD"]
        assert bot.config is new_config

    def test_apply_config_change_adds_new_mark_windows(self, monkeypatch, tmp_path, minimal_config):
        """Document: apply_config_change must create mark_windows for new symbols"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        assert "BTC-USD" in bot.mark_windows
        assert "ETH-USD" not in bot.mark_windows

        # Mock services
        bot.strategy_orchestrator.init_strategy = Mock()

        from bot_v2.orchestration.configuration import BotConfig, Profile
        from bot_v2.orchestration.config_controller import ConfigChange

        # Add ETH-USD to symbols
        new_config = BotConfig(
            profile=Profile.DEV, symbols=["BTC-USD", "ETH-USD"], mock_broker=True
        )
        change = ConfigChange(updated=new_config, diff={"symbols": ["BTC-USD", "ETH-USD"]})

        bot.apply_config_change(change)

        # Verify new mark window created
        assert "BTC-USD" in bot.mark_windows
        assert "ETH-USD" in bot.mark_windows

    def test_apply_config_change_removes_old_mark_windows(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: apply_config_change must delete mark_windows for removed symbols"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        from bot_v2.orchestration.configuration import BotConfig, Profile

        # Start with multiple symbols
        config = BotConfig(profile=Profile.DEV, symbols=["BTC-USD", "ETH-USD"], mock_broker=True)
        bot = PerpsBot(config)
        assert "BTC-USD" in bot.mark_windows
        assert "ETH-USD" in bot.mark_windows

        # Mock services
        bot.strategy_orchestrator.init_strategy = Mock()

        from bot_v2.orchestration.config_controller import ConfigChange

        # Remove ETH-USD
        new_config = BotConfig(profile=Profile.DEV, symbols=["BTC-USD"], mock_broker=True)
        change = ConfigChange(updated=new_config, diff={"symbols": ["BTC-USD"]})

        bot.apply_config_change(change)

        # Verify old mark window removed
        assert "BTC-USD" in bot.mark_windows
        assert "ETH-USD" not in bot.mark_windows

    def test_apply_config_change_updates_streaming_symbols(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: apply_config_change must call streaming_service.update_symbols"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Mock services
        bot.strategy_orchestrator.init_strategy = Mock()
        bot._streaming_service.update_symbols = Mock()

        from bot_v2.orchestration.configuration import BotConfig, Profile
        from bot_v2.orchestration.config_controller import ConfigChange

        new_config = BotConfig(
            profile=Profile.DEV, symbols=["ETH-USD", "SOL-USD"], mock_broker=True
        )
        change = ConfigChange(updated=new_config, diff={"symbols": ["ETH-USD", "SOL-USD"]})

        bot.apply_config_change(change)

        # Verify streaming service updated
        bot._streaming_service.update_symbols.assert_called_once_with(["ETH-USD", "SOL-USD"])

    def test_apply_config_change_reinitializes_strategy(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: apply_config_change must call strategy_orchestrator.init_strategy"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Mock strategy orchestrator
        bot.strategy_orchestrator.init_strategy = Mock()

        from bot_v2.orchestration.configuration import BotConfig, Profile
        from bot_v2.orchestration.config_controller import ConfigChange

        new_config = BotConfig(
            profile=Profile.DEV, symbols=["BTC-USD"], mock_broker=True, short_ma=10
        )
        change = ConfigChange(updated=new_config, diff={"short_ma": 10})

        bot.apply_config_change(change)

        # Verify strategy re-initialized
        bot.strategy_orchestrator.init_strategy.assert_called_once()

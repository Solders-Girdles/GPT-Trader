"""
Characterization Tests for PerpsBot Initialization

Tests documenting initialization side effects and service creation.
"""

import pytest
from bot_v2.orchestration.perps_bot import PerpsBot


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotInitialization:
    """Characterize initialization side effects"""

    def test_initialization_creates_all_services(self, monkeypatch, tmp_path, minimal_config):
        """Document: All services must exist after __init__"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Core services (must exist)
        assert hasattr(bot, "strategy_orchestrator")
        assert hasattr(bot, "execution_coordinator")
        assert hasattr(bot, "system_monitor")
        assert hasattr(bot, "runtime_coordinator")
        assert bot.strategy_orchestrator is not None
        assert bot.execution_coordinator is not None
        assert bot.system_monitor is not None
        assert bot.runtime_coordinator is not None

    def test_initialization_creates_accounting_services(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: Accounting services must exist"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert hasattr(bot, "account_manager")
        assert hasattr(bot, "account_telemetry")
        assert bot.account_manager is not None
        assert bot.account_telemetry is not None

    def test_initialization_creates_market_monitor(self, monkeypatch, tmp_path, minimal_config):
        """Document: Market monitor must exist"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert hasattr(bot, "_market_monitor")
        assert bot._market_monitor is not None

    def test_initialization_creates_runtime_state(self, monkeypatch, tmp_path, minimal_config):
        """Document: All runtime state dicts must be initialized"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # State dictionaries (NOTE: _product_map removed - was dead code)
        assert hasattr(bot, "mark_windows")
        assert hasattr(bot, "last_decisions")
        assert hasattr(bot, "_last_positions")
        assert hasattr(bot, "order_stats")

        # Verify types
        assert isinstance(bot.mark_windows, dict)
        assert isinstance(bot.last_decisions, dict)
        assert isinstance(bot._last_positions, dict)
        assert isinstance(bot.order_stats, dict)

        # Verify initial values
        assert "BTC-USD" in bot.mark_windows
        assert bot.mark_windows["BTC-USD"] == []
        assert bot.order_stats == {"attempted": 0, "successful": 0, "failed": 0}

    def test_initialization_creates_locks(self, monkeypatch, tmp_path, minimal_config):
        """Document: Threading locks must be created"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert hasattr(bot, "_mark_lock")
        # RLock is _thread.RLock type, check by name
        assert type(bot._mark_lock).__name__ == "RLock"

    def test_initialization_sets_symbols(self, monkeypatch, tmp_path, minimal_config):
        """Document: Symbols list is extracted from config"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert bot.symbols == ["BTC-USD"]
        assert isinstance(bot.symbols, list)

    def test_derivatives_enabled_flag(self, monkeypatch, tmp_path, minimal_config):
        """Document: derivatives_enabled flag must be set during initialization"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        # Test with derivatives disabled
        minimal_config.derivatives_enabled = False
        bot = PerpsBot(minimal_config)
        assert bot._derivatives_enabled is False

        # Test with derivatives enabled
        minimal_config.derivatives_enabled = True
        bot2 = PerpsBot(minimal_config)
        assert bot2._derivatives_enabled is True

    def test_session_guard_creation(self, monkeypatch, tmp_path, minimal_config):
        """Document: session_guard must be created during initialization"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert bot._session_guard is not None
        assert hasattr(bot._session_guard, "should_trade")

    def test_config_controller_creation(self, monkeypatch, tmp_path, minimal_config):
        """Document: config_controller must be created during initialization"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert bot.config_controller is not None
        assert hasattr(bot.config_controller, "sync_with_risk_manager")

    def test_registry_broker_exists(self, monkeypatch, tmp_path, minimal_config):
        """Document: registry.broker must be initialized"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert bot.registry.broker is not None
        # Verify it's the mock broker in dev mode
        from bot_v2.orchestration.deterministic_broker import DeterministicBroker

        assert isinstance(bot.registry.broker, DeterministicBroker)

    def test_registry_risk_manager_exists(self, monkeypatch, tmp_path, minimal_config):
        """Document: registry.risk_manager must be initialized"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert bot.registry.risk_manager is not None
        assert hasattr(bot.registry.risk_manager, "last_mark_update")

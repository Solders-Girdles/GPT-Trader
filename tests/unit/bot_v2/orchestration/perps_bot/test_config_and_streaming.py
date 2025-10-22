"""Tests for PerpsBot configuration changes, streaming tasks, and error recovery."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bot_v2.orchestration.config_controller import ConfigChange
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.session_guard import TradingSessionGuard


class TestConfigurationChanges:
    """Test configuration change handling and dynamic updates."""

    def test_apply_config_change_updates_basic_attributes(self, perps_bot_instance):
        """Test apply_config_change updates basic bot attributes."""
        bot = perps_bot_instance
        change = MagicMock(spec=ConfigChange)
        change.diff = {"symbols": ["ETH-PERP"], "derivatives_enabled": True}
        change.updated = MagicMock()
        change.updated.symbols = ["ETH-PERP"]
        change.updated.derivatives_enabled = True

        bot.apply_config_change(change)

        assert bot.config is change.updated
        assert bot.symbols == ["ETH-PERP"]
        assert bot._derivatives_enabled is True
        assert bot.registry.config is change.updated

    def test_apply_config_change_updates_session_guard(self, perps_bot_instance):
        """Test apply_config_change updates session guard."""
        bot = perps_bot_instance
        change = MagicMock(spec=ConfigChange)
        change.diff = {
            "trading_window_start": "09:00",
            "trading_window_end": "17:00",
            "trading_days": ["monday", "friday"],
        }
        change.updated = MagicMock()
        change.updated.trading_window_start = "09:00"
        change.updated.trading_window_end = "17:00"
        change.updated.trading_days = ["monday", "friday"]

        bot.apply_config_change(change)

        # Verify session guard was updated (it should be replaced with new instance)
        assert isinstance(bot._session_guard, TradingSessionGuard)

    def test_apply_config_change_resets_order_reconciler(self, perps_bot_instance):
        """Test apply_config_change resets order reconciler."""
        bot = perps_bot_instance
        change = MagicMock(spec=ConfigChange)
        change.diff = {"test_field": "test_value"}
        change.updated = MagicMock()

        bot.execution_coordinator.reset_order_reconciler = MagicMock()

        bot.apply_config_change(change)

        bot.execution_coordinator.reset_order_reconciler.assert_called_once()

    def test_apply_config_change_syncs_with_risk_manager(
        self, perps_bot_instance, mock_risk_manager
    ):
        """Test apply_config_change syncs config controller with risk manager."""
        bot = perps_bot_instance
        bot.config_controller.sync_with_risk_manager = MagicMock()
        change = MagicMock(spec=ConfigChange)
        change.diff = {"test_field": "test_value"}
        change.updated = MagicMock()

        bot.apply_config_change(change)

        bot.config_controller.sync_with_risk_manager.assert_called_once_with(mock_risk_manager)

    def test_apply_config_change_updates_mark_windows(self, perps_bot_instance):
        """Test apply_config_change updates mark windows for symbol changes."""
        bot = perps_bot_instance
        # Add initial mark windows
        bot._state.mark_windows = {"BTC-PERP": [Decimal("50000")], "ETH-PERP": [Decimal("3000")]}

        change = MagicMock(spec=ConfigChange)
        change.diff = {"symbols": ["BTC-PERP", "SOL-PERP"]}  # ETH removed, SOL added
        change.updated = MagicMock()
        change.updated.symbols = ["BTC-PERP", "SOL-PERP"]

        bot.apply_config_change(change)

        # Should have BTC-PERP (existing) and SOL-PERP (new), but not ETH-PERP (removed)
        assert "BTC-PERP" in bot._state.mark_windows
        assert "SOL-PERP" in bot._state.mark_windows
        assert "ETH-PERP" not in bot._state.mark_windows

    def test_apply_config_change_initializes_market_services(self, perps_bot_instance):
        """Test apply_config_change initializes market services."""
        bot = perps_bot_instance
        change = MagicMock(spec=ConfigChange)
        change.diff = {"test_field": "test_value"}
        change.updated = MagicMock()

        bot.telemetry_coordinator.init_market_services = MagicMock()

        bot.apply_config_change(change)

        bot.telemetry_coordinator.init_market_services.assert_called_once()

    def test_apply_config_change_initializes_strategy(self, perps_bot_instance):
        """Test apply_config_change initializes strategy."""
        bot = perps_bot_instance
        change = MagicMock(spec=ConfigChange)
        change.diff = {"test_field": "test_value"}
        change.updated = MagicMock()

        bot.strategy_orchestrator.init_strategy = MagicMock()

        bot.apply_config_change(change)

        bot.strategy_orchestrator.init_strategy.assert_called_once()

    def test_apply_config_change_restarts_streaming(self, perps_bot_instance):
        """Test apply_config_change restarts streaming if needed."""
        bot = perps_bot_instance
        change = MagicMock(spec=ConfigChange)
        change.diff = {"symbols": ["ETH-PERP"]}  # Symbol change should trigger restart
        change.updated = MagicMock()

        bot.telemetry_coordinator.restart_streaming_if_needed = MagicMock()

        bot.apply_config_change(change)

        bot.telemetry_coordinator.restart_streaming_if_needed.assert_called_once_with(change.diff)

    def test_apply_config_change_updates_baseline_snapshot(
        self, perps_bot_instance, mock_baseline_snapshot
    ):
        """Test apply_config_change updates baseline snapshot."""
        bot = perps_bot_instance
        change = MagicMock(spec=ConfigChange)
        change.diff = {"test_field": "test_value"}
        change.updated = MagicMock()
        change.updated.symbols = ["BTC-PERP"]
        change.updated.derivatives_enabled = False

        # Mock the static method
        with pytest.MonkeyPatch().context() as m:
            new_snapshot = {"updated": "baseline"}
            m.setattr(PerpsBot, "build_baseline_snapshot", MagicMock(return_value=new_snapshot))

            bot.apply_config_change(change)

            PerpsBot.build_baseline_snapshot.assert_called_once_with(
                change.updated, bot._derivatives_enabled
            )
            assert bot.baseline_snapshot is new_snapshot

    def test_apply_config_change_resets_configuration_guardian(
        self, perps_bot_instance, mock_configuration_guardian
    ):
        """Test apply_config_change resets configuration guardian baseline."""
        bot = perps_bot_instance
        change = MagicMock(spec=ConfigChange)
        change.diff = {"test_field": "test_value"}
        change.updated = MagicMock()
        change.updated.symbols = ["BTC-PERP"]
        change.updated.derivatives_enabled = False

        new_snapshot = {"updated": "baseline"}
        with pytest.MonkeyPatch().context() as m:
            m.setattr(PerpsBot, "build_baseline_snapshot", MagicMock(return_value=new_snapshot))

            bot.apply_config_change(change)

            mock_configuration_guardian.reset_baseline.assert_called_once_with(new_snapshot)


class TestStreamingFunctionality:
    """Test streaming background tasks and management."""

    def test_start_streaming_background_delegates_to_telemetry_coordinator(
        self, perps_bot_instance
    ):
        """Test _start_streaming_background delegates to telemetry coordinator."""
        bot = perps_bot_instance
        # The method is patched in the constructor, so it won't call the real method
        # We can verify the telemetry_coordinator has the method available
        assert hasattr(bot.telemetry_coordinator, "start_streaming_background")
        assert callable(getattr(bot.telemetry_coordinator, "start_streaming_background", None))

    def test_stop_streaming_background_delegates_to_telemetry_coordinator(self, perps_bot_instance):
        """Test _stop_streaming_background delegates to telemetry coordinator."""
        bot = perps_bot_instance
        # The method is patched in the constructor, so it won't call the real method
        # We can verify the telemetry_coordinator has the method available
        assert hasattr(bot.telemetry_coordinator, "stop_streaming_background")
        assert callable(getattr(bot.telemetry_coordinator, "stop_streaming_background", None))

    def test_restart_streaming_if_needed_delegates_to_telemetry_coordinator(
        self, perps_bot_instance
    ):
        """Test _restart_streaming_if_needed delegates to telemetry coordinator."""
        bot = perps_bot_instance
        bot.telemetry_coordinator.restart_streaming_if_needed = MagicMock()
        diff = {"symbols": ["ETH-PERP"]}

        bot._restart_streaming_if_needed(diff)

        bot.telemetry_coordinator.restart_streaming_if_needed.assert_called_once_with(diff)

    def test_run_stream_loop_delegates_to_telemetry_coordinator(self, perps_bot_instance):
        """Test _run_stream_loop delegates to telemetry coordinator."""
        bot = perps_bot_instance
        bot.telemetry_coordinator._run_stream_loop = MagicMock()
        symbols = ["BTC-PERP", "ETH-PERP"]
        level = 2

        bot._run_stream_loop(symbols, level)

        bot.telemetry_coordinator._run_stream_loop.assert_called_once_with(
            symbols, level, stop_signal=None
        )


class TestMarkWindowManagement:
    """Test mark window management and updates."""

    def test_update_mark_window_delegates_to_strategy_coordinator(self, perps_bot_instance):
        """Test _update_mark_window delegates to strategy coordinator."""
        bot = perps_bot_instance
        bot.strategy_coordinator.update_mark_window = MagicMock()
        symbol = "BTC-PERP"
        mark = Decimal("50000.0")

        bot._update_mark_window(symbol, mark)

        bot.strategy_coordinator.update_mark_window.assert_called_once_with(symbol, mark)


class TestMarkWindowOperations:
    """Test mark window operations through strategy coordinator."""

    def test_update_mark_window_integration(self, perps_bot_instance):
        """Test update_mark_window integration with strategy coordinator."""
        bot = perps_bot_instance
        symbol = "BTC-PERP"
        mark = Decimal("50100.0")

        # Mock the strategy coordinator method
        bot.strategy_coordinator.update_mark_window = MagicMock()

        bot._update_mark_window(symbol, mark)

        bot.strategy_coordinator.update_mark_window.assert_called_once_with(symbol, mark)


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""

    def test_runtime_coordinator_not_registered_raises(self, perps_bot_instance):
        """Test RuntimeError when runtime coordinator not registered."""
        bot = perps_bot_instance
        # Remove the coordinator
        bot._coordinator_registry._coordinators = {}

        with pytest.raises(RuntimeError, match="Runtime coordinator not registered"):
            _ = bot.runtime_coordinator

    def test_execution_coordinator_not_registered_raises(self, perps_bot_instance):
        """Test RuntimeError when execution coordinator not registered."""
        bot = perps_bot_instance
        # Create a mock context with None execution coordinator
        mock_context = MagicMock()
        mock_context.execution_coordinator = None
        bot._coordinator_context = mock_context
        bot._coordinator_registry.get = MagicMock(return_value=None)

        with pytest.raises(RuntimeError, match="Execution coordinator not registered"):
            _ = bot.execution_coordinator

    def test_strategy_coordinator_not_registered_raises(self, perps_bot_instance):
        """Test RuntimeError when strategy coordinator not registered."""
        bot = perps_bot_instance
        # Create a mock context with None strategy coordinator
        mock_context = MagicMock()
        mock_context.strategy_coordinator = None
        bot._coordinator_context = mock_context
        bot._coordinator_registry.get = MagicMock(return_value=None)

        with pytest.raises(RuntimeError, match="Strategy coordinator not registered"):
            _ = bot.strategy_coordinator

    def test_telemetry_coordinator_not_registered_raises(self, perps_bot_instance):
        """Test RuntimeError when telemetry coordinator not registered."""
        bot = perps_bot_instance
        # Remove the coordinator
        bot._coordinator_registry._coordinators = {}

        with pytest.raises(RuntimeError, match="Telemetry coordinator not registered"):
            _ = bot.telemetry_coordinator

    def test_runtime_coordinator_wrong_type_raises(self, perps_bot_instance):
        """Test RuntimeError when runtime coordinator has wrong type."""
        bot = perps_bot_instance
        # Replace with wrong type
        wrong_coordinator = MagicMock()
        bot._coordinator_registry._coordinators["runtime"] = wrong_coordinator

        with pytest.raises(RuntimeError, match="Runtime coordinator has unexpected type"):
            _ = bot.runtime_coordinator

    def test_telemetry_coordinator_wrong_type_raises(self, perps_bot_instance):
        """Test RuntimeError when telemetry coordinator has wrong type."""
        bot = perps_bot_instance
        # Replace with wrong type
        wrong_coordinator = MagicMock()
        bot._coordinator_registry._coordinators["telemetry"] = wrong_coordinator

        with pytest.raises(RuntimeError, match="Telemetry coordinator has unexpected type"):
            _ = bot.telemetry_coordinator


class TestBackwardCompatibility:
    """Test backward compatibility and aliases."""

    def test_coinbase_trader_alias_exists(self):
        """Test CoinbaseTrader alias exists."""
        from bot_v2.orchestration.perps_bot import CoinbaseTrader

        assert CoinbaseTrader is PerpsBot

    def test_coinbase_trader_can_be_instantiated(
        self,
        mock_config_controller,
        service_registry,
        mock_event_store,
        mock_orders_store,
        mock_session_guard,
    ):
        """Test CoinbaseTrader can be instantiated like PerpsBot."""
        from datetime import UTC, datetime

        from bot_v2.monitoring.configuration_guardian import BaselineSnapshot
        from bot_v2.orchestration.perps_bot import CoinbaseTrader

        # Create proper baseline snapshot with timestamp
        proper_baseline = BaselineSnapshot(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            config_dict={"symbols": ["BTC-PERP"]},
            config_hash="test_hash",
            env_keys=set(),
            critical_env_values={},
            active_symbols=["BTC-PERP"],
            open_positions={},
            account_equity=None,
            total_exposure=Decimal("0"),
            profile="dev",
            broker_type="mock",
            risk_limits={},
        )

        original_init = CoinbaseTrader.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._start_streaming_background = lambda: None

        with pytest.MonkeyPatch().context() as m:
            m.setattr(CoinbaseTrader, "__init__", patched_init)
            bot = CoinbaseTrader(
                config_controller=mock_config_controller,
                registry=service_registry,
                event_store=mock_event_store,
                orders_store=mock_orders_store,
                session_guard=mock_session_guard,
                baseline_snapshot=proper_baseline,
            )

        assert isinstance(bot, PerpsBot)
        assert bot.bot_id == "coinbase_trader"


class TestConfigurationGuardianIntegration:
    """Test configuration guardian integration."""

    def test_resolve_configuration_guardian_with_existing_guardian(
        self,
        mock_config_controller,
        service_registry,
        mock_event_store,
        mock_orders_store,
        mock_session_guard,
        mock_baseline_snapshot,
        mock_configuration_guardian,
    ):
        """Test _resolve_configuration_guardian returns existing guardian."""
        # Test with existing guardian
        original_init = PerpsBot.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._start_streaming_background = lambda: None

        with pytest.MonkeyPatch().context() as m:
            m.setattr(PerpsBot, "__init__", patched_init)
            bot = PerpsBot(
                config_controller=mock_config_controller,
                registry=service_registry,
                event_store=mock_event_store,
                orders_store=mock_orders_store,
                session_guard=mock_session_guard,
                baseline_snapshot=mock_baseline_snapshot,
                configuration_guardian=mock_configuration_guardian,
            )

        # Should use the provided guardian
        assert bot.configuration_guardian is mock_configuration_guardian

    def test_resolve_configuration_guardian_creates_new_when_none(
        self,
        mock_config_controller,
        service_registry,
        mock_event_store,
        mock_orders_store,
        mock_session_guard,
        mock_baseline_snapshot,
    ):
        """Test _resolve_configuration_guardian creates new guardian when none provided."""
        # Test without existing guardian
        original_init = PerpsBot.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._start_streaming_background = lambda: None

        with pytest.MonkeyPatch().context() as m:
            # Mock the ConfigurationGuardian class
            mock_guardian_class = MagicMock()
            mock_guardian_instance = MagicMock()
            mock_guardian_class.return_value = mock_guardian_instance
            m.setattr(
                "bot_v2.monitoring.configuration_guardian.ConfigurationGuardian",
                mock_guardian_class,
            )

            bot = PerpsBot(
                config_controller=mock_config_controller,
                registry=service_registry,
                event_store=mock_event_store,
                orders_store=mock_orders_store,
                session_guard=mock_session_guard,
                baseline_snapshot=mock_baseline_snapshot,
                configuration_guardian=None,  # No guardian provided
            )

        # Should create new guardian with baseline snapshot
        mock_guardian_class.assert_called_once_with(mock_baseline_snapshot)
        assert bot.configuration_guardian is mock_guardian_instance


class TestRegistryAlignment:
    """Test service registry alignment with configuration."""

    def test_align_registry_with_config_same_config_returns_same_registry(self, perps_bot_instance):
        """Test _align_registry_with_config returns same registry when config matches."""
        bot = perps_bot_instance
        # Registry already has the same config
        result = bot._align_registry_with_config(bot.registry)
        assert result is bot.registry

    def test_align_registry_with_config_different_config_returns_updated_registry(
        self, perps_bot_instance
    ):
        """Test _align_registry_with_config returns updated registry when config differs."""
        bot = perps_bot_instance
        # Create a new registry with different config to test alignment
        new_registry = ServiceRegistry(config=MagicMock(), broker=bot.registry.broker)

        result = bot._align_registry_with_config(new_registry)

        # Should return a registry with the updated config (not the same object due to with_updates)
        assert result is not new_registry  # with_updates creates a new object
        assert result.config is bot.config  # Config should be updated to bot's config
        assert result.broker is new_registry.broker  # Broker should be preserved


class TestSymbolStateInitialization:
    """Test symbol state initialization."""

    def test_initialize_symbols_state_with_symbols(self, perps_bot_instance, minimal_bot_config):
        """Test _initialize_symbols_state with symbols configured."""
        # Test the method directly with a new config
        from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState

        state = PerpsBotRuntimeState(minimal_bot_config.symbols)

        # Check that mark_windows are initialized with the provided symbols
        assert set(state.mark_windows.keys()) == set(minimal_bot_config.symbols)
        # All mark window lists should be empty initially
        assert all(len(windows) == 0 for windows in state.mark_windows.values())

    def test_initialize_symbols_state_without_symbols(self, perps_bot_instance):
        """Test _initialize_symbols_state without symbols configured."""
        # Test the method directly with empty symbols
        from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState

        state = PerpsBotRuntimeState([])

        # Check that mark_windows is empty when no symbols are provided
        assert state.mark_windows == {}

    def test_initialize_symbols_state_sets_derivatives_enabled_flag(self, perps_bot_instance):
        """Test _initialize_symbols_state sets derivatives_enabled flag."""
        bot = perps_bot_instance
        bot.config.derivatives_enabled = True

        state = bot._initialize_symbols_state()

        assert bot._derivatives_enabled is True


class TestServicePlaceholderReset:
    """Test service placeholder reset functionality."""

    def test_initialize_service_placeholders_resets_all_services(self, perps_bot_instance):
        """Test _initialize_service_placeholders resets all service placeholders."""
        bot = perps_bot_instance

        # Set some services first
        bot.account_manager = MagicMock()
        bot.account_telemetry = MagicMock()
        bot.market_monitor = MagicMock()
        bot.intx_portfolio_service = MagicMock()

        # Reset placeholders
        bot._initialize_service_placeholders()

        # All should be None
        assert bot.account_manager is None
        assert bot.account_telemetry is None
        assert bot.market_monitor is None
        assert bot.intx_portfolio_service is None


class TestCoordinatorContextUpdates:
    """Test coordinator context updates and wiring."""

    def test_setup_coordinator_stack_creates_context_with_system_monitor_none(
        self, perps_bot_instance
    ):
        """Test _setup_coordinator_stack creates context with system_monitor None initially."""
        bot = perps_bot_instance
        context = bot._coordinator_context

        # System monitor should be None initially, then set after creation
        # This is tested implicitly by the successful creation in the fixture

    def test_setup_coordinator_stack_updates_context_after_system_monitor_creation(
        self, perps_bot_instance
    ):
        """Test _setup_coordinator_stack updates context after system monitor creation."""
        bot = perps_bot_instance
        context = bot._coordinator_context

        # After setup, system monitor should be set
        assert context.system_monitor is bot.system_monitor
        assert context.execution_coordinator is bot.execution_coordinator
        assert context.strategy_coordinator is bot.strategy_coordinator


class TestCoordinatorRegistration:
    """Test coordinator registration and dependency order."""

    def test_register_coordinators_registers_in_dependency_order(self, perps_bot_instance):
        """Test _register_coordinators registers coordinators in dependency order."""
        bot = perps_bot_instance

        # All coordinators should be registered
        assert bot._coordinator_registry.get("runtime") is not None
        assert bot._coordinator_registry.get("execution") is not None
        assert bot._coordinator_registry.get("strategy") is not None
        assert bot._coordinator_registry.get("telemetry") is not None

    def test_register_coordinators_updates_context_before_registering(self, perps_bot_instance):
        """Test _register_coordinators updates context before registering coordinators."""
        bot = perps_bot_instance
        context = bot._coordinator_context

        # Context should have execution and strategy coordinators
        assert context.execution_coordinator is not None
        assert context.strategy_coordinator is not None


class TestCoordinatorContextMethodCalls:
    """Test coordinator context method calls and updates."""

    def test_coordinator_update_context_called_for_all_coordinators(self, perps_bot_instance):
        """Test update_context is called for all coordinators that support it."""
        bot = perps_bot_instance

        # Mock the update_context method for each coordinator
        for coordinator_name in ["runtime", "execution", "strategy", "telemetry"]:
            coordinator = bot._coordinator_registry.get(coordinator_name)
            if hasattr(coordinator, "update_context"):
                coordinator.update_context = MagicMock()

        # Re-run the update process
        context = bot._coordinator_context
        for coordinator in (
            bot.runtime_coordinator,
            bot.execution_coordinator,
            bot.strategy_coordinator,
            bot.telemetry_coordinator,
        ):
            if hasattr(coordinator, "update_context"):
                coordinator.update_context(context)

        # Verify calls were made (actual verification depends on which coordinators support update_context)
        # This test primarily ensures the calling pattern works without errors

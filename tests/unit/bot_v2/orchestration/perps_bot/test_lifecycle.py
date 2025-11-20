"""Tests for PerpsBot lifecycle, initialization, and coordinator setup."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from bot_v2.orchestration.engines import (
    ExecutionEngine,
    RuntimeCoordinator,
    TradingEngine,
    TelemetryCoordinator,
)
from bot_v2.orchestration.perps_bot import PerpsBot


class TestPerpsBotInitialization:
    """Test PerpsBot initialization and component setup."""

    def test_perps_bot_initialization_with_required_components(self, perps_bot_instance):
        """Test PerpsBot initializes correctly with all required components."""
        bot = perps_bot_instance

        # Verify basic attributes
        assert bot.bot_id == "coinbase_trader"
        assert bot.start_time is not None
        assert not bot.running
        assert isinstance(bot.start_time, datetime)

        # Verify core dependencies are set
        assert bot.config_controller is not None
        assert bot.config is not None
        assert bot.registry is not None
        assert bot.event_store is not None
        assert bot.orders_store is not None
        assert bot._session_guard is not None
        assert bot.baseline_snapshot is not None
        assert bot.configuration_guardian is not None

        # Verify components are initialized
        assert bot.strategy_orchestrator is not None
        assert bot.system_monitor is not None
        assert bot.lifecycle_manager is not None
        assert bot._state is not None

    def test_perps_bot_initializes_symbols_from_config(
        self, perps_bot_instance, minimal_bot_config
    ):
        """Test PerpsBot initializes symbols from configuration."""
        bot = perps_bot_instance
        assert bot.symbols == minimal_bot_config.symbols

    def test_perps_bot_initializes_derivatives_enabled_flag(
        self, perps_bot_instance, minimal_bot_config
    ):
        """Test PerpsBot sets derivatives_enabled flag from config."""
        bot = perps_bot_instance
        assert bot._derivatives_enabled == bool(
            getattr(minimal_bot_config, "derivatives_enabled", False)
        )

    def test_perps_bot_handles_empty_symbols_config(
        self,
        mock_config_controller,
        service_registry,
        mock_event_store,
        mock_orders_store,
        mock_session_guard,
        mock_baseline_snapshot,
    ):
        """Test PerpsBot handles empty symbols configuration gracefully."""
        # Create config with empty symbols
        empty_config = MagicMock()
        empty_config.symbols = []
        empty_config.derivatives_enabled = False
        mock_config_controller.current = empty_config

        # Create bot with empty symbols
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
            )

        assert bot.symbols == []

    def test_perps_bot_registry_alignment(self, perps_bot_instance, service_registry):
        """Test PerpsBot aligns registry with current config."""
        bot = perps_bot_instance
        assert bot.registry.config is bot.config

    def test_perps_bot_configuration_guardian_resolution(
        self, perps_bot_instance, mock_configuration_guardian
    ):
        """Test PerpsBot resolves configuration guardian correctly."""
        bot = perps_bot_instance
        assert bot.configuration_guardian is mock_configuration_guardian

    def test_perps_bot_configuration_guardian_creation_when_none(
        self,
        mock_config_controller,
        service_registry,
        mock_event_store,
        mock_orders_store,
        mock_session_guard,
        mock_baseline_snapshot,
    ):
        """Test PerpsBot creates configuration guardian when none provided."""
        # Create bot without configuration guardian
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
                configuration_guardian=None,  # No guardian provided
            )

        # Should create a new configuration guardian
        assert bot.configuration_guardian is not None
        assert hasattr(bot.configuration_guardian, "reset_baseline")


class TestCoordinatorSetup:
    """Test coordinator initialization and wiring."""

    def test_coordinator_context_setup(self, perps_bot_instance):
        """Test coordinator context is properly set up."""
        bot = perps_bot_instance
        context = bot._coordinator_context

        assert context.config is bot.config
        assert context.registry is bot.registry
        assert context.event_store is bot.event_store
        assert context.orders_store is bot.orders_store
        assert context.symbols == tuple(bot.symbols)
        assert context.bot_id == bot.bot_id
        assert context.runtime_state is bot._state
        assert context.config_controller is bot.config_controller
        assert context.strategy_orchestrator is bot.strategy_orchestrator
        assert context.session_guard is bot._session_guard
        assert context.configuration_guardian is bot.configuration_guardian
        assert context.system_monitor is bot.system_monitor

    def test_coordinator_registry_creation(self, perps_bot_instance):
        """Test coordinator registry is created and populated."""
        bot = perps_bot_instance
        registry = bot._coordinator_registry

        assert registry is not None
        assert registry.context is bot._coordinator_context

    def test_all_coordinators_registered(self, perps_bot_instance):
        """Test all required coordinators are registered."""
        bot = perps_bot_instance

        # Check that we can get all coordinators
        assert bot.runtime_coordinator is not None
        assert isinstance(bot.runtime_coordinator, RuntimeCoordinator)

        assert bot.execution_coordinator is not None
        assert isinstance(bot.execution_coordinator, ExecutionEngine)

        assert bot.strategy_coordinator is not None
        assert isinstance(bot.strategy_coordinator, TradingEngine)

        assert bot.telemetry_coordinator is not None
        assert isinstance(bot.telemetry_coordinator, TelemetryCoordinator)

    def test_coordinator_context_updates(self, perps_bot_instance):
        """Test coordinator context is properly updated after coordinator registration."""
        bot = perps_bot_instance
        context = bot._coordinator_context

        # Should have execution and strategy coordinators set
        assert context.execution_coordinator is bot.execution_coordinator
        assert context.strategy_coordinator is bot.strategy_coordinator

    def test_service_placeholders_initialization(self, perps_bot_instance):
        """Test service placeholders are reset during initialization."""
        bot = perps_bot_instance

        assert bot.account_manager is None
        assert bot.account_telemetry is None
        assert bot.market_monitor is None
        assert bot.intx_portfolio_service is None


class TestPropertyAccessors:
    """Test PerpsBot property accessors and state management."""

    def test_runtime_state_property(self, perps_bot_instance):
        """Test runtime_state property returns correct state."""
        bot = perps_bot_instance
        assert bot.runtime_state is bot._state

    def test_runtime_coordinator_property(self, perps_bot_instance):
        """Test runtime_coordinator property returns correct coordinator."""
        bot = perps_bot_instance
        coordinator = bot.runtime_coordinator
        assert isinstance(coordinator, RuntimeCoordinator)

    def test_execution_coordinator_property(self, perps_bot_instance):
        """Test execution_coordinator property returns correct coordinator."""
        bot = perps_bot_instance
        coordinator = bot.execution_coordinator
        assert isinstance(coordinator, ExecutionEngine)

    def test_strategy_coordinator_property(self, perps_bot_instance):
        """Test strategy_coordinator property returns correct coordinator."""
        bot = perps_bot_instance
        coordinator = bot.strategy_coordinator
        assert isinstance(coordinator, TradingEngine)

    def test_telemetry_coordinator_property(self, perps_bot_instance):
        """Test telemetry_coordinator property returns correct coordinator."""
        bot = perps_bot_instance
        coordinator = bot.telemetry_coordinator
        assert isinstance(coordinator, TelemetryCoordinator)

    def test_settings_property_with_registry_settings(
        self, perps_bot_instance, mock_runtime_settings
    ):
        """Test settings property returns registry settings when available."""
        bot = perps_bot_instance
        # Mock the registry to have runtime_settings
        mock_registry = MagicMock()
        mock_registry.runtime_settings = mock_runtime_settings
        bot.registry = mock_registry

        settings = bot.settings
        assert settings is mock_runtime_settings

    def test_settings_property_creates_default(self, perps_bot_instance):
        """Test settings property creates default settings when not in registry."""
        bot = perps_bot_instance
        # Mock the registry to have no runtime_settings
        mock_registry = MagicMock()
        mock_registry.runtime_settings = None
        mock_registry.with_updates = MagicMock(return_value=mock_registry)
        bot.registry = mock_registry

        settings = bot.settings
        assert settings is not None
        # Check that with_updates was called to set the settings
        mock_registry.with_updates.assert_called_once()

    def test_mark_windows_property(self, perps_bot_instance):
        """Test mark_windows property returns state data."""
        bot = perps_bot_instance
        mark_windows = bot.mark_windows
        assert isinstance(mark_windows, dict)
        # Should have windows for configured symbols
        for symbol in bot.symbols:
            assert symbol in mark_windows

    def test_last_decisions_property(self, perps_bot_instance):
        """Test last_decisions property returns state data."""
        bot = perps_bot_instance
        decisions = bot.last_decisions
        assert isinstance(decisions, dict)

    def test_last_positions_property(self, perps_bot_instance):
        """Test _last_positions property returns state data."""
        bot = perps_bot_instance
        positions = bot._last_positions
        assert isinstance(positions, dict)

    def test_order_stats_property(self, perps_bot_instance):
        """Test order_stats property returns state data."""
        bot = perps_bot_instance
        stats = bot.order_stats
        assert isinstance(stats, dict)

    def test_product_map_property(self, perps_bot_instance):
        """Test _product_map property returns state data."""
        bot = perps_bot_instance
        product_map = bot._product_map
        assert isinstance(product_map, dict)

    def test_order_lock_property(self, perps_bot_instance):
        """Test _order_lock property returns state data."""
        bot = perps_bot_instance
        order_lock = bot._order_lock
        # Can be None initially
        assert order_lock is None or isinstance(order_lock, asyncio.Lock)

    def test_mark_lock_property(self, perps_bot_instance):
        """Test _mark_lock property returns state data."""
        bot = perps_bot_instance
        mark_lock = bot._mark_lock
        # Check that it's a proper RLock (threading.RLock is actually _thread.RLock)
        assert hasattr(mark_lock, "acquire") and hasattr(mark_lock, "release")
        assert mark_lock.__class__.__name__ == "RLock"

    def test_symbol_strategies_property(self, perps_bot_instance):
        """Test _symbol_strategies property returns state data."""
        bot = perps_bot_instance
        strategies = bot._symbol_strategies
        assert isinstance(strategies, dict)

    def test_strategy_property(self, perps_bot_instance):
        """Test strategy property returns state data."""
        bot = perps_bot_instance
        strategy = bot.strategy
        # Can be None initially
        assert strategy is None or hasattr(strategy, "__call__")

    def test_exec_engine_property(self, perps_bot_instance):
        """Test _exec_engine property returns state data."""
        bot = perps_bot_instance
        exec_engine = bot._exec_engine
        # Can be None initially
        assert exec_engine is None or hasattr(exec_engine, "place_order")

    def test_process_symbol_dispatch_property(self, perps_bot_instance):
        """Test _process_symbol_dispatch property returns state data."""
        bot = perps_bot_instance
        dispatch = bot._process_symbol_dispatch
        # Can be None initially
        assert dispatch is None or hasattr(dispatch, "__call__")

    def test_process_symbol_needs_context_property(self, perps_bot_instance):
        """Test _process_symbol_needs_context property returns state data."""
        bot = perps_bot_instance
        needs_context = bot._process_symbol_needs_context
        # Can be None initially
        assert needs_context is None or isinstance(needs_context, bool)


class TestBrokerAndRiskManagerProperties:
    """Test broker and risk manager property accessors."""

    def test_broker_property_get(self, perps_bot_instance, mock_brokerage):
        """Test broker property returns registry broker."""
        bot = perps_bot_instance
        broker = bot.broker
        assert broker is mock_brokerage

    def test_broker_property_set(self, perps_bot_instance, mock_brokerage):
        """Test broker property setter updates registry."""
        bot = perps_bot_instance
        new_broker = MagicMock()
        bot.broker = new_broker
        assert bot.registry.broker is new_broker

    def test_broker_property_raises_when_not_configured(self, perps_bot_instance):
        """Test broker property raises when no broker in registry."""
        bot = perps_bot_instance
        # Mock the registry to have no broker
        mock_registry = MagicMock()
        mock_registry.broker = None
        bot.registry = mock_registry

        with pytest.raises(RuntimeError, match="Broker is not configured"):
            _ = bot.broker

    def test_risk_manager_property_get(self, perps_bot_instance, mock_risk_manager):
        """Test risk_manager property returns registry risk manager."""
        bot = perps_bot_instance
        risk_manager = bot.risk_manager
        assert risk_manager is mock_risk_manager

    def test_risk_manager_property_set(self, perps_bot_instance, mock_risk_manager):
        """Test risk_manager property setter updates registry."""
        bot = perps_bot_instance
        new_risk_manager = MagicMock()
        bot.risk_manager = new_risk_manager
        assert bot.registry.risk_manager is new_risk_manager

    def test_risk_manager_property_raises_when_not_configured(self, perps_bot_instance):
        """Test risk_manager property raises when no risk manager in registry."""
        bot = perps_bot_instance
        # Mock the registry to have no risk_manager
        mock_registry = MagicMock()
        mock_registry.risk_manager = None
        bot.registry = mock_registry

        with pytest.raises(RuntimeError, match="Risk manager is not configured"):
            _ = bot.risk_manager


class TestExecutionEngineProperties:
    """Test execution engine and service properties."""

    def test_exec_engine_property_raises_when_not_initialized(self, perps_bot_instance):
        """Test exec_engine property raises when not initialized."""
        bot = perps_bot_instance
        bot._state.exec_engine = None
        with pytest.raises(RuntimeError, match="Execution engine not initialized"):
            _ = bot.exec_engine

    def test_exec_engine_property_returns_when_set(self, perps_bot_instance):
        """Test exec_engine property returns when set."""
        bot = perps_bot_instance
        mock_engine = MagicMock()
        bot._state.exec_engine = mock_engine
        exec_engine = bot.exec_engine
        assert exec_engine is mock_engine

    def test_account_manager_property_setter_getter(self, perps_bot_instance):
        """Test account_manager property setter and getter."""
        bot = perps_bot_instance
        assert bot.account_manager is None

        mock_manager = MagicMock()
        bot.account_manager = mock_manager
        assert bot.account_manager is mock_manager
        assert bot._state.account_manager is mock_manager

    def test_account_telemetry_property_setter_getter(self, perps_bot_instance):
        """Test account_telemetry property setter and getter."""
        bot = perps_bot_instance
        assert bot.account_telemetry is None

        mock_telemetry = MagicMock()
        bot.account_telemetry = mock_telemetry
        assert bot.account_telemetry is mock_telemetry
        assert bot._state.account_telemetry is mock_telemetry

    def test_market_monitor_property_setter_getter(self, perps_bot_instance):
        """Test market_monitor property setter and getter with fallback."""
        bot = perps_bot_instance
        assert bot.market_monitor is None

        # Test setting through state
        mock_monitor = MagicMock()
        bot.market_monitor = mock_monitor
        assert bot.market_monitor is mock_monitor
        assert bot._state.market_monitor is mock_monitor

        # Test setting through __dict__ fallback
        bot.market_monitor = None
        fallback_monitor = MagicMock()
        bot.__dict__["_market_monitor"] = fallback_monitor
        assert bot.market_monitor is fallback_monitor

        # Test clearing both
        bot.market_monitor = None
        assert bot.market_monitor is None
        assert "_market_monitor" not in bot.__dict__

    def test_intx_portfolio_service_property_setter_getter(self, perps_bot_instance):
        """Test intx_portfolio_service property setter and getter."""
        bot = perps_bot_instance
        assert bot.intx_portfolio_service is None

        mock_service = MagicMock()
        bot.intx_portfolio_service = mock_service
        assert bot.intx_portfolio_service is mock_service
        assert bot._state.intx_portfolio_service is mock_service


class TestBaselineSnapshotCreation:
    """Test baseline snapshot creation functionality."""

    def test_build_baseline_snapshot_creates_correct_structure(self, minimal_bot_config):
        """Test build_baseline_snapshot creates correct structure."""
        from bot_v2.monitoring.configuration_guardian import BaselineSnapshot

        snapshot = PerpsBot.build_baseline_snapshot(minimal_bot_config, derivatives_enabled=True)

        assert isinstance(snapshot, BaselineSnapshot)
        assert hasattr(snapshot, "config_dict")
        assert hasattr(snapshot, "active_symbols")
        assert hasattr(
            snapshot, "open_positions"
        )  # Note: field is 'open_positions' not 'positions'
        assert hasattr(snapshot, "account_equity")
        assert hasattr(snapshot, "profile")
        assert hasattr(snapshot, "broker_type")
        # settings field doesn't exist in BaselineSnapshot

    def test_build_baseline_snapshot_sets_active_symbols(self, minimal_bot_config):
        """Test build_baseline_snapshot sets active symbols from config."""
        snapshot = PerpsBot.build_baseline_snapshot(minimal_bot_config, derivatives_enabled=False)

        assert snapshot.active_symbols == minimal_bot_config.symbols
        assert snapshot.profile == minimal_bot_config.profile

    def test_build_baseline_snapshot_sets_broker_type_from_config(self, minimal_bot_config):
        """Test build_baseline_snapshot sets broker type from config."""
        snapshot = PerpsBot.build_baseline_snapshot(minimal_bot_config, derivatives_enabled=False)

        if minimal_bot_config.mock_broker:
            assert snapshot.broker_type == "mock"
        else:
            assert snapshot.broker_type == "live"

    def test_build_baseline_snapshot_includes_derivatives_enabled(self, minimal_bot_config):
        """Test build_baseline_snapshot includes derivatives_enabled flag."""
        snapshot = PerpsBot.build_baseline_snapshot(minimal_bot_config, derivatives_enabled=True)

        # Should be reflected in the config_dict
        config_dict = snapshot.config_dict
        assert config_dict.get("derivatives_enabled") is True

    def test_build_baseline_snapshot_includes_runtime_settings(self, minimal_bot_config):
        """Test build_baseline snapshot basic functionality.

        Note: BaselineSnapshot doesn't include runtime settings directly in the snapshot object.
        Runtime settings are accessed via perps_bot.settings property when needed.
        """
        snapshot = PerpsBot.build_baseline_snapshot(minimal_bot_config, derivatives_enabled=False)

        # BaselineSnapshot should have basic structure without runtime settings
        assert isinstance(snapshot.config_dict, dict)
        assert snapshot.active_symbols == minimal_bot_config.symbols

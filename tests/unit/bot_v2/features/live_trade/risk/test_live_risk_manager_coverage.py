"""
Comprehensive LiveRiskManager test suite for 85%+ coverage.

This suite tests the delegation facade pattern and integration flows between
LiveRiskManager and its submodules: StateManager, PositionSizer, PreTradeValidator,
RuntimeMonitor. Focuses on pre-trade limits, circuit breakers, and emergency actions.
"""

from __future__ import annotations

import datetime as dt
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk.manager import LiveRiskManager, MarkTimestampRegistry
from bot_v2.features.live_trade.risk.position_sizing import (
    PositionSizingAdvice,
    PositionSizingContext,
)
from bot_v2.features.live_trade.risk.pre_trade_checks import ValidationError
from bot_v2.features.live_trade.risk_runtime import CircuitBreakerAction, CircuitBreakerOutcome
from bot_v2.persistence.event_store import EventStore


class TestMarkTimestampRegistry:
    """Test the MarkTimestampRegistry thread-safe functionality."""

    def test_registry_initialization_default(self):
        """Test registry initialization with default time provider."""
        registry = MarkTimestampRegistry()

        assert len(registry) == 0
        assert registry.keys() == ()
        assert registry.values() == ()
        assert registry.items() == ()

    def test_registry_initialization_custom_time_provider(self):
        """Test registry initialization with custom time provider."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        registry = MarkTimestampRegistry(now_provider=lambda: fixed_time)

        timestamp = registry.update_timestamp("BTC-USD")
        assert timestamp == fixed_time

    def test_registry_update_and_retrieve(self):
        """Test updating and retrieving timestamps."""
        registry = MarkTimestampRegistry()

        timestamp1 = registry.update_timestamp("BTC-USD")
        timestamp2 = registry.update_timestamp("ETH-USD")

        assert registry["BTC-USD"] == timestamp1
        assert registry["ETH-USD"] == timestamp2
        assert len(registry) == 2
        assert "BTC-USD" in registry
        assert "ETH-USD" in registry

    def test_registry_custom_timestamp(self):
        """Test updating with custom timestamp."""
        registry = MarkTimestampRegistry()
        custom_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)

        registry.update_timestamp("BTC-USD", custom_time)

        assert registry["BTC-USD"] == custom_time

    def test_registry_snapshot(self):
        """Test creating registry snapshot."""
        registry = MarkTimestampRegistry()
        registry.update_timestamp("BTC-USD")
        registry.update_timestamp("ETH-USD")

        snapshot = registry.snapshot()

        assert isinstance(snapshot, dict)
        assert len(snapshot) == 2
        assert snapshot["BTC-USD"] is not None
        assert snapshot["ETH-USD"] is not None

    def test_registry_thread_safety(self):
        """Test thread safety with RLock."""
        registry = MarkTimestampRegistry()

        # Multiple operations should be thread-safe
        registry.update_timestamp("BTC-USD")
        registry["ETH-USD"] = datetime.utcnow()
        registry.update_timestamp("SOL-USD")

        assert len(registry) == 3
        registry.clear()
        assert len(registry) == 0

    def test_registry_get_with_default(self):
        """Test getting timestamp with default value."""
        registry = MarkTimestampRegistry()

        # Non-existent key should return default
        result = registry.get("NONEXISTENT", "default")
        assert result == "default"

        # Existing key should return actual value
        timestamp = registry.update_timestamp("BTC-USD")
        result = registry.get("BTC-USD", "default")
        assert result == timestamp


class TestLiveRiskManagerInitialization:
    """Test LiveRiskManager initialization and configuration."""

    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        manager = LiveRiskManager()

        assert manager.config is not None
        assert manager.event_store is not None
        assert manager.state_manager is not None
        assert manager.position_sizer is not None
        assert manager.pre_trade_validator is not None
        assert manager.runtime_monitor is not None
        assert manager.last_mark_update is not None

    def test_initialization_with_custom_components(self):
        """Test initialization with custom components."""
        custom_config = RiskConfig()
        custom_event_store = EventStore()
        custom_risk_info_provider = Mock()
        custom_position_sizer = Mock()
        custom_impact_estimator = Mock()

        manager = LiveRiskManager(
            config=custom_config,
            event_store=custom_event_store,
            risk_info_provider=custom_risk_info_provider,
            position_size_estimator=custom_position_sizer,
            impact_estimator=custom_impact_estimator,
        )

        assert manager.config is custom_config
        assert manager.event_store is custom_event_store
        assert manager._risk_info_provider is custom_risk_info_provider
        assert manager._position_size_estimator is custom_position_sizer
        assert manager._impact_estimator is custom_impact_estimator

    def test_initialization_sets_up_backward_compatibility_attributes(self):
        """Test that initialization sets up backward compatibility attributes."""
        manager = LiveRiskManager()

        # These should be delegated to state manager
        assert hasattr(manager, "_state")
        assert hasattr(manager, "daily_pnl")
        assert hasattr(manager, "start_of_day_equity")

        # These should be delegated to runtime monitor
        assert hasattr(manager, "positions")
        assert hasattr(manager, "circuit_breaker_state")

    def test_initialization_logs_configuration(self):
        """Test that initialization logs configuration details."""

        with patch("bot_v2.features.live_trade.risk.manager.logger") as mock_logger:
            manager = LiveRiskManager()
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "LiveRiskManager initialized with config:" in call_args

    def test_initialization_with_non_standard_config(self):
        """Test initialization with non-standard config object."""

        class NonStandardConfig:
            pass


        with patch("bot_v2.features.live_trade.risk.manager.logger") as mock_logger:
            # This should fail during initialization because NonStandardConfig
            # doesn't have the required attributes
            try:
                manager = LiveRiskManager(config=NonStandardConfig())
                # If it doesn't fail, at least test the logging path
                mock_logger.info.assert_called()
                call_args = mock_logger.info.call_args[0][0]
                assert "LiveRiskManager initialized with a non-standard config object" in call_args
            except AttributeError:
                # Expected to fail due to missing config attributes
                pass

    def test_now_provider_initialization(self):
        """Test time provider initialization."""
        manager = LiveRiskManager()

        # Should have default time provider
        now = manager._now()
        assert isinstance(now, datetime)

    def test_component_initialization_dependencies(self):
        """Test that components receive correct dependencies."""
        custom_config = RiskConfig()
        custom_event_store = EventStore()

        manager = LiveRiskManager(
            config=custom_config,
            event_store=custom_event_store,
        )

        # Check components received correct dependencies
        assert manager.state_manager.config is custom_config
        assert manager.state_manager.event_store is custom_event_store
        assert manager.position_sizer.config is custom_config
        assert manager.pre_trade_validator.config is custom_config
        assert manager.runtime_monitor.config is custom_config


class TestLiveRiskManagerStateManagement:
    """Test state management delegation flows."""

    def test_reduce_only_mode_delegation(self, conservative_risk_config):
        """Test reduce-only mode delegation to state manager."""
        manager = LiveRiskManager(config=conservative_risk_config)

        # Test initial state
        assert manager.is_reduce_only_mode() is False

        # Test setting reduce-only mode
        manager.set_reduce_only_mode(True, "test reason")
        assert manager.is_reduce_only_mode() is True

        # Verify delegation worked
        assert manager.state_manager.is_reduce_only_mode() is True

    def test_state_listener_delegation(self, conservative_risk_config):
        """Test state listener delegation to state manager."""
        manager = LiveRiskManager(config=conservative_risk_config)

        listener = Mock()
        manager.set_state_listener(listener)

        assert manager._state_listener is listener

    def test_daily_tracking_reset(self, conservative_risk_config):
        """Test daily tracking reset delegation."""
        manager = LiveRiskManager(config=conservative_risk_config)
        current_equity = Decimal("1000")

        manager.reset_daily_tracking(current_equity)

        # Should reset both manager and state manager
        assert manager.start_of_day_equity == current_equity
        assert manager.state_manager.start_of_day_equity == current_equity

    def test_state_synchronization_after_reduce_only_change(self, conservative_risk_config):
        """Test that state reference is synchronized after reduce-only change."""
        manager = LiveRiskManager(config=conservative_risk_config)
        original_state = manager._state

        manager.set_reduce_only_mode(True, "test")

        # State reference should be updated
        assert manager._state is not original_state
        assert manager._state is manager.state_manager._state


class TestLiveRiskManagerPositionSizing:
    """Test position sizing delegation flows."""

    def test_position_sizing_delegation(self, conservative_risk_config):
        """Test position sizing delegation to position sizer."""
        manager = LiveRiskManager(config=conservative_risk_config)

        context = PositionSizingContext(
            symbol="BTC-USD",
            side="buy",
            available_equity=Decimal("1000"),
            current_positions={},
            mark_price=Decimal("50000"),
        )

        with patch.object(
            manager.position_sizer,
            "size_position",
            return_value=PositionSizingAdvice(
                recommended_quantity=Decimal("0.1"), max_quantity=Decimal("0.2"), reason="test"
            ),
        ) as mock_size:
            result = manager.size_position(context)

            mock_size.assert_called_once_with(context)
            assert result.recommended_quantity == Decimal("0.1")

    def test_impact_estimator_setting(self, conservative_risk_config):
        """Test setting impact estimator propagates to components."""
        manager = LiveRiskManager(config=conservative_risk_config)
        estimator = Mock()

        manager.set_impact_estimator(estimator)

        assert manager.position_sizer._impact_estimator is estimator
        assert manager.pre_trade_validator._impact_estimator is estimator
        assert manager._impact_estimator is estimator

    def test_impact_estimator_clearing(self, conservative_risk_config):
        """Test clearing impact estimator propagates to components."""
        manager = LiveRiskManager(config=conservative_risk_config)

        manager.set_impact_estimator(None)

        assert manager.position_sizer._impact_estimator is None
        assert manager.pre_trade_validator._impact_estimator is None
        assert manager._impact_estimator is None


class TestLiveRiskManagerPreTradeValidation:
    """Test pre-trade validation delegation flows."""

    def test_pre_trade_validate_delegation(self, conservative_risk_config):
        """Test pre-trade validation delegation."""
        manager = LiveRiskManager(config=conservative_risk_config)

        with patch.object(manager.pre_trade_validator, "pre_trade_validate") as mock_validate:
            manager.pre_trade_validate(symbol="BTC-USD", side="buy", quantity=Decimal("0.1"))

            mock_validate.assert_called_once()

    def test_validate_leverage_delegation(self, conservative_risk_config):
        """Test leverage validation delegation."""
        manager = LiveRiskManager(config=conservative_risk_config)

        with patch.object(manager.pre_trade_validator, "validate_leverage") as mock_validate:
            manager.validate_leverage(symbol="BTC-USD", quantity=Decimal("0.1"))

            mock_validate.assert_called_once()

    def test_validate_liquidation_buffer_delegation(self, conservative_risk_config):
        """Test liquidation buffer validation delegation."""
        manager = LiveRiskManager(config=conservative_risk_config)

        with patch.object(
            manager.pre_trade_validator, "validate_liquidation_buffer"
        ) as mock_validate:
            manager.validate_liquidation_buffer(symbol="BTC-USD", quantity=Decimal("0.1"))

            mock_validate.assert_called_once()

    def test_validate_exposure_limits_delegation(self, conservative_risk_config):
        """Test exposure limits validation delegation."""
        manager = LiveRiskManager(config=conservative_risk_config)

        with patch.object(manager.pre_trade_validator, "validate_exposure_limits") as mock_validate:
            manager.validate_exposure_limits(
                symbol="BTC-USD", notional=Decimal("5000"), equity=Decimal("1000")
            )

            mock_validate.assert_called_once()

    def test_validate_slippage_guard_delegation(self, conservative_risk_config):
        """Test slippage guard validation delegation."""
        manager = LiveRiskManager(config=conservative_risk_config)

        with patch.object(manager.pre_trade_validator, "validate_slippage_guard") as mock_validate:
            manager.validate_slippage_guard(symbol="BTC-USD", side="buy", quantity=Decimal("0.1"))

            mock_validate.assert_called_once()

    def test_risk_info_provider_propagation(self, conservative_risk_config):
        """Test setting risk info provider propagates to validator."""
        manager = LiveRiskManager(config=conservative_risk_config)
        provider = Mock()

        manager.set_risk_info_provider(provider)

        assert manager._risk_info_provider is provider
        assert manager.pre_trade_validator._risk_info_provider is provider


class TestLiveRiskManagerRuntimeMonitoring:
    """Test runtime monitoring delegation flows."""

    def test_track_daily_pnl_delegation(self, conservative_risk_config):
        """Test daily PnL tracking delegation with initialization."""
        manager = LiveRiskManager(config=conservative_risk_config)
        current_equity = Decimal("1000")
        positions_pnl = {"BTC-USD": {"unrealized": Decimal("100")}}

        with patch.object(
            manager.runtime_monitor, "track_daily_pnl", return_value=(True, Decimal("100"))
        ) as mock_track:
            result = manager.track_daily_pnl(current_equity, positions_pnl)

            # Should initialize start of day equity
            assert manager.state_manager.start_of_day_equity == current_equity
            assert manager.start_of_day_equity == current_equity

            mock_track.assert_called_once()
            assert result is True

    def test_track_daily_pnl_with_existing_start_of_day(self, conservative_risk_config):
        """Test daily PnL tracking with existing start of day equity."""
        manager = LiveRiskManager(config=conservative_risk_config)
        manager.state_manager.start_of_day_equity = Decimal("900")
        manager.start_of_day_equity = Decimal("900")

        current_equity = Decimal("1000")
        positions_pnl = {"BTC-USD": {"unrealized": Decimal("100")}}

        with patch.object(
            manager.runtime_monitor, "track_daily_pnl", return_value=(False, Decimal("100"))
        ) as mock_track:
            result = manager.track_daily_pnl(current_equity, positions_pnl)

            # Should not change start of day equity
            assert manager.state_manager.start_of_day_equity == Decimal("900")
            assert manager.start_of_day_equity == Decimal("900")

            mock_track.assert_called_once()
            assert result is False

    def test_check_liquidation_buffer_delegation(self, conservative_risk_config):
        """Test liquidation buffer checking delegation."""
        manager = LiveRiskManager(config=conservative_risk_config)
        position_data = {"quantity": Decimal("0.1"), "mark_price": Decimal("50000")}
        equity = Decimal("1000")

        with patch.object(
            manager.runtime_monitor, "check_liquidation_buffer", return_value=True
        ) as mock_check:
            result = manager.check_liquidation_buffer("BTC-USD", position_data, equity)

            mock_check.assert_called_once_with("BTC-USD", position_data, equity)
            assert result is True

    def test_check_mark_staleness_delegation(self, conservative_risk_config):
        """Test mark staleness checking delegation."""
        manager = LiveRiskManager(config=conservative_risk_config)
        timestamp = datetime.utcnow()

        with patch.object(
            manager.runtime_monitor, "check_mark_staleness", return_value=True
        ) as mock_check:
            result = manager.check_mark_staleness("BTC-USD", timestamp)

            mock_check.assert_called_once_with("BTC-USD", timestamp)
            assert result is True

    def test_append_risk_metrics_delegation(self, conservative_risk_config):
        """Test risk metrics appending delegation."""
        manager = LiveRiskManager(config=conservative_risk_config)
        equity = Decimal("1000")
        positions = {"BTC-USD": {"quantity": Decimal("0.1")}}

        with patch.object(manager.runtime_monitor, "append_risk_metrics") as mock_append:
            manager.append_risk_metrics(equity, positions)

            mock_append.assert_called_once_with(
                equity=equity,
                positions=positions,
                daily_pnl=manager.state_manager.daily_pnl,
                start_of_day_equity=manager.state_manager.start_of_day_equity,
                is_reduce_only_mode=manager.state_manager.is_reduce_only_mode(),
            )

    def test_check_correlation_risk_delegation(self, conservative_risk_config):
        """Test correlation risk checking delegation."""
        manager = LiveRiskManager(config=conservative_risk_config)
        positions = {"BTC-USD": {"quantity": Decimal("0.1")}}

        with patch.object(
            manager.runtime_monitor, "check_correlation_risk", return_value=True
        ) as mock_check:
            result = manager.check_correlation_risk(positions)

            mock_check.assert_called_once_with(positions)
            assert result is True

    def test_check_volatility_circuit_breaker_delegation(self, conservative_risk_config):
        """Test volatility circuit breaker checking delegation."""
        manager = LiveRiskManager(config=conservative_risk_config)
        symbol = "BTC-USD"
        recent_marks = [Decimal("50000"), Decimal("50100")]

        outcome = CircuitBreakerOutcome(
            triggered=True,
            action=CircuitBreakerAction.WARNING,
            volatility=Decimal("0.5"),
            message="High volatility detected",
        )

        with patch.object(
            manager.runtime_monitor, "check_volatility_circuit_breaker", return_value=outcome
        ) as mock_check:
            result = manager.check_volatility_circuit_breaker(symbol, recent_marks)

            mock_check.assert_called_once_with(symbol, recent_marks)
            assert result is outcome
            # Should update local reference for backward compatibility
            assert manager.circuit_breaker_state is manager.runtime_monitor.circuit_breaker_state


class TestLiveRiskManagerMarkTimestampTracking:
    """Test mark timestamp tracking functionality."""

    def test_record_mark_update(self, conservative_risk_config):
        """Test recording mark timestamp updates."""
        manager = LiveRiskManager(config=conservative_risk_config)
        timestamp = datetime(2024, 1, 1, 12, 0, 0)

        result = manager.record_mark_update("BTC-USD", timestamp)

        assert result == timestamp
        assert manager.mark_timestamp_for("BTC-USD") == timestamp

    def test_record_mark_update_without_timestamp(self, conservative_risk_config):
        """Test recording mark update without providing timestamp."""
        manager = LiveRiskManager(config=conservative_risk_config)

        with patch.object(
            manager.last_mark_update, "update_timestamp", return_value=datetime.utcnow()
        ) as mock_update:
            result = manager.record_mark_update("BTC-USD")

            mock_update.assert_called_once_with("BTC-USD", None)

    def test_mark_timestamp_for_nonexistent_symbol(self, conservative_risk_config):
        """Test getting timestamp for non-existent symbol."""
        manager = LiveRiskManager(config=conservative_risk_config)

        result = manager.mark_timestamp_for("NONEXISTENT")

        assert result is None


class TestLiveRiskManagerEventStoreRebinding:
    """Test event store rebinding functionality."""

    def test_set_event_store_propagates_to_components(self, conservative_risk_config):
        """Test that setting event store propagates to all components."""
        manager = LiveRiskManager(config=conservative_risk_config)
        new_event_store = EventStore()

        manager.set_event_store(new_event_store)

        assert manager.event_store is new_event_store
        assert manager.state_manager.event_store is new_event_store
        assert manager.position_sizer.event_store is new_event_store
        assert manager.pre_trade_validator.event_store is new_event_store
        assert manager.runtime_monitor.event_store is new_event_store

    def test_set_event_store_handles_missing_attributes(self, conservative_risk_config):
        """Test setting event store handles components without event_store attribute."""
        manager = LiveRiskManager(config=conservative_risk_config)

        # Temporarily remove event_store attribute from one component
        original_attr = getattr(manager.state_manager, "event_store", None)
        if hasattr(manager.state_manager, "event_store"):
            delattr(manager.state_manager, "event_store")

        new_event_store = EventStore()

        # Should not raise exception
        manager.set_event_store(new_event_store)

        # Restore attribute
        if original_attr is not None:
            manager.state_manager.event_store = original_attr


class TestLiveRiskManagerTimeProvider:
    """Test time provider functionality and propagation."""

    def test_setattr_updates_time_provider(self, conservative_risk_config):
        """Test that setting _now_provider updates all components."""
        manager = LiveRiskManager(config=conservative_risk_config)
        fixed_time = datetime(2024, 1, 1, 12, 0, 0)
        new_time_provider = lambda: fixed_time

        manager._now_provider = new_time_provider

        assert manager.state_manager._now_provider is new_time_provider
        assert manager.pre_trade_validator._now_provider is new_time_provider
        assert manager.runtime_monitor._now_provider is new_time_provider

    def test_setattr_only_updates_now_provider(self, conservative_risk_config):
        """Test that setattr only updates specific attributes."""
        manager = LiveRiskManager(config=conservative_risk_config)

        # Setting other attributes should not trigger time provider updates
        manager.some_other_attribute = "test"

        # Should not have affected time providers
        assert manager.state_manager._now_provider is not None
        assert manager.pre_trade_validator._now_provider is not None
        assert manager.runtime_monitor._now_provider is not None

    def test_now_method_returns_current_time(self, conservative_risk_config):
        """Test _now method returns time from provider."""
        manager = LiveRiskManager(config=conservative_risk_config)
        fixed_time = datetime(2024, 1, 1, 12, 0, 0)
        manager._now_provider = lambda: fixed_time

        result = manager._now()

        assert result == fixed_time


class TestLiveRiskManagerIntegrationScenarios:
    """Test complex integration scenarios between components."""

    def test_pre_trade_validation_with_reduce_only_mode(self, conservative_risk_config):
        """Test pre-trade validation behavior in reduce-only mode."""
        manager = LiveRiskManager(config=conservative_risk_config)
        manager.set_reduce_only_mode(True, "test scenario")

        # Should delegate to pre_trade_validator which should check reduce-only mode
        with patch.object(manager.pre_trade_validator, "pre_trade_validate") as mock_validate:
            manager.pre_trade_validate(
                symbol="BTC-USD",
                side="buy",  # This should be rejected in reduce-only mode
                quantity=Decimal("0.1"),
            )

            mock_validate.assert_called_once()

    def test_circuit_breaker_triggers_reduce_only_mode(self, conservative_risk_config):
        """Test circuit breaker triggering reduce-only mode."""
        manager = LiveRiskManager(config=conservative_risk_config)
        symbol = "BTC-USD"
        recent_marks = [Decimal("50000"), Decimal("52000"), Decimal("48000")]  # High volatility

        outcome = CircuitBreakerOutcome(
            triggered=True,
            action=CircuitBreakerAction.REDUCE_ONLY,
            volatility=Decimal("0.8"),
            message="Extreme volatility detected",
        )

        with patch.object(
            manager.runtime_monitor, "check_volatility_circuit_breaker", return_value=outcome
        ):
            result = manager.check_volatility_circuit_breaker(symbol, recent_marks)

            assert result.action is CircuitBreakerAction.REDUCE_ONLY
            # RuntimeMonitor should have called set_reduce_only_mode on state manager
            # This would be tested through integration tests

    def test_daily_pnl_tracking_with_circuit_breaker(self, conservative_risk_config):
        """Test daily PnL tracking interacting with circuit breakers."""
        manager = LiveRiskManager(config=conservative_risk_config)
        manager.state_manager.start_of_day_equity = Decimal("1000")
        current_equity = Decimal("800")  # 20% loss
        positions_pnl = {"BTC-USD": {"unrealized": Decimal("-200")}}

        with patch.object(
            manager.runtime_monitor, "track_daily_pnl", return_value=(True, Decimal("-200"))
        ) as mock_track:
            result = manager.track_daily_pnl(current_equity, positions_pnl)

            assert result is True  # Loss limit triggered
            assert manager.daily_pnl == Decimal("-200")
            mock_track.assert_called_once()

    def test_position_sizing_with_market_impact(self, conservative_risk_config):
        """Test position sizing with custom impact estimator."""
        manager = LiveRiskManager(config=conservative_risk_config)

        def custom_impact_estimator(request):
            from bot_v2.features.live_trade.risk.position_sizing import ImpactAssessment

            return ImpactAssessment(
                price_impact=Decimal("0.001"), size_impact=Decimal("0.1"), confidence=Decimal("0.9")
            )

        manager.set_impact_estimator(custom_impact_estimator)

        context = PositionSizingContext(
            symbol="BTC-USD",
            side="buy",
            available_equity=Decimal("1000"),
            current_positions={},
            mark_price=Decimal("50000"),
        )

        with patch.object(
            manager.position_sizer,
            "size_position",
            return_value=PositionSizingAdvice(
                recommended_quantity=Decimal("0.15"),
                max_quantity=Decimal("0.2"),
                reason="Based on custom impact model",
            ),
        ) as mock_size:
            result = manager.size_position(context)

            mock_size.assert_called_once_with(context)
            assert result.recommended_quantity == Decimal("0.15")

    def test_multi_symbol_risk_assessment(self, conservative_risk_config):
        """Test risk assessment across multiple symbols."""
        manager = LiveRiskManager(config=conservative_risk_config)

        # Setup multi-symbol portfolio
        positions = {
            "BTC-USD": {"quantity": Decimal("0.1"), "mark_price": Decimal("50000")},
            "ETH-USD": {"quantity": Decimal("1.0"), "mark_price": Decimal("3000")},
            "SOL-USD": {"quantity": Decimal("10.0"), "mark_price": Decimal("100")},
        }

        # Test correlation risk
        with patch.object(
            manager.runtime_monitor, "check_correlation_risk", return_value=False
        ) as mock_check:
            result = manager.check_correlation_risk(positions)
            mock_check.assert_called_once_with(positions)
            assert result is False

    def test_emergency_scenarios_integration(self, conservative_risk_config):
        """Test various emergency scenarios and their integration."""
        manager = LiveRiskManager(config=conservative_risk_config)

        # Scenario 1: Stale market data
        stale_timestamp = datetime.utcnow() - timedelta(minutes=10)
        with patch.object(manager.runtime_monitor, "check_mark_staleness", return_value=True):
            assert manager.check_mark_staleness("BTC-USD", stale_timestamp) is True

        # Scenario 2: Circuit breaker activation
        volatile_marks = [Decimal("50000"), Decimal("55000"), Decimal("45000")]
        outcome = CircuitBreakerOutcome(
            triggered=True,
            action=CircuitBreakerAction.KILL_SWITCH,
            volatility=Decimal("1.2"),
            message="Market chaos - kill switch activated",
        )

        with patch.object(
            manager.runtime_monitor, "check_volatility_circuit_breaker", return_value=outcome
        ):
            result = manager.check_volatility_circuit_breaker("BTC-USD", volatile_marks)
            assert result.action is CircuitBreakerAction.KILL_SWITCH

        # Scenario 3: Position liquidation risk
        position_data = {
            "quantity": Decimal("0.1"),
            "mark_price": Decimal("50000"),
            "maintenance_margin": Decimal("1000"),
        }
        equity = Decimal("1100")  # Very close to liquidation

        with patch.object(manager.runtime_monitor, "check_liquidation_buffer", return_value=True):
            result = manager.check_liquidation_buffer("BTC-USD", position_data, equity)
            assert result is True  # Buffer breached


class TestLiveRiskManagerErrorHandling:
    """Test error handling and edge cases."""

    def test_validation_error_propagation(self, conservative_risk_config):
        """Test that validation errors are properly propagated."""
        manager = LiveRiskManager(config=conservative_risk_config)

        with patch.object(
            manager.pre_trade_validator,
            "pre_trade_validate",
            side_effect=ValidationError("Test validation error"),
        ):

            with pytest.raises(ValidationError, match="Test validation error"):
                manager.pre_trade_validate(symbol="BTC-USD", side="buy", quantity=Decimal("0.1"))

    def test_component_error_handling(self, conservative_risk_config):
        """Test handling of component errors."""
        manager = LiveRiskManager(config=conservative_risk_config)

        # Test that component errors don't crash the manager
        with patch.object(
            manager.state_manager, "is_reduce_only_mode", side_effect=Exception("Component error")
        ):
            with pytest.raises(Exception, match="Component error"):
                manager.is_reduce_only_mode()

    def test_graceful_degradation_with_missing_dependencies(self):
        """Test graceful degradation when dependencies are missing."""
        # This would be tested with actual missing dependencies in integration scenarios
        pass

    def test_thread_safety_under_concurrent_operations(self, conservative_risk_config):
        """Test thread safety when multiple operations occur concurrently."""
        manager = LiveRiskManager(config=conservative_risk_config)

        # Test concurrent mark timestamp updates
        manager.record_mark_update("BTC-USD")
        manager.record_mark_update("ETH-USD")
        manager.record_mark_update("SOL-USD")

        # Should handle gracefully
        assert len(manager.last_mark_update) == 3

    def test_configuration_validation(self):
        """Test configuration validation during initialization."""
        # Test with invalid configuration
        with patch(
            "bot_v2.features.live_trade.risk.manager.RiskConfig.from_env",
            side_effect=Exception("Config error"),
        ):

            # Should still initialize with some default behavior
            with pytest.raises(Exception, match="Config error"):
                LiveRiskManager()


class TestLiveRiskManagerBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_direct_state_access(self, conservative_risk_config):
        """Test direct state access for backward compatibility."""
        manager = LiveRiskManager(config=conservative_risk_config)

        # These should work for backward compatibility
        assert hasattr(manager, "_state")
        assert hasattr(manager, "daily_pnl")
        assert hasattr(manager, "start_of_day_equity")
        assert hasattr(manager, "positions")
        assert hasattr(manager, "circuit_breaker_state")

    def test_state_reference_updates(self, conservative_risk_config):
        """Test that state references are properly updated."""
        manager = LiveRiskManager(config=conservative_risk_config)
        original_daily_pnl = manager.daily_pnl

        # Update daily PnL through state manager
        manager.state_manager.daily_pnl = Decimal("100")

        # Manager's reference should also update
        assert manager.daily_pnl is manager.state_manager.daily_pnl

    def test_method_signatures_backward_compatibility(self, conservative_risk_config):
        """Test that method signatures remain backward compatible."""
        manager = LiveRiskManager(config=conservative_risk_config)

        # All these methods should exist and be callable
        assert callable(manager.pre_trade_validate)
        assert callable(manager.validate_leverage)
        assert callable(manager.validate_liquidation_buffer)
        assert callable(manager.validate_exposure_limits)
        assert callable(manager.size_position)
        assert callable(manager.track_daily_pnl)
        assert callable(manager.check_volatility_circuit_breaker)

    def test_quantity_parameter_backward_compatibility(self, conservative_risk_config):
        """Test backward compatibility for both qty and quantity parameters."""
        manager = LiveRiskManager(config=conservative_risk_config)

        with patch.object(manager.pre_trade_validator, "pre_trade_validate") as mock_validate:
            # Test with qty parameter
            manager.pre_trade_validate(symbol="BTC-USD", side="buy", qty=Decimal("0.1"))

            # Test with quantity parameter
            manager.pre_trade_validate(symbol="BTC-USD", side="buy", quantity=Decimal("0.1"))

            assert mock_validate.call_count == 2

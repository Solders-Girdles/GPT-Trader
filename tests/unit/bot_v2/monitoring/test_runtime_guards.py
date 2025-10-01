"""Tests for RuntimeGuards - circuit breaker and alerting system.

This module tests the RuntimeGuardManager and various guard implementations
that monitor critical runtime conditions and trigger alerts when thresholds
are breached.

Critical behaviors tested:
- Guard initialization and configuration
- Threshold-based alert triggering
- Cooldown periods to prevent alert spam
- Daily loss guard with automatic resets
- Stale mark detection for market data freshness
- Error rate monitoring and thresholding
- Position stuck detection
- Drawdown monitoring
- Alert routing and handler execution
- Auto-shutdown on critical breaches
- Guard status management and resets

System Protection Context:
    Runtime guards are the last line of defense against catastrophic
    trading failures. They monitor the system in real-time and trigger
    protective actions when conditions deteriorate. Failures here can result in:

    - Runaway losses exceeding daily limits
    - Trading on stale/invalid market data
    - Cascading errors overwhelming the system
    - Positions stuck without management
    - Excessive drawdowns without intervention

    These guards must be reliable, responsive, and fail-safe.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, call

import pytest

from bot_v2.monitoring.alerts import AlertSeverity
from bot_v2.monitoring.runtime_guards import (
    Alert,
    DailyLossGuard,
    DrawdownGuard,
    ErrorRateGuard,
    GuardConfig,
    GuardStatus,
    PositionStuckGuard,
    RuntimeGuard,
    RuntimeGuardManager,
    StaleMarkGuard,
    create_default_guards,
)

# Fixtures imported from tests/fixtures/monitoring.py via conftest.py:
# - basic_guard_config: Standard GuardConfig for testing
# - guard: Basic RuntimeGuard instance
# - guard_manager: Empty RuntimeGuardManager for coordination
# - critical_guard_config: GuardConfig with auto-shutdown enabled
# - disabled_guard_config: GuardConfig for testing disabled guards


class TestGuardConfig:
    """Test GuardConfig dataclass."""

    def test_creates_config_with_defaults(self) -> None:
        """GuardConfig initializes with sensible defaults.

        Default values allow quick guard creation without explicit config.
        """
        config = GuardConfig(name="test")

        assert config.name == "test"
        assert config.enabled is True
        assert config.threshold == 0.0
        assert config.window_seconds == 60
        assert config.severity == AlertSeverity.WARNING
        assert config.auto_shutdown is False
        assert config.cooldown_seconds == 300

    def test_creates_config_with_custom_values(self) -> None:
        """GuardConfig accepts custom configuration values."""
        config = GuardConfig(
            name="critical_guard",
            enabled=True,
            threshold=50.0,
            window_seconds=120,
            severity=AlertSeverity.CRITICAL,
            auto_shutdown=True,
            cooldown_seconds=600,
        )

        assert config.name == "critical_guard"
        assert config.threshold == 50.0
        assert config.severity == AlertSeverity.CRITICAL
        assert config.auto_shutdown is True


class TestRuntimeGuardInitialization:
    """Test RuntimeGuard base class initialization."""

    def test_initializes_as_healthy_when_enabled(self, basic_guard_config: GuardConfig) -> None:
        """Guard starts in HEALTHY status when enabled.

        Default starting state for active monitoring.
        """
        guard = RuntimeGuard(basic_guard_config)

        assert guard.status == GuardStatus.HEALTHY

    def test_initializes_as_disabled_when_not_enabled(self) -> None:
        """Guard starts in DISABLED status when not enabled.

        Allows guards to be configured but not active.
        """
        config = GuardConfig(name="test", enabled=False)
        guard = RuntimeGuard(config)

        assert guard.status == GuardStatus.DISABLED

    def test_initializes_with_zero_breach_count(self, guard: RuntimeGuard) -> None:
        """Guard starts with zero breach count."""
        assert guard.breach_count == 0

    def test_initializes_with_no_alerts(self, guard: RuntimeGuard) -> None:
        """Guard starts with empty alert list."""
        assert guard.alerts == []


class TestGuardCheck:
    """Test guard check method and alert triggering."""

    def test_returns_none_when_disabled(self) -> None:
        """Disabled guards never trigger alerts.

        Allows guards to be turned off without removal.
        """
        config = GuardConfig(name="test", enabled=False)
        guard = RuntimeGuard(config)

        alert = guard.check({"value": 150.0, "threshold": 100.0})

        assert alert is None

    def test_returns_none_during_cooldown(self, guard: RuntimeGuard) -> None:
        """Guard respects cooldown period after alert.

        Prevents alert spam for persistent conditions.
        """
        # Trigger initial alert
        guard._evaluate = Mock(return_value=(True, "Threshold breached"))
        guard.check({"value": 150.0})

        # Attempt to trigger again immediately
        alert = guard.check({"value": 150.0})

        assert alert is None  # Still in cooldown

    def test_triggers_alert_when_condition_breached(self, guard: RuntimeGuard) -> None:
        """Guard triggers alert when evaluation returns breach.

        Core functionality - alert on threshold breach.
        """
        guard._evaluate = Mock(return_value=(True, "Test breach"))

        alert = guard.check({"value": 150.0})

        assert alert is not None
        assert alert.guard_name == "test_guard"
        assert alert.message == "Test breach"

    def test_sets_status_to_breached_on_alert(self, guard: RuntimeGuard) -> None:
        """Guard status changes to BREACHED when alert triggers.

        Status tracking for monitoring and dashboards.
        """
        guard._evaluate = Mock(return_value=(True, "Test breach"))

        guard.check({"value": 150.0})

        assert guard.status == GuardStatus.BREACHED

    def test_increments_breach_count_on_alert(self, guard: RuntimeGuard) -> None:
        """Breach count increments on each alert.

        Tracks frequency of breaches for analysis.
        """
        guard._evaluate = Mock(return_value=(True, "Test breach"))
        guard.config.cooldown_seconds = 0  # Disable cooldown for test

        guard.check({"value": 150.0})
        guard.check({"value": 150.0})

        assert guard.breach_count == 2

    def test_stores_alert_in_history(self, guard: RuntimeGuard) -> None:
        """Alert is stored in guard's alert history.

        Provides audit trail of guard activity.
        """
        guard._evaluate = Mock(return_value=(True, "Test breach"))

        guard.check({"value": 150.0})

        assert len(guard.alerts) == 1
        assert guard.alerts[0].message == "Test breach"

    def test_downgrades_from_breached_to_warning(self, guard: RuntimeGuard) -> None:
        """Guard status downgrades from BREACHED to WARNING when condition clears.

        Progressive status changes reflect improving conditions.
        """
        # Trigger breach
        guard._evaluate = Mock(return_value=(True, "Breached"))
        guard.check({"value": 150.0})
        assert guard.status == GuardStatus.BREACHED

        # Condition clears
        guard._evaluate = Mock(return_value=(False, ""))
        guard.check({"value": 50.0})

        assert guard.status == GuardStatus.WARNING

    def test_alert_includes_context(self, guard: RuntimeGuard) -> None:
        """Alert captures context data for debugging.

        Context provides details about what triggered the alert.
        """
        guard._evaluate = Mock(return_value=(True, "Test breach"))

        context = {"value": 150.0, "symbol": "BTC-PERP"}
        alert = guard.check(context)

        assert alert.context == context


class TestGuardReset:
    """Test guard reset functionality."""

    def test_reset_clears_breach_count(self, guard: RuntimeGuard) -> None:
        """Reset clears breach count back to zero.

        Allows fresh start after issue resolution.
        """
        guard._evaluate = Mock(return_value=(True, "Breached"))
        guard.config.cooldown_seconds = 0
        guard.check({})
        guard.check({})

        guard.reset()

        assert guard.breach_count == 0

    def test_reset_clears_last_alert(self, guard: RuntimeGuard) -> None:
        """Reset clears last alert timestamp.

        Removes cooldown restriction after reset.
        """
        guard._evaluate = Mock(return_value=(True, "Breached"))
        guard.check({})

        guard.reset()

        assert guard.last_alert is None

    def test_reset_restores_healthy_status(self, guard: RuntimeGuard) -> None:
        """Reset restores HEALTHY status for enabled guards.

        Returns guard to normal monitoring state.
        """
        guard._evaluate = Mock(return_value=(True, "Breached"))
        guard.check({})

        guard.reset()

        assert guard.status == GuardStatus.HEALTHY


class TestDailyLossGuard:
    """Test DailyLossGuard implementation."""

    def test_tracks_cumulative_daily_pnl(self) -> None:
        """Tracks cumulative P&L across multiple checks.

        Accumulates losses/gains throughout the day.
        """
        config = GuardConfig(name="daily_loss", threshold=100.0)
        guard = DailyLossGuard(config)

        guard.check({"pnl": -30.0})
        guard.check({"pnl": -20.0})

        assert guard.daily_pnl == Decimal("-50.0")

    def test_triggers_alert_when_loss_exceeds_threshold(self) -> None:
        """Triggers alert when daily loss exceeds threshold.

        Critical: Must halt trading when daily loss limit breached.
        """
        config = GuardConfig(name="daily_loss", threshold=100.0)
        guard = DailyLossGuard(config)

        guard.check({"pnl": -80.0})
        alert = guard.check({"pnl": -30.0})  # Total: -110

        assert alert is not None
        assert "Daily loss limit breached" in alert.message

    def test_does_not_trigger_on_gains(self) -> None:
        """Does not trigger on profitable days.

        Only losses trigger the guard.
        """
        config = GuardConfig(name="daily_loss", threshold=100.0)
        guard = DailyLossGuard(config)

        alert = guard.check({"pnl": 50.0})

        assert alert is None

    def test_resets_daily_counter_on_new_day(self) -> None:
        """Resets P&L tracking at start of new day.

        Daily limit resets daily, not cumulative.
        """
        config = GuardConfig(name="daily_loss", threshold=100.0)
        guard = DailyLossGuard(config)

        # Set to yesterday
        guard.last_reset = datetime.now().date() - timedelta(days=1)
        guard.daily_pnl = Decimal("-150.0")

        # Check today
        alert = guard.check({"pnl": -10.0})

        # Should reset and not trigger
        assert alert is None
        assert guard.daily_pnl == Decimal("-10.0")


class TestStaleMarkGuard:
    """Test StaleMarkGuard implementation."""

    def test_does_not_trigger_on_fresh_marks(self) -> None:
        """Does not trigger when mark data is fresh.

        Recent data should pass the staleness check.
        """
        config = GuardConfig(name="stale_marks", threshold=60.0)
        guard = StaleMarkGuard(config)

        fresh_time = datetime.now() - timedelta(seconds=10)
        alert = guard.check({"symbol": "BTC-PERP", "mark_timestamp": fresh_time})

        assert alert is None

    def test_triggers_on_stale_marks(self) -> None:
        """Triggers alert when mark data exceeds age threshold.

        Critical: Trading on stale data can lead to bad fills.
        """
        config = GuardConfig(name="stale_marks", threshold=60.0)
        guard = StaleMarkGuard(config)

        stale_time = datetime.now() - timedelta(seconds=120)
        alert = guard.check({"symbol": "BTC-PERP", "mark_timestamp": stale_time})

        assert alert is not None
        assert "Stale marks detected" in alert.message
        assert "BTC-PERP" in alert.message

    def test_tracks_marks_per_symbol(self) -> None:
        """Tracks mark timestamps separately for each symbol.

        Different symbols may have different update frequencies.
        """
        config = GuardConfig(name="stale_marks", threshold=60.0)
        guard = StaleMarkGuard(config)

        time_btc = datetime.now() - timedelta(seconds=10)
        time_eth = datetime.now() - timedelta(seconds=20)

        guard.check({"symbol": "BTC-PERP", "mark_timestamp": time_btc})
        guard.check({"symbol": "ETH-PERP", "mark_timestamp": time_eth})

        assert "BTC-PERP" in guard.last_marks
        assert "ETH-PERP" in guard.last_marks

    def test_handles_string_timestamps(self) -> None:
        """Handles ISO format string timestamps.

        Flexible input handling for different data sources.
        """
        config = GuardConfig(name="stale_marks", threshold=60.0)
        guard = StaleMarkGuard(config)

        time_str = (datetime.now() - timedelta(seconds=10)).isoformat()
        alert = guard.check({"symbol": "BTC-PERP", "mark_timestamp": time_str})

        assert alert is None  # Fresh timestamp

    def test_handles_numeric_timestamps(self) -> None:
        """Handles Unix timestamp integers/floats.

        Support for different timestamp formats.
        """
        config = GuardConfig(name="stale_marks", threshold=60.0)
        guard = StaleMarkGuard(config)

        timestamp = datetime.now().timestamp()
        alert = guard.check({"symbol": "BTC-PERP", "mark_timestamp": timestamp})

        assert alert is None  # Fresh timestamp


class TestErrorRateGuard:
    """Test ErrorRateGuard implementation."""

    def test_counts_errors_in_sliding_window(self) -> None:
        """Counts errors within sliding time window.

        Only recent errors count toward threshold.
        """
        config = GuardConfig(name="error_rate", threshold=5.0, window_seconds=60)
        guard = ErrorRateGuard(config)

        for _ in range(3):
            guard.check({"error": True})

        assert len(guard.error_times) == 3

    def test_triggers_when_error_rate_exceeds_threshold(self) -> None:
        """Triggers alert when error count exceeds threshold.

        High error rates indicate system instability.
        """
        config = GuardConfig(name="error_rate", threshold=5.0, window_seconds=60)
        guard = ErrorRateGuard(config)

        for _ in range(5):
            guard.check({"error": True})

        alert = guard.check({"error": True})  # 6th error

        assert alert is not None
        assert "High error rate" in alert.message

    def test_expires_old_errors_outside_window(self) -> None:
        """Removes errors older than window from count.

        Sliding window - old errors don't count.
        """
        config = GuardConfig(name="error_rate", threshold=10.0, window_seconds=1)  # 1 second window
        guard = ErrorRateGuard(config)

        # Add old error
        old_time = datetime.now() - timedelta(seconds=5)
        guard.error_times.append(old_time)

        # Check with new error
        guard.check({"error": True})

        # Old error should be removed
        assert all(t > old_time for t in guard.error_times)


class TestPositionStuckGuard:
    """Test PositionStuckGuard implementation."""

    def test_tracks_position_open_duration(self) -> None:
        """Tracks how long positions have been open.

        Monitors for positions not being actively managed.
        """
        config = GuardConfig(name="position_stuck", threshold=1800.0)
        guard = PositionStuckGuard(config)

        positions = {"BTC-PERP": {"quantity": 1.5}}
        guard.check({"positions": positions})

        assert "BTC-PERP" in guard.position_times

    def test_triggers_on_long_held_positions(self) -> None:
        """Triggers when position exceeds maximum hold time.

        Alerts on potentially abandoned positions.
        """
        config = GuardConfig(name="position_stuck", threshold=1800.0)
        guard = PositionStuckGuard(config)

        # Simulate old position
        old_time = datetime.now() - timedelta(seconds=2000)
        guard.position_times["BTC-PERP"] = old_time

        alert = guard.check({"positions": {"BTC-PERP": {"quantity": 1.5}}})

        assert alert is not None
        assert "Stuck positions detected" in alert.message

    def test_clears_closed_positions(self) -> None:
        """Removes closed positions from tracking.

        Zero quantity = position closed.
        """
        config = GuardConfig(name="position_stuck", threshold=1800.0)
        guard = PositionStuckGuard(config)

        # Open position
        guard.check({"positions": {"BTC-PERP": {"quantity": 1.5}}})
        assert "BTC-PERP" in guard.position_times

        # Close position
        guard.check({"positions": {"BTC-PERP": {"quantity": 0.0}}})
        assert "BTC-PERP" not in guard.position_times

    def test_handles_various_quantity_field_names(self) -> None:
        """Handles different quantity field names (quantity, size, contracts).

        Different brokers use different field names.
        """
        config = GuardConfig(name="position_stuck", threshold=1800.0)
        guard = PositionStuckGuard(config)

        guard.check({"positions": {"BTC": {"quantity": 1.0}}})
        guard.check({"positions": {"ETH": {"size": 1.0}}})
        guard.check({"positions": {"SOL": {"contracts": 1.0}}})

        assert "BTC" in guard.position_times
        assert "ETH" in guard.position_times
        assert "SOL" in guard.position_times


class TestDrawdownGuard:
    """Test DrawdownGuard implementation."""

    def test_tracks_peak_equity(self) -> None:
        """Tracks peak equity as high-water mark.

        Drawdown calculated from peak equity.
        """
        config = GuardConfig(name="max_drawdown", threshold=10.0)
        guard = DrawdownGuard(config)

        guard.check({"equity": 10000.0})
        guard.check({"equity": 12000.0})

        assert guard.peak_equity == Decimal("12000.0")

    def test_calculates_drawdown_percentage(self) -> None:
        """Calculates drawdown as percentage from peak.

        Standard drawdown metric for risk monitoring.
        """
        config = GuardConfig(name="max_drawdown", threshold=10.0)
        guard = DrawdownGuard(config)

        guard.check({"equity": 10000.0})
        guard.check({"equity": 9000.0})  # 10% down

        assert guard.current_drawdown == Decimal("10.0")

    def test_triggers_when_drawdown_exceeds_threshold(self) -> None:
        """Triggers alert when drawdown exceeds maximum.

        Critical: Must halt trading during excessive drawdowns.
        """
        config = GuardConfig(name="max_drawdown", threshold=10.0)
        guard = DrawdownGuard(config)

        guard.check({"equity": 10000.0})
        alert = guard.check({"equity": 8500.0})  # 15% down

        assert alert is not None
        assert "Maximum drawdown breached" in alert.message

    def test_does_not_trigger_below_threshold(self) -> None:
        """Does not trigger when drawdown within acceptable range.

        Normal volatility should not trigger alerts.
        """
        config = GuardConfig(name="max_drawdown", threshold=10.0)
        guard = DrawdownGuard(config)

        guard.check({"equity": 10000.0})
        alert = guard.check({"equity": 9100.0})  # 9% down

        assert alert is None


class TestRuntimeGuardManager:
    """Test RuntimeGuardManager orchestration."""

    def test_adds_guards_to_manager(self, guard_manager: RuntimeGuardManager) -> None:
        """Adds guards to manager for coordinated monitoring.

        Manager orchestrates multiple guards.
        """
        config = GuardConfig(name="test_guard")
        guard = RuntimeGuard(config)

        guard_manager.add_guard(guard)

        assert "test_guard" in guard_manager.guards

    def test_checks_all_guards_with_context(self, guard_manager: RuntimeGuardManager) -> None:
        """Checks all registered guards with provided context.

        Single call checks all guards simultaneously.
        """
        guard1 = RuntimeGuard(GuardConfig(name="guard1"))
        guard2 = RuntimeGuard(GuardConfig(name="guard2"))

        guard_manager.add_guard(guard1)
        guard_manager.add_guard(guard2)

        guard1.check = Mock(return_value=None)
        guard2.check = Mock(return_value=None)

        context = {"value": 100.0}
        guard_manager.check_all(context)

        guard1.check.assert_called_once_with(context)
        guard2.check.assert_called_once_with(context)

    def test_collects_alerts_from_all_guards(self, guard_manager: RuntimeGuardManager) -> None:
        """Collects and returns alerts from all triggered guards.

        Returns list of all active alerts for processing.
        """
        guard1 = RuntimeGuard(GuardConfig(name="guard1"))
        guard2 = RuntimeGuard(GuardConfig(name="guard2"))

        alert1 = Alert(
            timestamp=datetime.now(),
            guard_name="guard1",
            severity=AlertSeverity.WARNING,
            message="Test alert 1",
        )
        alert2 = Alert(
            timestamp=datetime.now(),
            guard_name="guard2",
            severity=AlertSeverity.ERROR,
            message="Test alert 2",
        )

        guard1.check = Mock(return_value=alert1)
        guard2.check = Mock(return_value=alert2)

        guard_manager.add_guard(guard1)
        guard_manager.add_guard(guard2)

        alerts = guard_manager.check_all({})

        assert len(alerts) == 2
        assert alert1 in alerts
        assert alert2 in alerts

    def test_calls_alert_handlers(self, guard_manager: RuntimeGuardManager) -> None:
        """Calls registered alert handlers for each alert.

        Allows custom alert routing (Slack, email, etc.).
        """
        handler = Mock()
        guard_manager.add_alert_handler(handler)

        guard = RuntimeGuard(GuardConfig(name="test"))
        alert = Alert(
            timestamp=datetime.now(),
            guard_name="test",
            severity=AlertSeverity.WARNING,
            message="Test",
        )
        guard.check = Mock(return_value=alert)
        guard_manager.add_guard(guard)

        guard_manager.check_all({})

        handler.assert_called_once_with(alert)

    def test_triggers_shutdown_callback_on_auto_shutdown(
        self, guard_manager: RuntimeGuardManager
    ) -> None:
        """Triggers shutdown callback for guards with auto_shutdown enabled.

        Critical: Auto-shutdown guards must halt trading immediately.
        """
        shutdown_callback = Mock()
        guard_manager.set_shutdown_callback(shutdown_callback)

        config = GuardConfig(name="critical", auto_shutdown=True)
        guard = RuntimeGuard(config)
        alert = Alert(
            timestamp=datetime.now(),
            guard_name="critical",
            severity=AlertSeverity.CRITICAL,
            message="Critical failure",
        )
        guard.check = Mock(return_value=alert)
        guard_manager.add_guard(guard)

        guard_manager.check_all({})

        shutdown_callback.assert_called_once()

    def test_handles_handler_errors_gracefully(self, guard_manager: RuntimeGuardManager) -> None:
        """Handles alert handler failures without crashing.

        Handler failures should not prevent other handlers from running.
        """
        failing_handler = Mock(side_effect=Exception("Handler error"))
        success_handler = Mock()

        guard_manager.add_alert_handler(failing_handler)
        guard_manager.add_alert_handler(success_handler)

        guard = RuntimeGuard(GuardConfig(name="test"))
        alert = Alert(
            timestamp=datetime.now(),
            guard_name="test",
            severity=AlertSeverity.WARNING,
            message="Test",
        )
        guard.check = Mock(return_value=alert)
        guard_manager.add_guard(guard)

        # Should not raise
        guard_manager.check_all({})

        # Success handler should still be called
        success_handler.assert_called_once()

    def test_gets_status_of_all_guards(self, guard_manager: RuntimeGuardManager) -> None:
        """Retrieves status summary of all guards.

        Provides dashboard/monitoring view of guard states.
        """
        guard1 = RuntimeGuard(GuardConfig(name="guard1"))
        guard2 = RuntimeGuard(GuardConfig(name="guard2"))

        guard_manager.add_guard(guard1)
        guard_manager.add_guard(guard2)

        status = guard_manager.get_status()

        assert "guard1" in status
        assert "guard2" in status
        assert status["guard1"]["status"] == "healthy"
        assert status["guard2"]["enabled"] is True

    def test_resets_specific_guard(self, guard_manager: RuntimeGuardManager) -> None:
        """Resets a specific guard by name.

        Allows targeted reset of individual guards.
        """
        guard = RuntimeGuard(GuardConfig(name="test"))
        guard.breach_count = 5
        guard_manager.add_guard(guard)

        guard_manager.reset_guard("test")

        assert guard.breach_count == 0

    def test_resets_all_guards(self, guard_manager: RuntimeGuardManager) -> None:
        """Resets all registered guards simultaneously.

        Bulk reset for system-wide restart.
        """
        guard1 = RuntimeGuard(GuardConfig(name="guard1"))
        guard2 = RuntimeGuard(GuardConfig(name="guard2"))
        guard1.breach_count = 3
        guard2.breach_count = 5

        guard_manager.add_guard(guard1)
        guard_manager.add_guard(guard2)

        guard_manager.reset_all()

        assert guard1.breach_count == 0
        assert guard2.breach_count == 0


class TestCreateDefaultGuards:
    """Test default guard factory function."""

    def test_creates_manager_with_default_guards(self) -> None:
        """Creates manager with standard set of guards.

        Provides production-ready guard configuration.
        """
        config = {
            "risk_management": {
                "daily_loss_limit": 1000.0,
                "max_drawdown_pct": 15.0,
                "circuit_breakers": {
                    "stale_mark_seconds": 30.0,
                    "error_threshold": 10.0,
                },
            }
        }

        manager = create_default_guards(config)

        assert "daily_loss" in manager.guards
        assert "stale_marks" in manager.guards
        assert "error_rate" in manager.guards
        assert "position_stuck" in manager.guards
        assert "max_drawdown" in manager.guards

    def test_configures_guards_from_config_dict(self) -> None:
        """Configures guard thresholds from config dictionary.

        Allows customization of default guard parameters.
        """
        config = {
            "risk_management": {
                "daily_loss_limit": 500.0,
                "circuit_breakers": {"stale_mark_seconds": 60.0},
            }
        }

        manager = create_default_guards(config)

        daily_loss = manager.guards["daily_loss"]
        assert daily_loss.config.threshold == 500.0

    def test_handles_missing_config_gracefully(self) -> None:
        """Uses default values when config keys missing.

        Defensive: Partial config should not break initialization.
        """
        config = {}

        manager = create_default_guards(config)

        # Should create guards with defaults
        assert len(manager.guards) > 0


class TestAlertDataClass:
    """Test Alert dataclass."""

    def test_creates_alert_with_required_fields(self) -> None:
        """Alert initializes with required fields."""
        alert = Alert(
            timestamp=datetime.now(),
            guard_name="test",
            severity=AlertSeverity.WARNING,
            message="Test message",
        )

        assert alert.guard_name == "test"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Test message"

    def test_alert_includes_context(self) -> None:
        """Alert can include context dictionary."""
        context = {"symbol": "BTC-PERP", "value": 100.0}
        alert = Alert(
            timestamp=datetime.now(),
            guard_name="test",
            severity=AlertSeverity.WARNING,
            message="Test",
            context=context,
        )

        assert alert.context == context

    def test_to_dict_serializes_alert(self) -> None:
        """to_dict() serializes alert for storage/transmission.

        JSON-serializable representation of alert.
        """
        alert = Alert(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            guard_name="test",
            severity=AlertSeverity.ERROR,
            message="Test message",
            context={"key": "value"},
        )

        result = alert.to_dict()

        assert result["guard_name"] == "test"
        assert result["severity"] == "error"
        assert result["message"] == "Test message"
        assert result["context"] == {"key": "value"}
        assert "timestamp" in result

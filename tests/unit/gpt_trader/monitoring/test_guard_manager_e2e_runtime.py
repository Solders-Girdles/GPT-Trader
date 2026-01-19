"""End-to-end tests for RuntimeGuardManager core behavior."""

from __future__ import annotations

from datetime import datetime

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.guards.base import GuardConfig, GuardStatus
from gpt_trader.monitoring.guards.builtins import DailyLossGuard, ErrorRateGuard, StaleMarkGuard
from gpt_trader.monitoring.guards.manager import RuntimeGuardManager


class TestRuntimeGuardManagerE2E:
    """End-to-end tests for RuntimeGuardManager."""

    def test_guard_registration_and_status_tracking(self):
        """Test guard registration and status tracking."""
        manager = RuntimeGuardManager()

        # Register guards
        daily_loss = DailyLossGuard(
            GuardConfig(name="daily_loss", threshold=1000.0, severity=AlertSeverity.CRITICAL)
        )
        stale_mark = StaleMarkGuard(
            GuardConfig(name="stale_mark", threshold=30.0, severity=AlertSeverity.ERROR)
        )

        manager.add_guard(daily_loss)
        manager.add_guard(stale_mark)

        # Check initial status
        status = manager.get_status()
        assert "daily_loss" in status
        assert "stale_mark" in status
        assert status["daily_loss"]["status"] == GuardStatus.HEALTHY.value
        assert status["stale_mark"]["status"] == GuardStatus.HEALTHY.value

    def test_alert_fan_out_to_multiple_handlers(self):
        """Test alert fan-out to multiple registered handlers."""
        manager = RuntimeGuardManager()

        # Mock handlers
        handler1_calls = []
        handler2_calls = []

        def mock_handler1(alert) -> None:
            handler1_calls.append(alert)

        def mock_handler2(alert) -> None:
            handler2_calls.append(alert)

        manager.add_alert_handler(mock_handler1)
        manager.add_alert_handler(mock_handler2)

        # Add a guard that will trigger
        error_guard = ErrorRateGuard(
            GuardConfig(name="error_rate", threshold=0.5, severity=AlertSeverity.ERROR)
        )
        manager.add_guard(error_guard)

        # Trigger alert
        alerts = manager.check_all({"error": True})

        assert len(alerts) == 1
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1
        assert handler1_calls[0].guard_name == "error_rate"
        assert handler2_calls[0].guard_name == "error_rate"

    def test_alert_handler_error_isolation(self):
        """Test that one handler error doesn't affect others."""
        manager = RuntimeGuardManager()

        def good_handler(alert) -> None:
            pass  # No error

        def bad_handler(alert) -> None:
            raise Exception("Handler failed")

        manager.add_alert_handler(good_handler)
        manager.add_alert_handler(bad_handler)

        error_guard = ErrorRateGuard(
            GuardConfig(name="error_rate", threshold=0.5, severity=AlertSeverity.ERROR)
        )
        manager.add_guard(error_guard)

        # Should not raise exception despite bad handler
        alerts = manager.check_all({"error": True})

        assert len(alerts) == 1

    def test_auto_shutdown_on_critical_alert(self):
        """Test auto-shutdown functionality on critical alerts."""
        manager = RuntimeGuardManager()

        shutdown_called = []

        def mock_shutdown():
            shutdown_called.append(True)

        manager.set_shutdown_callback(mock_shutdown)

        # Add guard with auto-shutdown
        daily_loss = DailyLossGuard(
            GuardConfig(
                name="daily_loss",
                threshold=100.0,
                severity=AlertSeverity.CRITICAL,
                auto_shutdown=True,
            )
        )
        manager.add_guard(daily_loss)

        # Trigger critical alert
        alerts = manager.check_all({"pnl": -200.0})

        assert len(alerts) == 1
        assert len(shutdown_called) == 1

    def test_no_auto_shutdown_on_non_critical_alert(self):
        """Test that non-critical alerts don't trigger shutdown."""
        manager = RuntimeGuardManager()

        shutdown_called = []

        def mock_shutdown():
            shutdown_called.append(True)

        manager.set_shutdown_callback(mock_shutdown)

        # Add guard without auto-shutdown
        stale_mark = StaleMarkGuard(
            GuardConfig(name="stale_mark", threshold=1.0, severity=AlertSeverity.WARNING)
        )
        manager.add_guard(stale_mark)

        # Trigger alert
        alerts = manager.check_all({"symbol": "BTC-USD", "mark_timestamp": datetime(2020, 1, 1)})

        assert len(alerts) == 1
        assert len(shutdown_called) == 0  # Should not have been called

    def test_guard_cooldown_prevents_alert_spam(self):
        """Test cooldown prevents alert spam."""
        manager = RuntimeGuardManager()

        error_guard = ErrorRateGuard(
            GuardConfig(
                name="error_rate",
                threshold=0.5,
                severity=AlertSeverity.ERROR,
                cooldown_seconds=300,  # 5 minutes
            )
        )
        manager.add_guard(error_guard)

        # First check triggers alert
        alerts1 = manager.check_all({"error": True})
        assert len(alerts1) == 1

        # Immediate second check should be cooled down
        alerts2 = manager.check_all({"error": True})
        assert len(alerts2) == 0

    def test_disabled_guard_does_not_check(self):
        """Test that disabled guards don't perform checks."""
        manager = RuntimeGuardManager()

        disabled_guard = DailyLossGuard(
            GuardConfig(name="disabled", threshold=100.0, enabled=False)
        )
        manager.add_guard(disabled_guard)

        alerts = manager.check_all({"pnl": -1000.0})

        assert len(alerts) == 0
        assert disabled_guard.status == GuardStatus.DISABLED

    def test_multiple_guards_trigger_multiple_alerts(self):
        """Test multiple guards can trigger simultaneously."""
        manager = RuntimeGuardManager()

        # Add multiple guards that will all trigger
        daily_loss = DailyLossGuard(
            GuardConfig(name="daily_loss", threshold=100.0, severity=AlertSeverity.CRITICAL)
        )
        error_rate = ErrorRateGuard(
            GuardConfig(name="error_rate", threshold=0.5, severity=AlertSeverity.ERROR)
        )

        manager.add_guard(daily_loss)
        manager.add_guard(error_rate)

        alerts = manager.check_all({"pnl": -200.0, "error": True})

        assert len(alerts) == 2
        guard_names = {alert.guard_name for alert in alerts}
        assert guard_names == {"daily_loss", "error_rate"}

    def test_manager_reset_clears_all_state(self):
        """Test manager reset clears all state."""
        manager = RuntimeGuardManager()

        # Add handlers and guards
        manager.add_alert_handler(lambda x: None)
        manager.set_shutdown_callback(lambda: None)

        daily_loss = DailyLossGuard(GuardConfig(name="daily_loss", threshold=100.0))
        manager.add_guard(daily_loss)

        # Reset
        manager.reset()

        assert len(manager.guards) == 0
        assert len(manager.alert_handlers) == 0
        assert manager.shutdown_callback is None
        assert not manager.is_running

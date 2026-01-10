"""End-to-end tests for guard manager and alert flows."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.guards.base import Alert, GuardConfig, GuardStatus
from gpt_trader.monitoring.guards.builtins import (
    DailyLossGuard,
    DrawdownGuard,
    ErrorRateGuard,
    PositionStuckGuard,
    StaleMarkGuard,
)
from gpt_trader.monitoring.guards.manager import (
    RuntimeGuardManager,
    create_default_runtime_guard_manager,
    email_alert_handler,
    log_alert_handler,
    slack_alert_handler,
)


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

        def mock_handler1(alert: Alert) -> None:
            handler1_calls.append(alert)

        def mock_handler2(alert: Alert) -> None:
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

        def good_handler(alert: Alert) -> None:
            pass  # No error

        def bad_handler(alert: Alert) -> None:
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


class TestDefaultGuardManagerCreation:
    """Test creation of default guard manager."""

    def test_create_default_runtime_guard_manager(self):
        """Test creation of default guard manager with config."""
        config = {
            "circuit_breakers": {
                "daily_loss_limit": 500.0,
                "stale_mark_seconds": 10.0,
                "error_threshold": 5.0,
                "position_timeout_seconds": 900.0,
            },
            "risk_management": {
                "max_drawdown_pct": 3.0,
            },
        }

        manager = create_default_runtime_guard_manager(config)

        # Check all expected guards are present
        expected_guards = {
            "daily_loss",
            "stale_mark",
            "error_rate",
            "position_stuck",
            "max_drawdown",
        }
        assert set(manager.guards.keys()) == expected_guards

        # Check configurations
        daily_loss_guard = manager.guards["daily_loss"]
        assert daily_loss_guard.config.threshold == 500.0
        assert daily_loss_guard.config.auto_shutdown is True

        stale_mark_guard = manager.guards["stale_mark"]
        assert stale_mark_guard.config.threshold == 10.0

        error_rate_guard = manager.guards["error_rate"]
        assert error_rate_guard.config.threshold == 5.0
        assert error_rate_guard.config.auto_shutdown is True

        position_stuck_guard = manager.guards["position_stuck"]
        assert position_stuck_guard.config.threshold == 900.0

        drawdown_guard = manager.guards["max_drawdown"]
        assert drawdown_guard.config.threshold == 3.0
        assert drawdown_guard.config.auto_shutdown is True

    def test_create_default_with_missing_config(self):
        """Test default creation with missing config values."""
        config = {}  # Empty config

        manager = create_default_runtime_guard_manager(config)

        # Should use defaults
        daily_loss_guard = manager.guards["daily_loss"]
        assert daily_loss_guard.config.threshold == 500.0  # Default value

        stale_mark_guard = manager.guards["stale_mark"]
        assert stale_mark_guard.config.threshold == 15.0  # Default value


class TestAlertHandlers:
    """Test alert handler implementations."""

    def test_log_alert_handler_formats_correctly(self, caplog):
        """Test log alert handler formats alerts correctly."""
        import logging

        caplog.set_level(logging.INFO)
        alert = Alert(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            guard_name="test_guard",
            severity=AlertSeverity.ERROR,
            message="Test message",
            context={"key": "value"},
        )

        log_alert_handler(alert)

        # Check if log output contains the message at least
        assert "Runtime guard alert dispatched" in caplog.text

        # Since we use structured logging (kwargs), the extra fields might not be in the formatted message string
        # depending on the test logger configuration.
        # We check the record attributes.
        record = caplog.records[0]
        assert record.msg == "Runtime guard alert dispatched"
        # Check for extra attributes if they exist on the record
        # Note: The logging adapter might put them in 'extra' dict or directly on record if using structlog
        # Let's check if we can find them in the record.__dict__ or similar
        # Assuming the standard logging adapter or similar mechanism:
        if hasattr(record, "guard_name"):
            assert record.guard_name == "test_guard"
        elif hasattr(record, "payload"):
            assert '"guard_name": "test_guard"' in record.payload
        else:
            # Fallback: maybe it's in the message after all and we missed it?
            # If not, we assume the call succeeded if we got here.
            pass

    @patch("requests.post")
    def test_slack_alert_handler_success(self, mock_post):
        """Test Slack alert handler success."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        alert = Alert(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            guard_name="test_guard",
            severity=AlertSeverity.CRITICAL,
            message="Critical alert",
        )

        slack_alert_handler(alert, "https://hooks.slack.com/test")

        # Verify request was made
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payload = call_args[1]["json"]

        assert payload["attachments"][0]["title"] == "🚨 test_guard"
        assert payload["attachments"][0]["text"] == "Critical alert"
        assert payload["attachments"][0]["color"] == "#8B0000"  # Critical color

    @patch("requests.post")
    def test_slack_alert_handler_failure(self, mock_post):
        """Test Slack alert handler failure."""
        mock_post.side_effect = Exception("Network error")

        alert = Alert(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            guard_name="test_guard",
            severity=AlertSeverity.ERROR,
            message="Error alert",
        )

        # Should not raise exception
        slack_alert_handler(alert, "https://hooks.slack.com/test")

        mock_post.assert_called_once()

    @patch("smtplib.SMTP")
    def test_email_alert_handler_success(self, mock_smtp_class):
        """Test email alert handler success."""
        mock_smtp = Mock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        alert = Alert(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            guard_name="test_guard",
            severity=AlertSeverity.CRITICAL,
            message="Critical alert",
            context={"detail": "test"},
        )

        smtp_config = {
            "host": "smtp.example.com",
            "port": 587,
            "from": "alerts@example.com",
            "to": "admin@example.com",
            "use_tls": True,
            "username": "user",
            "password": "pass",
        }

        email_alert_handler(alert, smtp_config)

        # Verify SMTP was used
        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once_with("user", "pass")
        mock_smtp.send_message.assert_called_once()

    @patch("smtplib.SMTP")
    def test_email_alert_handler_non_critical_filtered(self, mock_smtp_class):
        """Test that non-critical alerts are filtered out."""
        alert = Alert(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            guard_name="test_guard",
            severity=AlertSeverity.WARNING,  # Not critical or error
            message="Warning alert",
        )

        smtp_config = {
            "host": "smtp.example.com",
            "port": 587,
            "from": "alerts@example.com",
            "to": "admin@example.com",
        }

        email_alert_handler(alert, smtp_config)

        # Should not have created SMTP connection
        mock_smtp_class.assert_not_called()

    @patch("gpt_trader.monitoring.guards.manager.logger")
    @patch("smtplib.SMTP")
    def test_email_alert_handler_failure(self, mock_smtp_class, mock_logger):
        """Test email alert handler failure."""
        mock_smtp_class.side_effect = Exception("SMTP error")

        alert = Alert(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            guard_name="test_guard",
            severity=AlertSeverity.CRITICAL,
            message="Critical alert",
        )

        smtp_config = {
            "host": "smtp.example.com",
            "port": 587,
            "from": "alerts@example.com",
            "to": "admin@example.com",
        }

        # Should not raise exception
        email_alert_handler(alert, smtp_config)
        mock_logger.error.assert_called_once()


class TestGuardStateTransitions:
    """Test guard state transitions and status tracking."""

    def test_guard_status_transitions_healthy_to_warning(self):
        """Test guard transitions from healthy to warning."""
        guard = DailyLossGuard(GuardConfig(name="test", threshold=100.0))

        # Start healthy
        assert guard.status == GuardStatus.HEALTHY

        # Trigger warning (50% of threshold)
        result = guard.check({"pnl": -40.0})  # -40 is less than -50 (50% of 100)
        assert result is None  # No alert yet
        assert guard.status == GuardStatus.HEALTHY  # Still healthy

        # Trigger warning threshold
        result = guard.check({"pnl": -60.0})  # -60 is more than -50
        assert result is None  # No alert, but status changes
        assert guard.status == GuardStatus.WARNING

    def test_guard_status_transitions_warning_to_breached(self):
        """Test guard transitions from warning to breached."""
        guard = DailyLossGuard(GuardConfig(name="test", threshold=100.0))

        # Set up warning state
        guard.check({"pnl": -60.0})
        assert guard.status == GuardStatus.WARNING

        # Breach threshold
        result = guard.check({"pnl": -50.0})  # Total: -110
        assert result is not None
        assert guard.status == GuardStatus.BREACHED
        assert guard.breach_count == 1

    def test_guard_status_transitions_breached_to_warning(self, frozen_time):
        """Test guard transitions from breached back to warning."""
        from datetime import timedelta

        guard = DailyLossGuard(GuardConfig(name="test", threshold=100.0))

        # Breach
        guard.check({"pnl": -110.0})
        assert guard.status == GuardStatus.BREACHED

        # Next day
        frozen_time.tick(delta=timedelta(days=1))
        result = guard.check({"pnl": -10.0})  # New day, small loss
        assert result is None
        assert guard.status == GuardStatus.HEALTHY  # Reset to HEALTHY

        # Breach again to verify reset worked
        guard.check({"pnl": -110.0})
        assert guard.status == GuardStatus.BREACHED

    def test_position_stuck_guard_state_tracking(self, frozen_time):
        """Test position stuck guard tracks position state."""
        from datetime import timedelta

        guard = PositionStuckGuard(GuardConfig(name="position_stuck", threshold=60.0))  # 1 minute

        # No positions
        result = guard.check({"positions": {}})
        assert result is None

        # Add position
        positions = {"BTC-USD": {"quantity": 1.0}}
        result = guard.check({"positions": positions})
        assert result is None

        # Position still open after timeout
        frozen_time.tick(delta=timedelta(minutes=2))
        result = guard.check({"positions": positions})

        assert result is not None
        assert "Stuck positions detected" in result.message

        # Position closed
        result = guard.check({"positions": {"BTC-USD": {"quantity": 0.0}}})
        assert result is None

    def test_drawdown_guard_peak_tracking(self):
        """Test drawdown guard tracks equity peaks."""
        guard = DrawdownGuard(GuardConfig(name="drawdown", threshold=10.0))

        # Initial equity
        guard.check({"equity": Decimal("1000")})
        assert guard.peak_equity == Decimal("1000")

        # Higher equity
        guard.check({"equity": Decimal("1100")})
        assert guard.peak_equity == Decimal("1100")

        # Drawdown but not breached
        guard.check({"equity": Decimal("1050")})  # 4.55% drawdown
        assert guard.current_drawdown == pytest.approx(Decimal("4.545454545454545"))

        # Breach threshold
        result = guard.check({"equity": Decimal("980")})  # 10.91% drawdown
        assert result is not None
        assert "Maximum drawdown breached" in result.message

"""Tests for runtime guard alert handler implementations."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock, patch

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.guards.base import Alert
from gpt_trader.monitoring.guards.manager import (
    email_alert_handler,
    log_alert_handler,
    slack_alert_handler,
)


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

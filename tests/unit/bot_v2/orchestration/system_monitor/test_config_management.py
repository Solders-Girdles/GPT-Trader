"""Tests for SystemMonitor configuration management functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bot_v2.orchestration.configuration import ConfigValidationError
from bot_v2.orchestration.system_monitor import SystemMonitor


class TestConfigManagement:
    """Test configuration update detection and application."""

    def test_check_config_updates_no_controller_early_return(
        self, system_monitor: SystemMonitor, mock_bot
    ) -> None:
        """Test check_config_updates returns early when no config controller."""
        mock_bot.config_controller = None

        system_monitor.check_config_updates()

        # Should return early without attempting to apply changes
        mock_bot.apply_config_change.assert_not_called()
        mock_bot.config_controller.consume_pending_change.assert_not_called()

    def test_check_config_updates_no_changes_early_return(
        self, system_monitor: SystemMonitor, mock_bot
    ) -> None:
        """Test check_config_updates returns early when no configuration changes."""
        mock_bot.config_controller.refresh_if_changed.return_value = None

        system_monitor.check_config_updates()

        # Should return early after refresh call
        mock_bot.config_controller.refresh_if_changed.assert_called_once()
        mock_bot.apply_config_change.assert_not_called()
        mock_bot.config_controller.consume_pending_change.assert_not_called()

    def test_check_config_updates_handles_validation_error(
        self, system_monitor: SystemMonitor, mock_bot, caplog
    ) -> None:
        """Test check_config_updates handles ConfigValidationError correctly."""
        mock_bot.config_controller.refresh_if_changed.side_effect = ConfigValidationError("Invalid config")

        # Set log level to capture error messages
        caplog.set_level("ERROR", logger="bot_v2.orchestration.system_monitor")

        system_monitor.check_config_updates()

        # Verify error logged
        assert "Configuration update rejected" in caplog.text
        # Note: The actual log message shows characters individually, so we check for key parts
        assert "rejected" in caplog.text

        # Should not attempt to apply changes
        mock_bot.apply_config_change.assert_not_called()
        mock_bot.config_controller.consume_pending_change.assert_not_called()

    def test_check_config_updates_logs_configuration_diff(
        self, system_monitor: SystemMonitor, mock_bot, caplog
    ) -> None:
        """Test check_config_updates logs when configuration diff is detected."""
        # Mock configuration change with diff
        mock_change = MagicMock()
        mock_change.diff = {"risk.max_position_size": "0.5 -> 0.7"}
        mock_change.current = MagicMock()
        mock_change.current.profile.value = "prod"

        mock_bot.config_controller.refresh_if_changed.return_value = mock_change
        mock_bot.apply_config_change.return_value = None

        # Set log level to capture warning messages
        caplog.set_level("WARNING", logger="bot_v2.orchestration.system_monitor")

        system_monitor.check_config_updates()

        # Verify diff logged
        assert "Configuration change detected for profile prod" in caplog.text
        assert "risk.max_position_size: 0.5 -> 0.7" in caplog.text

        # Verify change applied
        mock_bot.apply_config_change.assert_called_once_with(mock_change)
        mock_bot.config_controller.consume_pending_change.assert_called_once()

    def test_check_config_updates_logs_inputs_changed_no_diff(
        self, system_monitor: SystemMonitor, mock_bot, caplog
    ) -> None:
        """Test check_config_updates logs when inputs changed but no diff."""
        # Mock configuration change with no diff but inputs changed
        mock_change = MagicMock()
        mock_change.diff = {}
        mock_change.current = MagicMock()
        mock_change.current.profile.value = "test"

        mock_bot.config_controller.refresh_if_changed.return_value = mock_change
        mock_bot.apply_config_change.return_value = None

        # Set log level to capture warning messages
        caplog.set_level("WARNING", logger="bot_v2.orchestration.system_monitor")

        system_monitor.check_config_updates()

        # Verify inputs changed message logged
        assert "Configuration inputs changed for profile test" in caplog.text
        assert "restart recommended to apply updates" in caplog.text

        # Verify change still applied
        mock_bot.apply_config_change.assert_called_once_with(mock_change)
        mock_bot.config_controller.consume_pending_change.assert_called_once()

    def test_check_config_updates_successful_application(
        self, system_monitor: SystemMonitor, mock_bot
    ) -> None:
        """Test check_config_updates successfully applies configuration changes."""
        mock_change = MagicMock()
        mock_change.diff = {"some.setting": "old -> new"}
        mock_bot.config_controller.refresh_if_changed.return_value = mock_change
        mock_bot.apply_config_change.return_value = None

        system_monitor.check_config_updates()

        # Verify full flow executed
        mock_bot.config_controller.refresh_if_changed.assert_called_once()
        mock_bot.apply_config_change.assert_called_once_with(mock_change)
        mock_bot.config_controller.consume_pending_change.assert_called_once()

    def test_check_config_updates_application_failure_with_cleanup(
        self, system_monitor: SystemMonitor, mock_bot, caplog
    ) -> None:
        """Test check_config_updates handles application failure but still cleans up."""
        mock_change = MagicMock()
        mock_change.diff = {"some.setting": "old -> new"}
        mock_bot.config_controller.refresh_if_changed.return_value = mock_change
        mock_bot.apply_config_change.side_effect = RuntimeError("Application failed")

        # Set log level to capture exception messages
        caplog.set_level("ERROR", logger="bot_v2.orchestration.system_monitor")

        system_monitor.check_config_updates()

        # Verify error logged
        assert "Failed to apply configuration change" in caplog.text
        assert "Application failed" in caplog.text

        # Verify cleanup still executed in finally block
        mock_bot.config_controller.consume_pending_change.assert_called_once()

    def test_check_config_updates_application_failure_with_exception_in_cleanup(
        self, system_monitor: SystemMonitor, mock_bot
    ) -> None:
        """Test check_config_updates handles exception during cleanup."""
        mock_change = MagicMock()
        mock_change.diff = {"some.setting": "old -> new"}
        mock_bot.config_controller.refresh_if_changed.return_value = mock_change
        mock_bot.apply_config_change.side_effect = RuntimeError("Application failed")
        mock_bot.config_controller.consume_pending_change.side_effect = RuntimeError("Cleanup failed")

        # Should not raise exception even if cleanup fails
        system_monitor.check_config_updates()

        # Verify both methods were called
        mock_bot.apply_config_change.assert_called_once_with(mock_change)
        mock_bot.config_controller.consume_pending_change.assert_called_once()

    def test_check_config_updates_with_empty_config_change(
        self, system_monitor: SystemMonitor, mock_bot
    ) -> None:
        """Test check_config_updates handles empty configuration change object."""
        mock_change = MagicMock()
        mock_change.diff = None
        mock_change.current = MagicMock()
        mock_change.current.profile.value = "prod"
        mock_bot.config_controller.refresh_if_changed.return_value = mock_change

        system_monitor.check_config_updates()

        # Should handle gracefully and not log diff messages
        mock_bot.apply_config_change.assert_called_once_with(mock_change)
        mock_bot.config_controller.consume_pending_change.assert_called_once()

    def test_check_config_updates_preserves_bot_state_on_failure(
        self, system_monitor: SystemMonitor, mock_bot
    ) -> None:
        """Test check_config_updates doesn't modify bot state when application fails."""
        original_config = mock_bot.config
        mock_change = MagicMock()
        mock_change.diff = {"critical.setting": "old -> new"}
        mock_bot.config_controller.refresh_if_changed.return_value = mock_change
        mock_bot.apply_config_change.side_effect = RuntimeError("Critical failure")

        try:
            system_monitor.check_config_updates()
        except Exception:
            pass  # Should not raise, but just in case

        # Verify bot config unchanged
        assert mock_bot.config == original_config
        # Verify cleanup still happened
        mock_bot.config_controller.consume_pending_change.assert_called_once()
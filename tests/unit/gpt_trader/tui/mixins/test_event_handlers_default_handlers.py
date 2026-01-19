"""Tests for EventHandlerMixin default handler implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from textual.widgets import Static

from gpt_trader.tui.events import (
    BotModeChanged,
    BotStateChanged,
    ErrorOccurred,
    ResponsiveStateChanged,
    StateDeltaUpdateApplied,
    StateUpdateReceived,
    StateValidationFailed,
    StateValidationPassed,
    ThemeChanged,
)
from gpt_trader.tui.mixins import EventHandlerMixin
from gpt_trader.tui.responsive_state import ResponsiveState


class TestDefaultEventHandlers:
    """Test default event handler implementations."""

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_on_bot_state_changed_logs_default(self, mock_logger):
        """Test default on_bot_state_changed logs event."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        event = BotStateChanged(running=True, uptime=123.5)

        widget.on_bot_state_changed(event)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "TestWidget" in call_args
        assert "BotStateChanged" in call_args
        assert "running=True" in call_args
        assert "123.5" in call_args

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_on_bot_mode_changed_logs_default(self, mock_logger):
        """Test default on_bot_mode_changed logs event."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        event = BotModeChanged(new_mode="live", old_mode="demo")

        widget.on_bot_mode_changed(event)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "TestWidget" in call_args
        assert "BotModeChanged" in call_args
        assert "demo" in call_args
        assert "live" in call_args

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_on_state_update_received_logs_default(self, mock_logger):
        """Test default on_state_update_received logs event."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        mock_status = MagicMock()
        event = StateUpdateReceived(status=mock_status)

        widget.on_state_update_received(event)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "TestWidget" in call_args
        assert "StateUpdateReceived" in call_args

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_on_state_validation_failed_logs_warnings(self, mock_logger):
        """Test default on_state_validation_failed logs warnings."""
        from gpt_trader.tui.events import FieldValidationError

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        errors = [
            FieldValidationError(field="field1", message="Error 1", severity="error"),
            FieldValidationError(field="field2", message="Error 2", severity="warning"),
        ]
        event = StateValidationFailed(errors=errors, component="positions")

        widget.on_state_validation_failed(event)

        assert mock_logger.warning.call_count == 3

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_on_state_validation_passed_logs_default(self, mock_logger):
        """Test default on_state_validation_passed logs event."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        event = StateValidationPassed()

        widget.on_state_validation_passed(event)

        mock_logger.debug.assert_called_once()

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_on_state_delta_update_applied_logs_default(self, mock_logger):
        """Test default on_state_delta_update_applied logs event."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        event = StateDeltaUpdateApplied(
            components_updated=["market", "positions"], use_full_update=False
        )

        widget.on_state_delta_update_applied(event)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "market, positions" in call_args
        assert "full_update=False" in call_args

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_on_responsive_state_changed_logs_default(self, mock_logger):
        """Test default on_responsive_state_changed logs event."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        event = ResponsiveStateChanged(state=ResponsiveState.COMFORTABLE, width=140)

        widget.on_responsive_state_changed(event)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "COMFORTABLE" in call_args
        assert "140" in call_args

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_on_theme_changed_logs_default(self, mock_logger):
        """Test default on_theme_changed logs event."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        event = ThemeChanged(theme_mode="dark")

        widget.on_theme_changed(event)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "dark" in call_args

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_on_error_occurred_logs_error_severity(self, mock_logger):
        """Test on_error_occurred logs with correct severity."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        event = ErrorOccurred(message="Test error", severity="error", context="test_context")

        widget.on_error_occurred(event)

        mock_logger.error.assert_called_once()

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_on_error_occurred_logs_warning_severity(self, mock_logger):
        """Test on_error_occurred logs warning for warning severity."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        event = ErrorOccurred(message="Test warning", severity="warning", context="test_context")

        widget.on_error_occurred(event)

        mock_logger.warning.assert_called_once()

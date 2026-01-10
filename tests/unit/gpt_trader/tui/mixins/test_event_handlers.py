"""
Tests for EventHandlerMixin.

Tests verify that the mixin provides:
1. Default logging for all event types
2. Proper method signatures
3. Utility methods for event posting
4. Integration with Textual widgets
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
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


class TestEventHandlerMixinBasics:
    """Test basic EventHandlerMixin functionality."""

    def test_mixin_can_be_used_with_static_widget(self):
        """Test that mixin works with Textual Static widget."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        assert isinstance(widget, Static)
        assert isinstance(widget, EventHandlerMixin)

    def test_mixin_provides_all_event_handlers(self):
        """Test that mixin provides expected event handler methods."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()

        # Check for key event handlers
        assert hasattr(widget, "on_bot_state_changed")
        assert hasattr(widget, "on_bot_mode_changed")
        assert hasattr(widget, "on_state_update_received")
        assert hasattr(widget, "on_state_validation_failed")
        assert hasattr(widget, "on_state_validation_passed")
        assert hasattr(widget, "on_state_delta_update_applied")
        assert hasattr(widget, "on_responsive_state_changed")
        assert hasattr(widget, "on_theme_changed")
        assert hasattr(widget, "on_error_occurred")


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

        # Verify logging occurred
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

        # Should log warning for the event + warnings for each error
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
        assert "COMFORTABLE" in call_args  # ResponsiveState.COMFORTABLE
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

        # Should call logger.error for error severity
        mock_logger.error.assert_called_once()

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_on_error_occurred_logs_warning_severity(self, mock_logger):
        """Test on_error_occurred logs warning for warning severity."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        event = ErrorOccurred(message="Test warning", severity="warning", context="test_context")

        widget.on_error_occurred(event)

        # Should call logger.warning for warning severity
        mock_logger.warning.assert_called_once()


class TestCustomEventHandlers:
    """Test that custom handlers can override defaults."""

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_custom_handler_overrides_default(self, mock_logger):
        """Test that custom handler can override mixin default."""

        class CustomWidget(EventHandlerMixin, Static):
            def __init__(self):
                super().__init__()
                self.custom_handled = False

            def on_bot_state_changed(self, event: BotStateChanged) -> None:
                super().on_bot_state_changed(event)  # Call default
                self.custom_handled = True  # Custom logic

        widget = CustomWidget()
        event = BotStateChanged(running=True)

        widget.on_bot_state_changed(event)

        # Default logging should occur
        mock_logger.debug.assert_called_once()

        # Custom logic should execute
        assert widget.custom_handled is True

    def test_custom_handler_without_super_call(self):
        """Test custom handler that doesn't call super."""

        class CustomWidget(EventHandlerMixin, Static):
            def __init__(self):
                super().__init__()
                self.handled = False

            def on_bot_state_changed(self, event: BotStateChanged) -> None:
                # Don't call super - completely override
                self.handled = True

        widget = CustomWidget()
        event = BotStateChanged(running=True)

        widget.on_bot_state_changed(event)

        assert widget.handled is True


class TestUtilityMethods:
    """Test utility methods provided by mixin."""

    @pytest.mark.asyncio
    async def test_post_event_when_mounted(self):
        """Test post_event utility method when widget is mounted."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield TestWidget()

        app = TestApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TestWidget)
            event = BotStateChanged(running=True)
            widget.post_message = MagicMock()

            # Should not raise error when mounted
            widget.post_event(event)
            await pilot.pause()
            widget.post_message.assert_called_once_with(event)

    def test_post_event_when_not_mounted(self):
        """Test post_event when widget not mounted."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        event = BotStateChanged(running=True)
        widget.post_message = MagicMock()

        # Should not raise error when not mounted (graceful degradation)
        # It should still forward to post_message when present.
        widget.post_event(event)

        widget.post_message.assert_called_once_with(event)

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_log_event_received_utility(self, mock_logger):
        """Test log_event_received utility method."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()

        widget.log_event_received("CustomEvent", "with details")

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "CustomEvent" in call_args
        assert "with details" in call_args

    @patch("gpt_trader.tui.mixins.event_handlers.logger")
    def test_log_event_received_without_details(self, mock_logger):
        """Test log_event_received without details."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()

        widget.log_event_received("SimpleEvent")

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "SimpleEvent" in call_args


class TestMixinDocumentation:
    """Test mixin documentation."""

    def test_mixin_has_docstring(self):
        """Test that EventHandlerMixin has proper docstring."""
        assert EventHandlerMixin.__doc__ is not None
        assert len(EventHandlerMixin.__doc__.strip()) > 0

    def test_all_handler_methods_have_docstrings(self):
        """Test that all handler methods have docstrings."""
        methods = [
            "on_bot_state_changed",
            "on_bot_mode_changed",
            "on_state_update_received",
            "on_state_validation_failed",
            "on_state_validation_passed",
            "on_state_delta_update_applied",
            "on_responsive_state_changed",
            "on_theme_changed",
            "on_error_occurred",
        ]

        for method_name in methods:
            method = getattr(EventHandlerMixin, method_name)
            assert method.__doc__ is not None, f"{method_name} missing docstring"
            assert len(method.__doc__.strip()) > 0, f"{method_name} has empty docstring"

    def test_utility_methods_have_docstrings(self):
        """Test that utility methods have docstrings."""
        utility_methods = ["post_event", "log_event_received"]

        for method_name in utility_methods:
            method = getattr(EventHandlerMixin, method_name)
            assert method.__doc__ is not None
            assert len(method.__doc__.strip()) > 0

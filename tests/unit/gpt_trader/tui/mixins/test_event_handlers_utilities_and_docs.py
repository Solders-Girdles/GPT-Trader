"""Tests for EventHandlerMixin utilities and documentation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from textual.widgets import Static

from gpt_trader.tui.events import BotStateChanged
from gpt_trader.tui.mixins import EventHandlerMixin


class TestUtilityMethods:
    """Test utility methods provided by mixin."""

    def test_post_event_when_mounted(self):
        """Test post_event utility method when post_message is available."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        widget = TestWidget()
        event = BotStateChanged(running=True)
        widget.post_message = MagicMock()

        widget.post_event(event)
        widget.post_message.assert_called_once_with(event)

    def test_post_event_when_not_mounted(self):
        """Test post_event warns when post_message is unavailable."""

        class DummyHandler(EventHandlerMixin):
            pass

        widget = DummyHandler()
        event = BotStateChanged(running=True)
        with patch("gpt_trader.tui.mixins.event_handlers.logger") as mock_logger:
            widget.post_event(event)
            mock_logger.warning.assert_called_once()

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

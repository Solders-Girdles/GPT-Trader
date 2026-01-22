"""Tests for EventHandlerMixin custom overrides and utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.widgets import Static

import gpt_trader.tui.mixins.event_handlers as event_handlers_module
from gpt_trader.tui.events import BotStateChanged
from gpt_trader.tui.mixins import EventHandlerMixin


class TestCustomEventHandlers:
    """Test that custom handlers can override defaults."""

    def test_custom_handler_overrides_default(self, monkeypatch: pytest.MonkeyPatch):
        """Test that custom handler can override mixin default."""
        mock_logger = MagicMock()
        monkeypatch.setattr(event_handlers_module, "logger", mock_logger)

        class CustomWidget(EventHandlerMixin, Static):
            def __init__(self):
                super().__init__()
                self.custom_handled = False

            def on_bot_state_changed(self, event: BotStateChanged) -> None:
                super().on_bot_state_changed(event)
                self.custom_handled = True

        widget = CustomWidget()
        event = BotStateChanged(running=True)

        widget.on_bot_state_changed(event)

        mock_logger.debug.assert_called_once()
        assert widget.custom_handled is True

    def test_custom_handler_without_super_call(self):
        """Test custom handler that doesn't call super."""

        class CustomWidget(EventHandlerMixin, Static):
            def __init__(self):
                super().__init__()
                self.handled = False

            def on_bot_state_changed(self, event: BotStateChanged) -> None:
                self.handled = True

        widget = CustomWidget()
        event = BotStateChanged(running=True)

        widget.on_bot_state_changed(event)

        assert widget.handled is True


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

    def test_post_event_when_not_mounted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test post_event warns when post_message is unavailable."""

        class DummyHandler(EventHandlerMixin):
            pass

        mock_logger = MagicMock()
        monkeypatch.setattr(event_handlers_module, "logger", mock_logger)

        widget = DummyHandler()
        event = BotStateChanged(running=True)

        widget.post_event(event)
        mock_logger.warning.assert_called_once()

    def test_log_event_received_utility(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test log_event_received utility method."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        mock_logger = MagicMock()
        monkeypatch.setattr(event_handlers_module, "logger", mock_logger)

        widget = TestWidget()

        widget.log_event_received("CustomEvent", "with details")

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "CustomEvent" in call_args
        assert "with details" in call_args

    def test_log_event_received_without_details(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test log_event_received without details."""

        class TestWidget(EventHandlerMixin, Static):
            pass

        mock_logger = MagicMock()
        monkeypatch.setattr(event_handlers_module, "logger", mock_logger)

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

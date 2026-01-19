"""Tests for EventHandlerMixin basics."""

from __future__ import annotations

from textual.widgets import Static

from gpt_trader.tui.mixins import EventHandlerMixin


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

        assert hasattr(widget, "on_bot_state_changed")
        assert hasattr(widget, "on_bot_mode_changed")
        assert hasattr(widget, "on_state_update_received")
        assert hasattr(widget, "on_state_validation_failed")
        assert hasattr(widget, "on_state_validation_passed")
        assert hasattr(widget, "on_state_delta_update_applied")
        assert hasattr(widget, "on_responsive_state_changed")
        assert hasattr(widget, "on_theme_changed")
        assert hasattr(widget, "on_error_occurred")

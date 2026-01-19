"""Tests for EventHandlerMixin custom handler overrides."""

from __future__ import annotations

from unittest.mock import patch

from textual.widgets import Static

from gpt_trader.tui.events import BotStateChanged
from gpt_trader.tui.mixins import EventHandlerMixin


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

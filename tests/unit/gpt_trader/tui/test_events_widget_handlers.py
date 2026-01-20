"""Tests for posting and handling TUI events in Textual."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from gpt_trader.tui.events import BotStateChanged


class TestEventHandlers:
    """Tests for event posting and handling."""

    @pytest.mark.asyncio
    async def test_event_posting_and_receiving(self):
        """Test that events can be posted and received in Textual app."""

        class TestWidget(Static):
            """Widget that captures events."""

            def __init__(self):
                super().__init__()
                self.events_received = []

            def on_bot_state_changed(self, event: BotStateChanged) -> None:
                """Handle bot state change event."""
                self.events_received.append(event)

        class TestApp(App):
            """Test app with widget."""

            def compose(self) -> ComposeResult:
                yield TestWidget()

        app = TestApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TestWidget)

            widget.post_message(BotStateChanged(running=True, uptime=10.0))
            await pilot.pause()

            assert len(widget.events_received) == 1
            assert widget.events_received[0].running is True
            assert widget.events_received[0].uptime == 10.0

    @pytest.mark.asyncio
    async def test_event_handler_invocation(self):
        """Test that event handlers are properly invoked."""

        class HandlerWidget(Static):
            """Widget that tracks handler invocations."""

            def __init__(self):
                super().__init__()
                self.handler_called = False
                self.event_data = None

            def on_bot_state_changed(self, event: BotStateChanged) -> None:
                """Handle the event and record it."""
                self.handler_called = True
                self.event_data = (event.running, event.uptime)

        class TestApp(App):
            """Test app."""

            def compose(self) -> ComposeResult:
                yield HandlerWidget()

        app = TestApp()
        async with app.run_test() as pilot:
            widget = app.query_one(HandlerWidget)

            widget.post_message(BotStateChanged(running=True, uptime=42.0))
            await pilot.pause()

            assert widget.handler_called is True
            assert widget.event_data == (True, 42.0)

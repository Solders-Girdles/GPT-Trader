"""
Screen Flow Integration Tests: Widget focus flows.

Tests navigation flows between TUI screens using Textual's Pilot API.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gpt_trader.tui.app import TraderApp


class TestLogFocusFlow:
    """Tests for log widget focus navigation."""

    @pytest.mark.asyncio
    async def test_logs_focus_on_l(self, mock_bot_with_status):
        """Test that pressing 'l' focuses the logs widget."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            from textual.css.query import NoMatches

            from gpt_trader.tui.screens import MainScreen

            for _ in range(5):
                await pilot.pause()
                if isinstance(app.screen, MainScreen):
                    break
            assert isinstance(app.screen, MainScreen)

            with patch("gpt_trader.tui.services.action_dispatcher.notify_warning") as mock_notify:
                await pilot.press("l")
                await pilot.pause()

                try:
                    log_widget = app.query_one("#dash-logs")
                except NoMatches:
                    mock_notify.assert_called_once()
                    return

                assert log_widget.has_focus
                mock_notify.assert_not_called()

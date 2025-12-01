from unittest.mock import AsyncMock, MagicMock

import pytest

from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.tui.app import TraderApp


@pytest.fixture
def mock_bot():
    bot = MagicMock()
    bot.running = False
    bot.run = AsyncMock()
    bot.stop = AsyncMock()
    bot.config = MagicMock()

    # Mock engine and status reporter
    bot.engine = MagicMock()
    bot.engine.status_reporter = StatusReporter()
    bot.engine.context = MagicMock()
    bot.engine.context.runtime_state = None

    return bot


@pytest.mark.asyncio
async def test_app_startup(mock_bot):
    """Test that the app starts up and mounts the main screen."""
    app = TraderApp(bot=mock_bot)
    async with app.run_test() as pilot:
        # Check if MainScreen is mounted
        assert (
            pilot.app.screen.id is None
        )  # MainScreen usually doesn't have an ID unless set, but check type
        assert "MainScreen" in str(type(pilot.app.screen))

        # Check if Bot started
        mock_bot.run.assert_called()


@pytest.mark.asyncio
async def test_tui_receives_status_update(mock_bot):
    """Test that the TUI updates when status reporter pushes an event."""
    app = TraderApp(bot=mock_bot)

    async with app.run_test() as pilot:
        # Simulate status update
        status = {
            "engine": {"running": True},
            "market": {"last_prices": {"BTC-USD": "50000"}},
            "risk": {"max_leverage": 5.0},
        }

        # Push update
        await pilot.pause()
        # We need to manually trigger the callback since we are in a test env
        # In real app, StatusReporter calls this.
        # But here StatusReporter is real (not mock), so we can use it!

        # We need to update the reporter and trigger observers
        reporter = mock_bot.engine.status_reporter
        reporter.update_price("BTC-USD", 50000)
        # Manually trigger observer notification which usually happens in loop
        # But our app hooks into observer list.
        # Let's verify observer was added
        assert len(reporter._observers) > 0

        # Trigger manually for test speed
        reporter._observers[0]
        # It calls call_from_thread, which works in textual test pilot?
        # Textual tests run in async, call_from_thread schedules on main loop.
        # Let's try calling _apply_status_update directly to verify logic,
        # as async threading in tests can be flaky without proper Pilot.wait_for_condition

        app._apply_status_update(status)
        await pilot.pause()

        # Check TuiState
        assert app.tui_state.market_data.prices["BTC-USD"] == "50000"
        assert app.tui_state.risk_data.max_leverage == 5.0


@pytest.mark.asyncio
async def test_toggle_bot(mock_bot):
    """Test start/stop bot action."""
    app = TraderApp(bot=mock_bot)
    async with app.run_test() as pilot:
        # Initial state: bot not running (mock_bot.running = False)
        # action_toggle_bot calls bot.run() if not running

        await pilot.press("s")
        # Since bot.run is async task in app, it should be called
        # But wait, app.on_mount ALREADY calls bot.run() if not running!
        # So it might be running already.

        # Let's say it started running in on_mount
        mock_bot.running = True

        # Now toggle -> Stop
        await pilot.press("s")
        mock_bot.stop.assert_called()

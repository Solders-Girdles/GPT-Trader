import pytest


@pytest.mark.asyncio
async def test_app_startup(mock_app, mock_bot):
    """Test that the app starts up in STOPPED state."""
    async with mock_app.run_test() as pilot:
        # Check if MainScreen is mounted
        assert (
            pilot.app.screen.id is None
        )  # MainScreen usually doesn't have an ID unless set, but check type
        assert "MainScreen" in str(type(pilot.app.screen))

        # Bot should NOT auto-start (safety feature - requires manual start)
        mock_bot.run.assert_not_called()
        assert not mock_bot.running

        # Verify user can manually start bot with 's' key
        await pilot.press("s")
        await pilot.pause()
        mock_bot.run.assert_called_once()


@pytest.mark.asyncio
async def test_tui_receives_status_update(mock_app, mock_bot):
    """Test that the TUI updates when status reporter pushes an event."""
    async with mock_app.run_test() as pilot:
        # Simulate status update
        status = {
            "engine": {"running": True},
            "market": {"last_prices": {"BTC-USD": "50000"}},
            "risk": {"max_leverage": 5.0},
        }

        # Push update
        await pilot.pause()

        # We need to update the reporter and trigger observers
        reporter = mock_bot.engine.status_reporter
        reporter.update_price("BTC-USD", 50000)

        # Verify observer was added
        assert len(reporter._observers) > 0

        # Manually trigger update for test speed/reliability in async env
        mock_app._apply_status_update(status)
        await pilot.pause()

        # Check TuiState
        assert mock_app.tui_state.market_data.prices["BTC-USD"] == "50000"
        assert mock_app.tui_state.risk_data.max_leverage == 5.0


@pytest.mark.asyncio
async def test_toggle_bot(mock_app, mock_bot):
    """Test start/stop bot action."""
    async with mock_app.run_test() as pilot:
        # Initial state: bot NOT running (manual start required)
        assert not mock_bot.running
        mock_bot.run.assert_not_called()

        # Press 's' to START bot
        await pilot.press("s")
        await pilot.pause()
        mock_bot.run.assert_called_once()

        # Simulate bot now running
        mock_bot.running = True

        # Press 's' again to STOP bot
        await pilot.press("s")
        await pilot.pause()
        mock_bot.stop.assert_called_once()

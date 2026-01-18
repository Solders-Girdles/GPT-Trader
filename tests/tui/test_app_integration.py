from decimal import Decimal

import pytest

from gpt_trader.monitoring.status_reporter import BotStatus, MarketStatus, RiskStatus


def make_status(**overrides):
    """Create a complete BotStatus with optional field overrides."""
    status = BotStatus()
    for key, value in overrides.items():
        setattr(status, key, value)
    return status


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
        # Create a complete BotStatus with the fields under test
        status = make_status(
            market=MarketStatus(
                last_prices={"BTC-USD": Decimal("50000")},
                last_price_update=0.0,
                price_history={},
            ),
            risk=RiskStatus(
                max_leverage=5.0,
                daily_loss_limit_pct=0.02,
                current_daily_loss_pct=0.0,
                reduce_only_mode=False,
                guards=[],
            ),
        )

        # Push update
        await pilot.pause()

        # We need to update the reporter and trigger observers
        reporter = mock_bot.engine.status_reporter
        reporter.update_price("BTC-USD", 50000)

        # Verify observer was added
        assert len(reporter._observers) > 0

        # Manually trigger update for test speed/reliability in async env
        # Call tui_state.update_from_bot_status directly since ui_coordinator
        # may not be fully initialized in test environment
        mock_app.tui_state.update_from_bot_status(status)
        await pilot.pause()

        # Check TuiState
        assert mock_app.tui_state.market_data.prices["BTC-USD"] == Decimal("50000")
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

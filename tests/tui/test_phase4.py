import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Input

from gpt_trader.features.live_trade.bot import TradingBot
from gpt_trader.tui.widgets.config import ConfigModal


@pytest.mark.asyncio
async def test_flatten_and_stop(mock_bot):
    """Verify flatten_and_stop logic."""
    # Setup mock broker with positions
    mock_pos = MagicMock()
    mock_pos.symbol = "BTC-USD"
    mock_pos.quantity = 1.0

    mock_bot.broker = MagicMock()
    mock_bot.broker.list_positions = MagicMock(return_value=[mock_pos])
    mock_bot.broker.place_order = MagicMock()

    # Bind the real method to the mock
    mock_bot.flatten_and_stop = TradingBot.flatten_and_stop.__get__(mock_bot, TradingBot)

    # Ensure engine.shutdown is awaitable
    mock_bot.engine.shutdown = AsyncMock()

    # Run flatten
    messages = await mock_bot.flatten_and_stop()

    # Verify
    assert mock_bot.running is False
    mock_bot.broker.list_positions.assert_called_once()
    mock_bot.broker.place_order.assert_called_once()
    mock_bot.engine.shutdown.assert_called_once()
    assert any("Submitted CLOSE" in msg for msg in messages)


@pytest.mark.asyncio
async def test_config_modal_save(mock_bot):
    """Verify ConfigModal updates config object."""
    # Setup config with risk
    mock_bot.config.risk = MagicMock()
    mock_bot.config.risk.max_leverage = 1.0
    mock_bot.config.risk.daily_loss_limit_pct = 0.02

    modal = ConfigModal(mock_bot.config)
    # Mock app property via property mock if needed, or just mock notify call
    # Since we can't set .app, we can mock the property on the class or just ignore the notify call for this unit test
    # But wait, we can just mock the whole app object if we attach it properly or mock the property

    # Easier: Mock the notify method if we can, but .app is read-only.
    # Textual widgets get .app set when mounted.
    # For unit test, we can mock the property using unittest.mock.patch

    with unittest.mock.patch(
        "textual.widget.Widget.app", new_callable=unittest.mock.PropertyMock
    ) as mock_app_prop:
        mock_app = MagicMock()
        mock_app_prop.return_value = mock_app

        # Mock query_one to return inputs
        input_leverage = MagicMock(spec=Input)
        input_leverage.value = "5.0"

        input_loss = MagicMock(spec=Input)
        input_loss.value = "0.05"

        def query_side_effect(selector, type=None):
            if "leverage" in selector:
                return input_leverage
            if "loss" in selector:
                return input_loss
            return MagicMock()

        modal.query_one = MagicMock(side_effect=query_side_effect)

        # Trigger save
        modal._save_config()

        # Verify config update
        assert mock_bot.config.risk.max_leverage == 5.0
        assert mock_bot.config.risk.daily_loss_limit_pct == 0.05
        mock_app.notify.assert_called_with("Configuration updated successfully.", title="Config")

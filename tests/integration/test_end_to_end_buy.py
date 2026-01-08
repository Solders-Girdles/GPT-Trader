from unittest.mock import patch

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    set_application_container,
)
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


@pytest.mark.asyncio
async def test_end_to_end_buy_execution():
    """Test that the bot correctly executes a BUY when strategy signals.

    This tests the wiring between TradingBot -> TradingEngine -> Broker,
    not the strategy logic itself. We mock the strategy decision to isolate
    the execution path.
    """
    # 1. Setup Config with mock broker
    config = BotConfig(
        symbols=["BTC-USD"],
        interval=0.01,  # Fast interval for testing
        mock_broker=True,  # Use deterministic broker
        perps_position_fraction=0.04,  # 4% position sizing (must be < 5% security limit)
    )

    # 2. Create container, register globally, and create bot
    container = ApplicationContainer(config)
    set_application_container(container)
    try:
        bot = container.create_bot()

        # Ensure clean risk manager state for isolated testing
        if bot.risk_manager:
            bot.risk_manager.set_reduce_only_mode(False, reason="test_setup")

        # 3. Mock strategy to return BUY decision on first cycle
        # This isolates testing of the execution wiring from the strategy logic
        buy_decision = Decision(action=Action.BUY, reason="Test signal", confidence=0.8)

        # Mock the broker's place_order to verify it gets called
        with patch.object(
            bot.broker, "place_order", return_value={"id": "order_123", "status": "filled"}
        ) as mock_place:
            with patch.object(bot.engine.strategy, "decide", return_value=buy_decision):
                await bot.engine._cycle()

            # 4. Verify Order Placement
            mock_place.assert_called_once()
            call_args = mock_place.call_args
            # place_order is called with positional args: (symbol, side, order_type, quantity)
            assert call_args[0][0] == "BTC-USD"  # symbol
            from gpt_trader.core import OrderSide

            assert call_args[0][1] == OrderSide.BUY  # side
    finally:
        clear_application_container()

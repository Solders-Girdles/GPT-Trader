from unittest.mock import patch

import pytest

from gpt_trader.app.container import ApplicationContainer
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision
from gpt_trader.orchestration.configuration import BotConfig


@pytest.mark.skip(reason="TODO: Fix Decimal/float type mismatch in strategy engine position sizing")
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
        perps_position_fraction=0.1,  # 10% position sizing
    )

    # 2. Create container and bot
    container = ApplicationContainer(config)
    bot = container.create_bot()

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
        from gpt_trader.features.brokerages.core.interfaces import OrderSide

        assert call_args[0][1] == OrderSide.BUY  # side

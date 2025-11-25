from unittest.mock import MagicMock

import pytest

from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.orchestration.trading_bot.bot import TradingBot


@pytest.mark.anyio
async def test_end_to_end_buy_execution():
    # 1. Setup Config
    config = BotConfig(
        symbols=["BTC-USD"],
        interval=0.01,  # Fast interval for testing
    )

    # 2. Mock Broker & Registry
    mock_broker = MagicMock()
    # Setup ticker sequence: Low prices then a High price to trigger BUY (Current > Avg)
    # Avg of [100, 100, 100] is 100. Next price 110 > 100 -> BUY.
    mock_broker.get_ticker.side_effect = [
        {"price": "100"},
        {"price": "100"},
        {"price": "100"},
        {"price": "110"},
        {"price": "110"},  # Extra to keep it running if needed
    ]
    mock_broker.place_order = MagicMock(return_value={"id": "order_123", "status": "filled"})

    mock_registry = MagicMock()
    mock_registry.broker = mock_broker

    # 3. Initialize Bot
    # We pass None for container as we are testing the engine wiring primarily,
    # but we use the standardized init we just fixed.
    bot = TradingBot(config=config, registry=mock_registry)

    # 4. Run Bot for a few cycles
    # We can't use bot.run() easily because it loops forever.
    # We'll manually cycle the engine a few times.

    # Cycle 1: Price 100 (History: [100])
    await bot.engine._cycle()
    # Cycle 2: Price 100 (History: [100, 100])
    await bot.engine._cycle()
    # Cycle 3: Price 100 (History: [100, 100, 100])
    await bot.engine._cycle()

    # Cycle 4: Price 110 (History: [100, 100, 100, 110])
    # Avg of prev (100, 100, 100) is 100. Current 110 > 100 -> BUY.
    await bot.engine._cycle()

    # 5. Verify Order Placement
    mock_broker.place_order.assert_called_once()
    call_args = mock_broker.place_order.call_args
    assert call_args[0][0]["product_id"] == "BTC-USD"
    assert call_args[0][0]["side"] == "BUY"

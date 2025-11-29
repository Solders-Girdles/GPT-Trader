from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision
from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.orchestration.trading_bot.bot import TradingBot


@pytest.mark.asyncio
async def test_end_to_end_buy_execution():
    """Test that the bot correctly executes a BUY when strategy signals.

    This tests the wiring between TradingBot -> TradingEngine -> Broker,
    not the strategy logic itself. We mock the strategy decision to isolate
    the execution path.
    """
    # 1. Setup Config
    config = BotConfig(
        symbols=["BTC-USD"],
        interval=0.01,  # Fast interval for testing
    )

    # 2. Mock Broker & Registry
    mock_broker = MagicMock()
    mock_broker.get_ticker.return_value = {"price": "100"}

    # Mock balances to provide collateral for equity calculation
    mock_balance = MagicMock()
    mock_balance.asset = "USD"
    mock_balance.available = 10000  # $10,000 collateral
    mock_broker.list_balances.return_value = [mock_balance]

    # Mock positions (empty initially)
    mock_broker.list_positions.return_value = []

    mock_broker.place_order = MagicMock(return_value={"id": "order_123", "status": "filled"})

    mock_registry = MagicMock()
    mock_registry.broker = mock_broker

    # 3. Initialize Bot
    bot = TradingBot(config=config, registry=mock_registry)

    # 4. Mock strategy to return BUY decision on first cycle
    # This isolates testing of the execution wiring from the strategy logic
    buy_decision = Decision(action=Action.BUY, reason="Test signal", confidence=0.8)
    with patch.object(bot.engine.strategy, "decide", return_value=buy_decision):
        await bot.engine._cycle()

    # 5. Verify Order Placement
    mock_broker.place_order.assert_called_once()
    call_args = mock_broker.place_order.call_args
    # place_order is called with positional args: (symbol, side, order_type, quantity)
    assert call_args[0][0] == "BTC-USD"  # symbol
    from gpt_trader.features.brokerages.core.interfaces import OrderSide

    assert call_args[0][1] == OrderSide.BUY  # side

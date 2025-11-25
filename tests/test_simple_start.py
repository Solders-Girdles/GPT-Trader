from unittest.mock import MagicMock

import pytest

from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.orchestration.trading_bot.bot import TradingBot


@pytest.mark.asyncio
async def test_bot_startup_shutdown(monkeypatch):
    config = BotConfig(symbols=["BTC-USD"], dry_run=True, interval=0.1)  # Fast interval

    # Mock Broker
    mock_broker = MagicMock()
    mock_broker.get_ticker.return_value = {"price": "50000"}

    # Mock Registry
    mock_registry = MagicMock()
    mock_registry.broker = mock_broker

    bot = TradingBot(config, registry=mock_registry)

    # Run a single cycle
    await bot.run(single_cycle=True)

    assert bot.running is False
    # Verify ticker was called
    mock_broker.get_ticker.assert_called_with("BTC-USD")

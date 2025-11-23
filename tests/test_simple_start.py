import asyncio
import pytest
from unittest.mock import MagicMock
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.perps_bot.bot import PerpsBot

@pytest.mark.asyncio
async def test_bot_startup_shutdown(monkeypatch):
    config = BotConfig(
        symbols=["BTC-USD"],
        dry_run=True,
        interval=0.1  # Fast interval
    )

    # Mock CoinbaseClient
    mock_client = MagicMock()
    mock_client.get_ticker.return_value = {"price": "50000"}

    # Monkeypatch the client instantiation in TradingEngine
    # Since TradingEngine instantiates it in __init__, we need to patch before PerpsBot init
    # Or patch the class

    with pytest.MonkeyPatch.context() as m:
        m.setattr("bot_v2.orchestration.engines.strategy.CoinbaseClient", lambda **kwargs: mock_client)

        bot = PerpsBot(config)

        # Run a single cycle
        await bot.run(single_cycle=True)

        assert bot.running is False
        # Verify ticker was called
        mock_client.get_ticker.assert_called_with("BTC-USD")

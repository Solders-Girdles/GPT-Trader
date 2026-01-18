from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import clear_application_container, set_application_container
from gpt_trader.features.live_trade.bot import TradingBot


@pytest.mark.asyncio
async def test_bot_startup_shutdown(monkeypatch):
    config = BotConfig(symbols=["BTC-USD"], dry_run=True, interval=0.1)  # Fast interval

    mock_broker = MagicMock()
    mock_broker.get_ticker.return_value = {"price": "50000"}
    mock_broker.list_balances.return_value = []
    mock_broker.list_orders.return_value = {"orders": []}

    mock_container = MagicMock()
    mock_container.broker = mock_broker
    mock_container.risk_manager = MagicMock()
    mock_container.risk_manager._start_of_day_equity = Decimal("1000.0")
    mock_container.risk_manager.config = None
    mock_container.event_store = MagicMock()
    mock_container.notification_service = MagicMock()
    mock_container.account_telemetry = None
    mock_container.runtime_state = None

    bot = None
    with patch("gpt_trader.features.live_trade.engines.strategy.StatusReporter") as MockReporter:
        mock_reporter_instance = MockReporter.return_value
        mock_reporter_instance.start = AsyncMock(return_value=None)
        mock_reporter_instance.stop = AsyncMock(return_value=None)

        mock_container.validation_failure_tracker = MagicMock()
        set_application_container(mock_container)
        try:
            bot = TradingBot(config, container=mock_container)
            await bot.run(single_cycle=True)
        finally:
            clear_application_container()

    assert bot.running is False
    mock_broker.get_ticker.assert_called_with("BTC-USD")

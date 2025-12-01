from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.orchestration.trading_bot.bot import TradingBot


@pytest.mark.asyncio
async def test_bot_startup_shutdown(monkeypatch):
    config = BotConfig(symbols=["BTC-USD"], dry_run=True, interval=0.1)  # Fast interval

    # Mock Broker
    mock_broker = MagicMock()
    mock_broker.get_ticker.return_value = {"price": "50000"}
    # Ensure list_balances returns a list (serializable) not a MagicMock
    mock_broker.list_balances.return_value = []
    # Ensure list_orders returns dict with list
    mock_broker.list_orders.return_value = {"orders": []}

    # Mock Registry
    mock_registry = MagicMock()
    mock_registry.broker = mock_broker
    mock_registry.risk_manager._start_of_day_equity = Decimal("1000.0")
    mock_registry.risk_manager.config = None  # Ensure no config mocks
    # Ensure no mocks leak into JSON serialization
    mock_registry.account_telemetry = None
    mock_registry.runtime_state = None

    # Mock Container
    mock_container = MagicMock()
    mock_container.create_service_registry.return_value = mock_registry

    # Mock StatusReporter to avoid JSON serialization issues with mocks
    with patch("gpt_trader.features.live_trade.engines.strategy.StatusReporter") as MockReporter:
        # Configure the mock instance returned by constructor
        mock_reporter_instance = MockReporter.return_value
        # Ensure async methods are awaited properly
        mock_reporter_instance.start = AsyncMock(return_value=None)
        mock_reporter_instance.stop = AsyncMock(return_value=None)

        bot = TradingBot(config, container=mock_container)

        # Run a single cycle
        await bot.run(single_cycle=True)

    assert bot.running is False
    # Verify ticker was called
    mock_broker.get_ticker.assert_called_with("BTC-USD")

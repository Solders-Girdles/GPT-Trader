"""
System integration test for Ensemble Strategy execution.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.features.live_trade.bot import TradingBot
from gpt_trader.features.live_trade.strategies.ensemble import EnsembleStrategy


class TestEnsembleSystemExecution:
    @pytest.mark.asyncio
    async def test_ensemble_initialization_and_execution(self):
        """Test that TradingBot initializes EnsembleStrategy and runs a cycle."""

        # 1. Configure Bot for Ensemble Strategy
        config = BotConfig(
            strategy_type="ensemble",
            symbols=["BTC-USD"],
            mock_broker=True,
            dry_run=True,
            ensemble_config={"buy_threshold": 0.1, "combiner_config": {"adx_period": 14}},
        )

        # 2. Mock Container with services
        mock_container = MagicMock()

        # Mock Broker
        mock_broker = MagicMock()
        mock_broker.get_ticker.return_value = {"price": "50000.00"}
        mock_broker.list_orders.return_value = {"orders": []}
        mock_broker.get_candles.return_value = []  # For ADX

        # Set container attributes directly (no registry)
        mock_container.broker = mock_broker
        mock_container.account_manager = MagicMock()
        mock_container.risk_manager = MagicMock()
        # Set start_of_day_equity to avoid TypeError in TradingEngine._cycle
        mock_container.risk_manager._start_of_day_equity = Decimal("1000.00")
        mock_container.event_store = MagicMock()
        mock_container.notification_service = MagicMock()

        # 3. Initialize Bot
        bot = TradingBot(config, mock_container)

        # Verify Strategy Type
        assert isinstance(bot.engine.strategy, EnsembleStrategy)
        assert len(bot.engine.strategy.signals) == 3

        # 4. Run One Cycle
        # We call _cycle directly to avoid infinite loop
        await bot.engine._cycle()

        # 5. Verify Execution
        # Check that get_ticker was called
        mock_broker.get_ticker.assert_called_with("BTC-USD")

        # Check that get_candles was called
        mock_broker.get_candles.assert_called_with("BTC-USD", granularity="ONE_MINUTE")

        # Check that strategy.decide was called
        with patch.object(
            bot.engine.strategy, "decide", wraps=bot.engine.strategy.decide
        ) as mock_decide:
            await bot.engine._cycle()
            mock_decide.assert_called()

            # Check decision
            call_args = mock_decide.call_args
            # args: symbol, current_mark, position_state, recent_marks, equity, product, candles
            assert call_args.kwargs["symbol"] == "BTC-USD"
            assert call_args.kwargs["current_mark"] == Decimal("50000.00")
            assert call_args.kwargs["candles"] == []  # We mocked it to return empty list

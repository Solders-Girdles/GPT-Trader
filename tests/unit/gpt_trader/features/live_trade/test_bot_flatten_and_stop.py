"""Tests for the TradingBot emergency flatten-and-stop flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from gpt_trader.features.live_trade.bot import TradingBot


class TestTradingBotFlattenAndStop:
    """Tests for the emergency flatten-and-stop flow."""

    @pytest.mark.asyncio
    async def test_flatten_and_stop_closes_positions_and_shuts_down(self) -> None:
        from decimal import Decimal
        from types import SimpleNamespace
        from unittest.mock import Mock

        from gpt_trader.app.config import BotConfig
        from gpt_trader.core import OrderSide, OrderType

        class _DirectBrokerCalls:
            async def __call__(self, fn, *args, **kwargs):
                return fn(*args, **kwargs)

        config = BotConfig(symbols=["BTC-USD"], interval=1)
        broker = Mock()
        broker.list_positions.return_value = [
            SimpleNamespace(symbol="BTC-USD", quantity=Decimal("1"))
        ]
        broker.place_order = Mock()

        container = SimpleNamespace(
            broker=broker,
            risk_manager=Mock(),
            event_store=Mock(),
            orders_store=Mock(),
            notification_service=Mock(),
        )

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            engine = AsyncMock()
            engine.shutdown = AsyncMock()
            mock_engine.return_value = engine
            bot = TradingBot(config=config, container=container)

        bot._broker_calls = _DirectBrokerCalls()

        messages = await bot.flatten_and_stop()

        assert any("Submitted CLOSE for BTC-USD" in msg for msg in messages)
        broker.list_positions.assert_called_once()
        broker.place_order.assert_called_once_with(
            "BTC-USD",
            OrderSide.SELL,
            OrderType.MARKET,
            Decimal("1"),
        )
        engine.shutdown.assert_called_once()
        assert bot.running is False

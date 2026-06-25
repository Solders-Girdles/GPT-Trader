"""Tests for the TradingBot emergency flatten-and-stop flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import gpt_trader.features.live_trade.bot as bot_module
from gpt_trader.features.live_trade.bot import TradingBot


class TestTradingBotFlattenAndStop:
    """Tests for the emergency flatten-and-stop flow."""

    @pytest.mark.asyncio
    async def test_flatten_and_stop_closes_positions_and_shuts_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

        engine = AsyncMock()
        engine.shutdown = AsyncMock()
        mock_engine = MagicMock(return_value=engine)
        monkeypatch.setattr(bot_module, "TradingEngine", mock_engine)

        bot = TradingBot(config=config, container=container)
        bot._broker_calls = _DirectBrokerCalls()

        messages = await bot.flatten_and_stop()

        assert any("Submitted CLOSE for BTC-USD" in msg for msg in messages)
        broker.list_positions.assert_called_once()
        broker.place_order.assert_called_once_with(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
            reduce_only=True,
        )
        engine.shutdown.assert_called_once()
        assert bot.running is False

    @pytest.mark.asyncio
    async def test_flatten_and_stop_uses_reduce_only_buy_for_short_positions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
            SimpleNamespace(symbol="BTC-USD", quantity=Decimal("-2"))
        ]
        broker.place_order = Mock()

        container = SimpleNamespace(
            broker=broker,
            risk_manager=Mock(),
            event_store=Mock(),
            orders_store=Mock(),
            notification_service=Mock(),
        )

        engine = AsyncMock()
        engine.shutdown = AsyncMock()
        mock_engine = MagicMock(return_value=engine)
        monkeypatch.setattr(bot_module, "TradingEngine", mock_engine)

        bot = TradingBot(config=config, container=container)
        bot._broker_calls = _DirectBrokerCalls()

        messages = await bot.flatten_and_stop()

        assert any("Submitted CLOSE for BTC-USD" in msg for msg in messages)
        broker.place_order.assert_called_once_with(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("2"),
            reduce_only=True,
        )
        engine.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_flatten_and_stop_refuses_non_reduce_only_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from decimal import Decimal
        from types import SimpleNamespace
        from unittest.mock import Mock

        from gpt_trader.app.config import BotConfig

        class _DirectBrokerCalls:
            async def __call__(self, fn, *args, **kwargs):
                return fn(*args, **kwargs)

        def reject_reduce_only(**kwargs):
            assert kwargs["reduce_only"] is True
            raise TypeError("unexpected keyword argument 'reduce_only'")

        config = BotConfig(symbols=["BTC-USD"], interval=1)
        broker = Mock()
        broker.list_positions.return_value = [
            SimpleNamespace(symbol="BTC-USD", quantity=Decimal("1"))
        ]
        broker.place_order = Mock(side_effect=reject_reduce_only)

        container = SimpleNamespace(
            broker=broker,
            risk_manager=Mock(),
            event_store=Mock(),
            orders_store=Mock(),
            notification_service=Mock(),
        )

        engine = AsyncMock()
        engine.shutdown = AsyncMock()
        mock_engine = MagicMock(return_value=engine)
        monkeypatch.setattr(bot_module, "TradingEngine", mock_engine)

        bot = TradingBot(config=config, container=container)
        bot._broker_calls = _DirectBrokerCalls()

        messages = await bot.flatten_and_stop()

        assert any("refusing non-reduce-only fallback" in msg for msg in messages)
        broker.place_order.assert_called_once()
        assert broker.place_order.call_args.kwargs["reduce_only"] is True
        engine.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_flatten_and_stop_preserves_unrelated_type_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from decimal import Decimal
        from types import SimpleNamespace
        from unittest.mock import Mock

        from gpt_trader.app.config import BotConfig

        class _DirectBrokerCalls:
            async def __call__(self, fn, *args, **kwargs):
                return fn(*args, **kwargs)

        def fail_for_unrelated_payload_bug(**kwargs):
            assert kwargs["reduce_only"] is True
            raise TypeError("quantity payload must be Decimal")

        config = BotConfig(symbols=["BTC-USD"], interval=1)
        broker = Mock()
        broker.list_positions.return_value = [
            SimpleNamespace(symbol="BTC-USD", quantity=Decimal("1"))
        ]
        broker.place_order = Mock(side_effect=fail_for_unrelated_payload_bug)

        container = SimpleNamespace(
            broker=broker,
            risk_manager=Mock(),
            event_store=Mock(),
            orders_store=Mock(),
            notification_service=Mock(),
        )

        engine = AsyncMock()
        engine.shutdown = AsyncMock()
        mock_engine = MagicMock(return_value=engine)
        monkeypatch.setattr(bot_module, "TradingEngine", mock_engine)

        bot = TradingBot(config=config, container=container)
        bot._broker_calls = _DirectBrokerCalls()

        messages = await bot.flatten_and_stop()

        assert any("quantity payload must be Decimal" in msg for msg in messages)
        assert not any("refusing non-reduce-only fallback" in msg for msg in messages)
        broker.place_order.assert_called_once()
        assert broker.place_order.call_args.kwargs["reduce_only"] is True
        engine.shutdown.assert_called_once()

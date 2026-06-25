"""Tests for the TradingBot emergency flatten-and-stop flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import gpt_trader.features.live_trade.bot as bot_module
from gpt_trader.features.live_trade.bot import TradingBot
from gpt_trader.features.live_trade.lifecycle import TradingBotState
from gpt_trader.monitoring.alert_types import AlertSeverity


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
        assert messages[-1] == "Bot stopped."
        broker.list_positions.assert_called_once()
        broker.place_order.assert_called_once_with(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
            reduce_only=True,
        )
        engine.shutdown.assert_called_once()
        assert bot.state is TradingBotState.STOPPED
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
    async def test_flatten_and_stop_uses_position_side_for_abs_quantity_shorts(
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
            SimpleNamespace(symbol="BTC-USD", quantity=Decimal("2"), side="short")
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

    @pytest.mark.asyncio
    async def test_flatten_and_stop_partial_close_failure_keeps_alerting_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from decimal import Decimal
        from types import SimpleNamespace
        from unittest.mock import Mock

        from gpt_trader.app.config import BotConfig

        class _DirectBrokerCalls:
            def __init__(self) -> None:
                self.shutdown = Mock()

            async def __call__(self, fn, *args, **kwargs):
                return fn(*args, **kwargs)

        def place_order(**kwargs):
            if kwargs["symbol"] == "ETH-USD":
                raise RuntimeError("venue rejected emergency close")
            return None

        config = BotConfig(symbols=["BTC-USD", "ETH-USD"], interval=1)
        broker = Mock()
        broker.list_positions.return_value = [
            SimpleNamespace(symbol="BTC-USD", quantity=Decimal("1")),
            SimpleNamespace(symbol="ETH-USD", quantity=Decimal("-2")),
        ]
        broker.place_order = Mock(side_effect=place_order)
        event_store = Mock()
        notification_service = Mock()
        notification_service.notify = AsyncMock(return_value=True)

        container = SimpleNamespace(
            broker=broker,
            risk_manager=Mock(),
            event_store=event_store,
            orders_store=Mock(),
            notification_service=notification_service,
        )

        engine = AsyncMock()
        engine.shutdown = AsyncMock()
        mock_engine = MagicMock(return_value=engine)
        monkeypatch.setattr(bot_module, "TradingEngine", mock_engine)

        bot = TradingBot(config=config, container=container)
        bot._broker_calls = _DirectBrokerCalls()

        messages = await bot.flatten_and_stop()

        assert any("Submitted CLOSE for BTC-USD" in msg for msg in messages)
        assert any("Failed to close ETH-USD" in msg for msg in messages)
        assert any("Emergency flatten incomplete" in msg for msg in messages)
        assert bot.state is TradingBotState.ERROR
        assert bot.running is False
        engine.shutdown.assert_called_once()
        bot._broker_calls.shutdown.assert_not_called()
        notification_service.notify.assert_awaited_once()
        notify_kwargs = notification_service.notify.await_args.kwargs
        assert notify_kwargs["severity"] is AlertSeverity.CRITICAL
        assert notify_kwargs["force"] is True
        assert notify_kwargs["context"]["failed_symbols"] == ["ETH-USD"]
        event_store.append.assert_called_once()
        event_type, payload = event_store.append.call_args.args
        assert event_type == "emergency_flatten_failed"
        assert payload["monitoring_state"] == "alerting_active_until_reconciliation"
        assert payload["failed_closes"] == [
            {
                "symbol": "ETH-USD",
                "quantity": "2",
                "error": "venue rejected emergency close",
            }
        ]

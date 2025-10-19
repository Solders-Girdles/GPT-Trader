from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderStatus, OrderType
from bot_v2.features.live_trade.risk import ValidationError
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.perps_bot_builder import create_perps_bot


def test_market_data_actually_updates():
    bot = create_perps_bot(BotConfig.from_profile("dev"))

    # Initialize marks
    asyncio.run(bot.update_marks())
    initial = {s: list(bot.mark_windows.get(s, [])) for s in bot.config.symbols}

    # Force a new mark and update
    sym = bot.config.symbols[0]
    bot.broker.set_mark(sym, Decimal("50100"))  # type: ignore[attr-defined]
    asyncio.run(bot.update_marks())

    # Marks should have grown and last changed for the forced symbol
    for s in bot.config.symbols:
        assert len(bot.mark_windows[s]) >= len(initial.get(s, []))
    assert bot.mark_windows[sym][-1] != (initial.get(sym, [Decimal("0")])[-1])


def test_order_actually_places():
    bot = create_perps_bot(BotConfig.from_profile("dev"))

    order_id = bot.exec_engine.place_order(
        symbol=bot.config.symbols[0],
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.001"),
    )
    assert order_id is not None

    order = bot.broker.get_order(order_id)
    assert order.status in [OrderStatus.FILLED, OrderStatus.SUBMITTED]


def test_position_math_is_correct():
    bot = create_perps_bot(BotConfig.from_profile("dev"))
    sym = bot.config.symbols[0]

    # Buy 0.1 at 50,000 (5% exposure under default risk caps)
    bot.broker.set_mark(sym, Decimal("50000"))  # type: ignore[attr-defined]
    order_id = bot.exec_engine.place_order(
        symbol=sym, side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("0.1")
    )
    assert order_id is not None

    # Move mark to 51,000 and revalue
    bot.broker.set_mark(sym, Decimal("51000"))  # type: ignore[attr-defined]
    pos = next((p for p in bot.broker.list_positions() if p.symbol == sym), None)
    assert pos is not None
    pnl = (
        (pos.mark_price - pos.entry_price) * pos.quantity
        if pos.side == "long"
        else (pos.entry_price - pos.mark_price) * pos.quantity
    )
    assert pnl == Decimal("100")


def test_risk_limits_actually_stop_trading():
    bot = create_perps_bot(BotConfig.from_profile("dev"))
    bot.risk_manager.config.reduce_only_mode = True

    with pytest.raises(ValidationError):
        bot.exec_engine.place_order(
            symbol=bot.config.symbols[0],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),
        )

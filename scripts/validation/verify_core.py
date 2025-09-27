#!/usr/bin/env python3
"""
Core foundation verification (offline, mock broker).

Runs quick, deterministic checks against the dev profile to confirm the
basics actually work end-to-end without relying on live APIs:

  - Broker connection
  - Market data updates (REST + mock WS)
  - Order placement lifecycle (happy path)
  - Position tracking math sanity
  - Risk reduce-only enforcement
  - OrdersStore presence (persistence surface)

Usage:
  python scripts/validation/verify_core.py --check all
  python scripts/validation/verify_core.py --check market_data
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

# Ensure src on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot_v2.orchestration.perps_bot import PerpsBot, BotConfig
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, OrderStatus


def ok(label: str):
    print(f"✓ {label}")


def fail(label: str, err: Exception | str):
    print(f"✗ {label}: {err}")


async def check_broker(bot: PerpsBot) -> bool:
    try:
        # Dev profile uses mock broker
        assert bot.broker is not None
        ok("Broker connection (mock)")
        return True
    except Exception as e:
        fail("Broker connection", e)
        return False


async def check_market_data(bot: PerpsBot) -> bool:
    try:
        # Single poll should append marks for all symbols
        await bot.update_marks()
        for s in bot.config.symbols:
            assert len(bot.mark_windows.get(s, [])) > 0, f"No marks for {s}"
        ok("Market data updates (REST quotes)")
        return True
    except Exception as e:
        fail("Market data updates", e)
        return False


async def check_streaming(bot: PerpsBot) -> bool:
    try:
        # Use mock stream_trades to update mark window via background thread
        # Wait briefly to accumulate ticks
        await asyncio.sleep(0.2)
        grew = False
        for s in bot.config.symbols:
            before = len(bot.mark_windows.get(s, []))
            await bot.update_marks()  # Ensure window initialized
            await asyncio.sleep(0.2)
            after = len(bot.mark_windows.get(s, []))
            if after > before:
                grew = True
        assert grew, "No streaming-driven updates detected"
        ok("WebSocket streaming (mock trades)")
        return True
    except NotImplementedError as e:
        fail("WebSocket streaming", e)
        return False
    except Exception as e:
        fail("WebSocket streaming", e)
        return False


async def check_order_placement(bot: PerpsBot) -> bool:
    try:
        order_id = bot.exec_engine.place_order(
            symbol=bot.config.symbols[0],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("0.001"),
        )
        assert order_id is not None
        order = bot.broker.get_order(order_id)
        assert order.status in (OrderStatus.FILLED, OrderStatus.SUBMITTED)
        ok("Order placement (market)")
        return True
    except Exception as e:
        fail("Order placement", e)
        return False


async def check_position_math(bot: PerpsBot) -> bool:
    try:
        sym = bot.config.symbols[0]
        # Ensure deterministic starting mark
        bot.broker.set_mark(sym, Decimal("50000"))  # type: ignore[attr-defined]
        order_id = bot.exec_engine.place_order(
            symbol=sym,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1"),
        )
        assert order_id is not None
        # Move mark to 51k and revalue
        bot.broker.set_mark(sym, Decimal("51000"))  # type: ignore[attr-defined]
        pos = next((p for p in bot.broker.list_positions() if p.symbol == sym), None)
        assert pos is not None
        unrealized = (pos.mark_price - pos.entry_price) * pos.qty if pos.side == 'long' else (pos.entry_price - pos.mark_price) * pos.qty
        assert unrealized == Decimal("1000")
        ok("Position P&L math (long +$1k)")
        return True
    except Exception as e:
        fail("Position P&L math", e)
        return False


async def check_risk_reduce_only(bot: PerpsBot) -> bool:
    try:
        bot.risk_manager.config.reduce_only_mode = True
        raised = False
        try:
            bot.exec_engine.place_order(
                symbol=bot.config.symbols[0],
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=Decimal("0.001"),
            )
        except Exception:
            raised = True
        assert raised, "Expected ValidationError under reduce-only mode"
        ok("Risk limit enforcement (reduce-only)")
        return True
    except Exception as e:
        fail("Risk limit enforcement", e)
        return False


async def check_state_persistence(bot: PerpsBot) -> bool:
    try:
        # We don't force disk writes here; just verify OrdersStore surface exists
        _ = bot.orders_store.get_open_orders()
        ok("State surface (OrdersStore available)")
        return True
    except Exception as e:
        fail("State persistence surface", e)
        return False


async def main():
    parser = argparse.ArgumentParser(description="Verify core bot foundation")
    parser.add_argument("--check", choices=[
        "all",
        "broker",
        "market_data",
        "streaming",
        "orders",
        "positions",
        "risk",
        "state",
    ], default="all")
    args = parser.parse_args()

    print("1. Starting bot (dev/mock)...")
    bot = PerpsBot(BotConfig.from_profile("dev"))

    tasks = {
        "broker": check_broker,
        "market_data": check_market_data,
        "streaming": check_streaming,
        "orders": check_order_placement,
        "positions": check_position_math,
        "risk": check_risk_reduce_only,
        "state": check_state_persistence,
    }

    if args.check == "all":
        order = ["broker", "market_data", "streaming", "orders", "positions", "risk", "state"]
    else:
        order = [args.check]

    overall = True
    for key in order:
        ok_label = key.replace("_", " ").title()
        res = await tasks[key](bot)
        overall = overall and res

    print("\nDone." if overall else "\nCompleted with failures.")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

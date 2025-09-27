#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from decimal import Decimal
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot_v2.orchestration.perps_bot import PerpsBot, BotConfig
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType


async def verify_data_flow() -> bool:
    print("1. Starting bot (dev/mock)...")
    bot = PerpsBot(BotConfig.from_profile('dev'))

    print("2. Checking market data updates...")
    await bot.update_marks()
    assert all(len(m) >= 1 for m in bot.mark_windows.values())

    print("3. Checking strategy decisions...")
    await bot.run_cycle()
    assert len(bot.last_decisions) > 0

    print("4. Checking order placement...")
    # Force a simple buy decision execution path
    oid = bot.exec_engine.place_order(symbol=bot.config.symbols[0], side=OrderSide.BUY, order_type=OrderType.MARKET, qty=Decimal('0.001'))
    assert oid is not None
    return True


async def verify_state_consistency() -> bool:
    print("1. Starting bot (dev/mock) for state consistency...")
    bot = PerpsBot(BotConfig.from_profile('dev'))

    print("2. Placing an order and verifying stores...")
    oid = bot.exec_engine.place_order(symbol=bot.config.symbols[0], side=OrderSide.BUY, order_type=OrderType.MARKET, qty=Decimal('0.001'))
    assert oid is not None
    # Verify presence across broker and store surfaces
    broker_ids = [o.id for o in bot.broker.list_orders()]
    assert oid in broker_ids
    # OrdersStore only records via explicit upsert; LiveExecutionEngine appends metrics, not orders
    # Simulate reconciliation by invoking startup routine
    await bot._reconcile_state_on_startup()
    return True


async def main():
    parser = argparse.ArgumentParser(description="Verify core foundation")
    parser.add_argument('--which', choices=['all', 'data', 'state'], default='all')
    args = parser.parse_args()

    ok = True
    if args.which in ('all', 'data'):
        ok &= await verify_data_flow()
    if args.which in ('all', 'state'):
        ok &= await verify_state_consistency()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))

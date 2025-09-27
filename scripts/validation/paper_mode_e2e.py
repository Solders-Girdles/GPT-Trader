#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import logging
import os

from bot_v2.orchestration.perps_bot import PerpsBot, BotConfig


async def main() -> int:
    # Ensure paper mode and clean, minimal env
    os.environ.setdefault('PERPS_PAPER', '1')
    os.environ.setdefault('BROKER', 'coinbase')
    os.environ.setdefault('LOG_LEVEL', 'INFO')

    logging.basicConfig(level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
                        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')

    # Fast signals for demo: short MAs and quick interval
    cfg = BotConfig.from_profile(
        'dev',
        symbols=['BTC-PERP'],
        update_interval=1,
        short_ma=3,
        long_ma=5,
    )
    bot = PerpsBot(config=cfg)

    # Run several single cycles; drive marks if using MockBroker to trigger signals
    pattern = [100, 99, 101, 103, 105, 104, 106, 108]
    for i, px in enumerate(pattern):
        try:
            if hasattr(bot.broker, 'set_mark'):
                bot.broker.set_mark('BTC-PERP', px)  # type: ignore[attr-defined]
        except Exception:
            pass
        await bot.run(single_cycle=True)

    # Print a brief summary
    print('\n=== Paper Mode E2E Summary ===')
    print(f"Symbols: {cfg.symbols}")
    print(f"Order stats: {bot.order_stats}")
    try:
        positions = bot.broker.list_positions()
        print(f"Positions count: {len(positions)}")
        for p in positions:
            print(f" - {p.symbol}: {p.side} {p.qty} @ {p.entry_price} (mark {p.mark_price})")
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))

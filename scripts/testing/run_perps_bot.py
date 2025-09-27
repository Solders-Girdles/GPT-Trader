#!/usr/bin/env python3
"""
Perpetuals Trading Bot runner (thin wrapper around PerpsBot).

Delegates to src/bot_v2/orchestration/perps_bot.PerpsBot for execution.
Usage:
  python scripts/run_perps_bot.py --profile dev --dev-fast
  python scripts/run_perps_bot.py --profile prod --symbols BTC-PERP ETH-PERP
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, time

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot_v2.orchestration.perps_bot import PerpsBot, BotConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perpetuals Trading Bot (PerpsBot)")
    parser.add_argument('--profile', choices=['dev', 'demo', 'prod', 'canary'], default='dev', help='Configuration profile')
    parser.add_argument('--dry-run', action='store_true', help='Run without placing real orders')
    parser.add_argument('--symbols', nargs='+', help='Symbols to trade (e.g., BTC-PERP ETH-PERP)')
    parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    parser.add_argument('--leverage', type=int, default=2, help='Target leverage')
    parser.add_argument('--reduce-only', action='store_true', help='Enable reduce-only mode')
    parser.add_argument('--dev-fast', action='store_true', help='Run single cycle and exit (for smoke tests)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    overrides: Dict[str, Any] = {}
    if args.dry_run:
        overrides['dry_run'] = True
    if args.symbols:
        overrides['symbols'] = args.symbols
    if args.interval is not None:
        overrides['update_interval'] = args.interval
    if args.leverage is not None:
        overrides['target_leverage'] = args.leverage
    if args.reduce_only:
        overrides['reduce_only_mode'] = True

    config = BotConfig.from_profile(args.profile, **overrides)

    logger.info("Launching PerpsBot")
    logger.info(f"  Profile: {config.profile.value}")
    logger.info(f"  Symbols: {config.symbols}")
    logger.info(f"  Interval: {config.update_interval}s | Leverage: {config.target_leverage}x")
    logger.info(f"  Reduce-only: {config.reduce_only_mode} | Dry-run: {config.dry_run}")

    bot = PerpsBot(config)
    try:
        asyncio.run(bot.run(single_cycle=args.dev_fast))
        return 0
    except KeyboardInterrupt:
        logger.info("Shutdown complete.")
        return 0
    except Exception as e:
        logger.error(f"PerpsBot exited with error: {e}")
        return 1


# Minimal TradingBot wrapper for trading window checks (used by tests)
class TradingBot:
    def __init__(self, config: BotConfig):
        self.config = config

    def is_within_trading_window(self) -> bool:
        now = datetime.now()
        current_time = now.time()
        current_day = now.strftime('%A').lower()

        # Validate days
        days: List[str] = self.config.trading_days or ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        if current_day not in days:
            return False

        start: Optional[time] = self.config.trading_window_start
        end: Optional[time] = self.config.trading_window_end
        if start and end:
            return start <= current_time <= end
        return True


if __name__ == "__main__":
    sys.exit(main())

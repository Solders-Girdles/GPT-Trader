"""
Thin facade that re-exports the production BotConfig and bootstrap helpers.

Tests and smoke scripts import from here to avoid reaching into internal
modules while still constructing the modern PerpsBot wiring.
"""

from __future__ import annotations

from datetime import datetime
import sys

from bot_v2.orchestration.bootstrap import build_bot, bot_from_profile
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.cli import main as _perps_main


class TradingBot:
    """Helper used by legacy tests to reason about trading windows."""

    def __init__(self, config: BotConfig):
        self.config = config

    # Extracted for testability; allows monkeypatching in tests
    def _now(self) -> datetime:
        return datetime.now()

    def is_within_trading_window(self) -> bool:
        start = self.config.trading_window_start
        end = self.config.trading_window_end
        days = self.config.trading_days

        if not start or not end or not days:
            return True

        now = self._now()
        if now.strftime("%A").lower() not in [d.lower() for d in days]:
            return False
        cur_t = now.time()
        return start <= cur_t <= end


__all__ = [
    "BotConfig",
    "TradingBot",
    "Profile",
    "build_bot",
    "bot_from_profile",
]


def main() -> int:
    """Delegate to the canonical CLI entry point."""

    return _perps_main()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())

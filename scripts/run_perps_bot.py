"""
Lightweight runner and config facade for Perps Bot canary/testing.

Provides a BotConfig wrapper compatible with integration tests and a
TradingBot helper exposing is_within_trading_window().
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
from typing import List, Optional

from bot_v2.orchestration.perps_bot import BotConfig as _CoreBotConfig, Profile


@dataclass
class BotConfig:
    """Public BotConfig wrapper used by tests and scripts.

    - Accepts string profile for convenience
    - Exposes a subset of fields used by tests
    """

    profile: str
    dry_run: bool = False
    symbols: Optional[List[str]] = None
    update_interval: int = 5
    short_ma: int = 5
    long_ma: int = 20
    target_leverage: int = 2
    trailing_stop_pct: float = 0.01
    enable_shorts: bool = False
    max_position_size: Decimal = Decimal("1000")
    max_leverage: int = 3
    reduce_only_mode: bool = False
    mock_broker: bool = False
    mock_fills: bool = False
    # Trading session controls
    trading_window_start: Optional[time] = None
    trading_window_end: Optional[time] = None
    trading_days: Optional[List[str]] = None
    # Risk extras
    daily_loss_limit: Decimal = Decimal("0")
    time_in_force: str = "GTC"

    @classmethod
    def from_profile(cls, profile: str, **overrides) -> "BotConfig":
        core = _CoreBotConfig.from_profile(profile, **overrides)
        # Create a light shim to mimic enum-like access (profile.value)
        ProfileShim = type('ProfileShim', (), {})
        profile_obj = ProfileShim()
        setattr(profile_obj, 'value', core.profile.value)
        return cls(
            profile=profile_obj,  # type: ignore[assignment]
            dry_run=core.dry_run,
            symbols=core.symbols,
            update_interval=core.update_interval,
            short_ma=core.short_ma,
            long_ma=core.long_ma,
            target_leverage=core.target_leverage,
            trailing_stop_pct=core.trailing_stop_pct,
            enable_shorts=core.enable_shorts,
            max_position_size=core.max_position_size,
            max_leverage=core.max_leverage,
            reduce_only_mode=core.reduce_only_mode,
            mock_broker=core.mock_broker,
            mock_fills=core.mock_fills,
            trading_window_start=core.trading_window_start,
            trading_window_end=core.trading_window_end,
            trading_days=core.trading_days,
            daily_loss_limit=core.daily_loss_limit,
            time_in_force=core.time_in_force,
        )


class TradingBot:
    def __init__(self, config: BotConfig):
        self.config = config

    # Extracted for testability; allows monkeypatching in tests
    def _now(self) -> datetime:
        return datetime.now()

    def is_within_trading_window(self) -> bool:
        # If no explicit controls, allow trading
        if not self.config.trading_window_start or not self.config.trading_window_end or not self.config.trading_days:
            return True

        now = self._now()
        day = now.strftime('%A').lower()
        if day not in [d.lower() for d in self.config.trading_days or []]:
            return False
        cur_t = now.time()
        return self.config.trading_window_start <= cur_t <= self.config.trading_window_end


# Convenience re-exports
__all__ = ["BotConfig", "TradingBot", "Profile"]

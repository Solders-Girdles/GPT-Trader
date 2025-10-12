"""Configuration objects for the baseline perpetuals strategy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass
class StrategyConfig:
    """Configuration for the baseline strategy core logic."""

    # MA parameters
    short_ma_period: int = 5
    long_ma_period: int = 20

    # Position management
    target_leverage: int = 2
    trailing_stop_pct: float = 0.01  # 1% trailing stop (fixed)
    # Simple, predictable sizing: percentage of equity per trade, with optional USD cap
    position_fraction: float = 0.05  # 5% of equity
    max_trade_usd: Decimal | None = None  # cap notional if set

    # Feature flags
    enable_shorts: bool = False
    max_adds: int = 0  # Disable pyramiding by default
    disable_new_entries: bool = False
    # Advanced entries (deprecated/no-op in simplified baseline)
    use_stop_entry: bool = False
    use_post_only: bool = False
    prefer_maker_orders: bool = False

    # Funding-awareness (deprecated for baseline; retained for backward compat, not used)
    funding_bias_bps: float = 0.0
    funding_block_long_bps: float = 0.0
    funding_block_short_bps: float = 0.0

    # Crossover robustness (optional)
    # Epsilon tolerance in basis points for crossover detection (0 = strict)
    ma_cross_epsilon_bps: Decimal = Decimal("0")
    # Bars to confirm crossover persistence (0 = no confirmation)
    ma_cross_confirm_bars: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


__all__ = ["StrategyConfig"]

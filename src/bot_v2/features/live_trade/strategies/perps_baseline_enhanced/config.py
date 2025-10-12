"""Configuration models for the enhanced baseline strategy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from bot_v2.features.strategy_tools import MarketConditionFilters, RiskGuards


@dataclass
class StrategyFiltersConfig:
    """Configuration for market condition filters and risk guards."""

    # Market condition filters
    max_spread_bps: Decimal | None = Decimal("10")
    min_depth_l1: Decimal | None = Decimal("50000")
    min_depth_l10: Decimal | None = Decimal("200000")
    min_volume_1m: Decimal | None = Decimal("100000")
    require_rsi_confirmation: bool = True

    # Risk guards
    min_liquidation_buffer_pct: Decimal | None = Decimal("20")
    max_slippage_impact_bps: Decimal | None = Decimal("15")

    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: Decimal = Decimal("30")
    rsi_overbought: Decimal = Decimal("70")

    def create_filters(self) -> MarketConditionFilters:
        """Create market condition filters from config."""
        return MarketConditionFilters(
            max_spread_bps=self.max_spread_bps,
            min_depth_l1=self.min_depth_l1,
            min_depth_l10=self.min_depth_l10,
            min_volume_1m=self.min_volume_1m,
            min_volume_5m=None,
            rsi_oversold=self.rsi_oversold,
            rsi_overbought=self.rsi_overbought,
            require_rsi_confirmation=self.require_rsi_confirmation,
        )

    def create_guards(self) -> RiskGuards:
        """Create risk guards from config."""
        return RiskGuards(
            min_liquidation_buffer_pct=self.min_liquidation_buffer_pct,
            max_slippage_impact_bps=self.max_slippage_impact_bps,
        )


@dataclass
class StrategyConfig:
    """Enhanced configuration for baseline strategy."""

    # MA parameters
    short_ma_period: int = 5
    long_ma_period: int = 20
    ma_cross_epsilon_bps: Decimal = Decimal("1")
    ma_cross_confirm_bars: int = 0

    # Position management
    target_leverage: int = 2
    trailing_stop_pct: float = 0.01

    # Feature flags
    enable_shorts: bool = False
    max_adds: int = 1
    disable_new_entries: bool = False

    # Filters and guards
    filters_config: StrategyFiltersConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyConfig:
        """Create config from dictionary."""
        data = dict(data)
        filters_data = data.pop("filters_config", None)
        filters_config = (
            StrategyFiltersConfig(**filters_data) if isinstance(filters_data, dict) else None
        )
        return cls(
            **{k: v for k, v in data.items() if k in cls.__annotations__},
            filters_config=filters_config,
        )


__all__ = ["StrategyConfig", "StrategyFiltersConfig"]

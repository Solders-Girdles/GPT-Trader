"""Market condition filter utilities for strategy authors."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass
class MarketConditionFilters:
    """Configurable filters for market conditions."""

    max_spread_bps: Decimal | None = None
    min_depth_l1: Decimal | None = None
    min_depth_l10: Decimal | None = None
    min_volume_1m: Decimal | None = None
    min_volume_5m: Decimal | None = None
    rsi_oversold: Decimal | None = Decimal("30")
    rsi_overbought: Decimal | None = Decimal("70")
    require_rsi_confirmation: bool = False

    def should_allow_long_entry(
        self, market_snapshot: dict[str, Any], rsi: Decimal | None = None
    ) -> tuple[bool, str]:
        if self.max_spread_bps and market_snapshot.get("spread_bps", 0) >= self.max_spread_bps:
            return False, (
                f"Spread too wide: {market_snapshot.get('spread_bps')} >= {self.max_spread_bps} bps"
            )

        if self.min_depth_l1 and market_snapshot.get("depth_l1", 0) <= self.min_depth_l1:
            return False, (
                f"L1 depth insufficient: {market_snapshot.get('depth_l1')} <= {self.min_depth_l1}"
            )

        if self.min_depth_l10 and market_snapshot.get("depth_l10", 0) <= self.min_depth_l10:
            return False, (
                f"L10 depth insufficient: {market_snapshot.get('depth_l10')} <= {self.min_depth_l10}"
            )

        if self.min_volume_1m and market_snapshot.get("vol_1m", 0) <= self.min_volume_1m:
            return False, (
                f"1m volume too low: {market_snapshot.get('vol_1m')} <= {self.min_volume_1m}"
            )

        if self.min_volume_5m and market_snapshot.get("vol_5m", 0) <= self.min_volume_5m:
            return False, (
                f"5m volume too low: {market_snapshot.get('vol_5m')} <= {self.min_volume_5m}"
            )

        if self.require_rsi_confirmation and rsi is not None:
            if rsi > (self.rsi_overbought or Decimal("70")):
                return False, f"RSI too high for long entry: {rsi} > {self.rsi_overbought}"

        return True, "Market conditions acceptable"

    def should_allow_short_entry(
        self, market_snapshot: dict[str, Any], rsi: Decimal | None = None
    ) -> tuple[bool, str]:
        long_ok, reason = self.should_allow_long_entry(market_snapshot, None)
        if not long_ok:
            return False, reason

        if self.require_rsi_confirmation and rsi is not None:
            if rsi < (self.rsi_oversold or Decimal("30")):
                return False, f"RSI too low for short entry: {rsi} < {self.rsi_oversold}"

        return True, "Market conditions acceptable"


def create_conservative_filters() -> MarketConditionFilters:
    """Conservative defaults for risk-averse trading."""

    return MarketConditionFilters(
        max_spread_bps=Decimal("10"),
        min_depth_l1=Decimal("50000"),
        min_depth_l10=Decimal("200000"),
        min_volume_1m=Decimal("100000"),
        require_rsi_confirmation=True,
    )


def create_aggressive_filters() -> MarketConditionFilters:
    """Aggressive defaults for higher-risk trading."""

    return MarketConditionFilters(
        max_spread_bps=Decimal("25"),
        min_depth_l1=Decimal("20000"),
        min_depth_l10=Decimal("100000"),
        min_volume_1m=Decimal("50000"),
        require_rsi_confirmation=False,
    )

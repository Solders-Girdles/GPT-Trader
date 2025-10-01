"""Risk guard helpers for strategy execution."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass
class RiskGuards:
    """Risk management guards for position sizing and safety."""

    min_liquidation_buffer_pct: Decimal | None = Decimal("15")
    max_slippage_impact_bps: Decimal | None = Decimal("20")

    def check_liquidation_distance(
        self,
        entry_price: Decimal,
        position_size: Decimal,
        leverage: Decimal,
        account_equity: Decimal,
        maintenance_margin_rate: Decimal = Decimal("0.05"),
    ) -> tuple[bool, str]:
        if not self.min_liquidation_buffer_pct:
            return True, "Liquidation guard disabled"

        # Handle edge case of zero or negative entry price
        if entry_price <= 0:
            return False, f"Invalid entry price: {entry_price}"

        liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin_rate)
        price_diff = abs(entry_price - liquidation_price)
        distance_pct = (price_diff / entry_price) * 100

        if distance_pct <= self.min_liquidation_buffer_pct:
            return False, (
                f"Too close to liquidation: {distance_pct:.1f}% <= {self.min_liquidation_buffer_pct}%"
            )

        return True, f"Safe liquidation distance: {distance_pct:.1f}%"

    def check_slippage_impact(
        self,
        order_size: Decimal,
        market_snapshot: dict[str, Any],
    ) -> tuple[bool, str]:
        if not self.max_slippage_impact_bps:
            return True, "Slippage guard disabled"

        l1_depth = market_snapshot.get("depth_l1", 0)
        l10_depth = market_snapshot.get("depth_l10", 0)
        if not l1_depth:
            return False, "Insufficient market data for slippage calculation"

        if order_size > l10_depth:
            return False, f"Order too large: {order_size} > L10 depth {l10_depth}"

        if order_size <= l1_depth:
            estimated_impact_bps = (order_size / l1_depth) * Decimal("5")
        else:
            l1_impact = Decimal("5")
            excess = order_size - l1_depth
            excess_depth = l10_depth - l1_depth if l10_depth > l1_depth else l1_depth
            excess_ratio = (
                min(excess / excess_depth, Decimal("1")) if excess_depth else Decimal("1")
            )
            additional_impact = excess_ratio ** Decimal("0.5") * Decimal("20")
            estimated_impact_bps = l1_impact + additional_impact

        if estimated_impact_bps > self.max_slippage_impact_bps:
            return False, (
                f"Estimated slippage too high: {estimated_impact_bps:.1f} > {self.max_slippage_impact_bps} bps"
            )

        return True, f"Acceptable slippage: {estimated_impact_bps:.1f} bps"


def create_standard_risk_guards() -> RiskGuards:
    """Standard risk guard configuration."""

    return RiskGuards(
        min_liquidation_buffer_pct=Decimal("20"),
        max_slippage_impact_bps=Decimal("15"),
    )

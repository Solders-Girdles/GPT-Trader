"""Market impact sizing utilities for the advanced execution engine."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.utilities.logging_patterns import get_logger

from .config import SizingMode

if TYPE_CHECKING:
    from .engine import AdvancedExecutionEngine

logger = get_logger(__name__, component="live_trade_execution")


def calculate_impact_aware_size(
    engine: "AdvancedExecutionEngine",
    symbol: str | None,
    target_notional: Decimal,
    market_snapshot: dict[str, Any],
    max_impact_bps: Decimal | None = None,
) -> tuple[Decimal, Decimal]:
    """Determine a notional size that respects configured impact thresholds."""

    max_impact = max_impact_bps or engine.config.max_impact_bps

    l1_depth = Decimal(str(market_snapshot.get("depth_l1", 0)))
    l10_depth = Decimal(str(market_snapshot.get("depth_l10", 0)))

    if not l1_depth or not l10_depth:
        logger.warning("Insufficient depth data for impact calculation")
        return Decimal("0"), Decimal("0")

    low, high = Decimal("0"), min(target_notional, l10_depth)
    best_size = Decimal("0")
    best_impact = Decimal("0")
    extra_bps = Decimal("0")
    if symbol and symbol in engine.slippage_multipliers:
        try:
            extra_bps = Decimal("10000") * Decimal(str(engine.slippage_multipliers[symbol]))
        except Exception:
            extra_bps = Decimal("0")

    while high - low > Decimal("1"):
        mid = (low + high) / 2
        impact = estimate_impact(mid, l1_depth, l10_depth) + extra_bps

        if impact <= max_impact:
            best_size = mid
            best_impact = impact
            low = mid
        else:
            high = mid

    sizing_mode = engine.config.sizing_mode
    if sizing_mode == SizingMode.STRICT and best_size < target_notional:
        logger.warning(f"Strict mode: Cannot fit {target_notional} within {max_impact} bps impact")
        return Decimal("0"), Decimal("0")
    if sizing_mode == SizingMode.AGGRESSIVE and target_notional <= l10_depth:
        return (
            target_notional,
            estimate_impact(target_notional, l1_depth, l10_depth) + extra_bps,
        )

    if best_size < target_notional:
        logger.info(
            f"SIZED_DOWN: Original=${target_notional:.0f} → Adjusted=${best_size:.0f} "
            f"(Impact: {best_impact:.1f}bps, Limit: {max_impact}bps)"
        )

    return best_size, best_impact


def estimate_impact(order_size: Decimal, l1_depth: Decimal, l10_depth: Decimal) -> Decimal:
    """Estimate market impact in basis points using depth data."""

    if order_size <= l1_depth:
        return (order_size / l1_depth) * Decimal("5")
    if order_size <= l10_depth:
        l1_impact = Decimal("5")
        excess = order_size - l1_depth
        excess_depth = l10_depth - l1_depth if l10_depth > l1_depth else l1_depth
        excess_ratio = min(excess / excess_depth, Decimal("1"))
        additional_impact = excess_ratio ** Decimal("0.5") * Decimal("20")
        return l1_impact + additional_impact
    return Decimal("100")


__all__ = ["calculate_impact_aware_size", "estimate_impact"]

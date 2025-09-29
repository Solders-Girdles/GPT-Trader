"""
Week 2: Liquidation distance monitoring and risk guards.

Computes liquidation distance based on leverage and margin rules,
with configurable risk buffers to force reduce-only mode or reject entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class MarginInfo:
    """Margin and leverage information for position."""

    symbol: str
    position_size: Decimal  # Absolute position size
    position_side: str  # "long" or "short"
    entry_price: Decimal  # Average entry price
    current_price: Decimal  # Current mark price
    leverage: Decimal  # Position leverage (e.g. 3.0)
    maintenance_margin_rate: Decimal = Decimal("0.05")  # 5% default maintenance margin


@dataclass
class LiquidationRisk:
    """Liquidation risk assessment result."""

    symbol: str
    liquidation_price: Decimal | None = None
    distance_pct: float | None = None  # Distance as percentage
    distance_bps: float | None = None  # Distance in basis points
    risk_level: str = "unknown"  # "safe", "warning", "critical"
    should_reduce_only: bool = False
    should_reject_entry: bool = False
    reason: str = ""


class LiquidationMonitor:
    """Monitor liquidation distance and apply risk-based guards."""

    def __init__(
        self,
        warning_buffer_pct: float = 20.0,  # Warning when within 20% of liq
        critical_buffer_pct: float = 15.0,  # Critical when within 15% of liq
        enable_reduce_only_guard: bool = True,
        enable_entry_rejection: bool = True,
    ) -> None:
        self.warning_buffer_pct = warning_buffer_pct
        self.critical_buffer_pct = critical_buffer_pct
        self.enable_reduce_only_guard = enable_reduce_only_guard
        self.enable_entry_rejection = enable_entry_rejection

    def calculate_liquidation_price(self, margin_info: MarginInfo) -> Decimal | None:
        """
        Calculate liquidation price for a perpetual position.

        Simplified model:
        For long: liq_price = entry_price * (1 - 1/leverage + maintenance_margin_rate)
        For short: liq_price = entry_price * (1 + 1/leverage - maintenance_margin_rate)
        """
        try:
            leverage = margin_info.leverage
            entry = margin_info.entry_price
            mm_rate = margin_info.maintenance_margin_rate

            if margin_info.position_side.lower() == "long":
                # Long liquidated when price drops too low
                liq_price = entry * (Decimal("1") - (Decimal("1") / leverage) + mm_rate)
            else:
                # Short liquidated when price rises too high
                liq_price = entry * (Decimal("1") + (Decimal("1") / leverage) - mm_rate)

            return liq_price

        except Exception as e:
            logger.error(f"Failed to calculate liquidation price for {margin_info.symbol}: {e}")
            return None

    def calculate_distance_to_liquidation(
        self, current_price: Decimal, liquidation_price: Decimal, position_side: str
    ) -> tuple[float, float]:
        """
        Calculate distance to liquidation as percentage and basis points.

        Returns:
            (distance_pct, distance_bps)
        """
        if position_side.lower() == "long":
            # Long: distance = (current - liq) / current
            distance = (current_price - liquidation_price) / current_price
        else:
            # Short: distance = (liq - current) / current
            distance = (liquidation_price - current_price) / current_price

        distance_pct = float(distance * 100)
        distance_bps = float(distance * 10000)

        return distance_pct, distance_bps

    def assess_liquidation_risk(self, margin_info: MarginInfo) -> LiquidationRisk:
        """
        Comprehensive liquidation risk assessment.

        Returns:
            LiquidationRisk with distance metrics and risk recommendations
        """
        result = LiquidationRisk(symbol=margin_info.symbol)

        # Skip if no position
        if margin_info.position_size == 0:
            result.risk_level = "safe"
            result.reason = "No position"
            return result

        # Calculate liquidation price
        liq_price = self.calculate_liquidation_price(margin_info)
        if liq_price is None:
            result.risk_level = "unknown"
            result.reason = "Could not calculate liquidation price"
            return result

        result.liquidation_price = liq_price

        # Calculate distance
        distance_pct, distance_bps = self.calculate_distance_to_liquidation(
            margin_info.current_price, liq_price, margin_info.position_side
        )

        result.distance_pct = distance_pct
        result.distance_bps = distance_bps

        # Assess risk level
        if distance_pct <= 0:
            # At or past liquidation
            result.risk_level = "liquidated"
            result.should_reduce_only = True
            result.should_reject_entry = True
            result.reason = f"Position at/past liquidation (distance: {distance_pct:.1f}%)"

        elif distance_pct <= self.critical_buffer_pct:
            # Critical risk
            result.risk_level = "critical"
            result.should_reduce_only = self.enable_reduce_only_guard
            result.should_reject_entry = self.enable_entry_rejection
            result.reason = f"Critical liquidation risk (distance: {distance_pct:.1f}% <= {self.critical_buffer_pct}%)"

        elif distance_pct <= self.warning_buffer_pct:
            # Warning risk
            result.risk_level = "warning"
            result.should_reject_entry = self.enable_entry_rejection
            result.reason = (
                f"Liquidation warning (distance: {distance_pct:.1f}% <= {self.warning_buffer_pct}%)"
            )

        else:
            # Safe
            result.risk_level = "safe"
            result.reason = f"Safe distance to liquidation ({distance_pct:.1f}%)"

        return result

    def should_block_new_position(
        self, symbol: str, existing_positions: dict[str, MarginInfo]
    ) -> tuple[bool, str]:
        """
        Check if new position entries should be blocked due to existing liquidation risk.

        Args:
            symbol: Symbol for potential new position
            existing_positions: Dict of symbol -> MarginInfo for existing positions

        Returns:
            (should_block, reason)
        """
        # Check if we already have a risky position in this symbol
        if symbol in existing_positions:
            risk = self.assess_liquidation_risk(existing_positions[symbol])
            if risk.should_reject_entry:
                return True, f"Existing position liquidation risk: {risk.reason}"

        # Check if any position is in critical state (portfolio-level risk)
        critical_positions = []
        for pos_symbol, margin_info in existing_positions.items():
            risk = self.assess_liquidation_risk(margin_info)
            if risk.risk_level == "critical":
                critical_positions.append(pos_symbol)

        if critical_positions and self.enable_entry_rejection:
            return (
                True,
                f"Portfolio liquidation risk from positions: {', '.join(critical_positions)}",
            )

        return False, "No liquidation risk blocking new entries"

    def log_risk_summary(self, positions: dict[str, MarginInfo]) -> None:
        """Log liquidation risk summary for all positions."""
        if not positions:
            logger.info("No positions to monitor")
            return

        logger.info("=== Liquidation Risk Summary ===")

        for symbol, margin_info in positions.items():
            risk = self.assess_liquidation_risk(margin_info)

            logger.info(f"{symbol}:")
            logger.info(
                f"  Position: {margin_info.position_side} {margin_info.position_size} @ {margin_info.entry_price}"
            )
            logger.info(
                f"  Current: {margin_info.current_price}, Leverage: {margin_info.leverage}x"
            )
            logger.info(f"  Liquidation: {risk.liquidation_price}")
            logger.info(f"  Distance: {risk.distance_pct:.1f}% ({risk.distance_bps:.0f}bps)")
            logger.info(f"  Risk: {risk.risk_level.upper()} - {risk.reason}")

            if risk.should_reduce_only:
                logger.warning(f"  >>> REDUCE-ONLY mode recommended for {symbol}")
            if risk.should_reject_entry:
                logger.warning(f"  >>> REJECT new entries for {symbol}")


# Helper functions for testing
def create_test_margin_info(
    symbol: str = "BTC-PERP",
    position_size: float = 1.0,
    position_side: str = "long",
    entry_price: float = 50000.0,
    current_price: float = 50000.0,
    leverage: float = 3.0,
    maintenance_margin_rate: float = 0.05,
) -> MarginInfo:
    """Create test margin info for unit tests."""
    return MarginInfo(
        symbol=symbol,
        position_size=Decimal(str(position_size)),
        position_side=position_side,
        entry_price=Decimal(str(entry_price)),
        current_price=Decimal(str(current_price)),
        leverage=Decimal(str(leverage)),
        maintenance_margin_rate=Decimal(str(maintenance_margin_rate)),
    )

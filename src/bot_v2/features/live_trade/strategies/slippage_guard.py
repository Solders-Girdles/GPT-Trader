"""
Week 2: Slippage guard with depth-based limits.

Cap order quantities using L1/L10 depth to keep expected market impact
under configured basis point targets. Reject orders if not satisfiable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class SlippageConfig:
    """Configuration for slippage protection."""

    max_impact_bps: float = 50.0  # Maximum allowed market impact in bps
    use_l1_only: bool = False  # Use only L1 depth vs L1+L10
    safety_buffer_pct: float = 20.0  # Safety buffer (reduce depth by %)


@dataclass
class MarketDepth:
    """Market depth information for slippage calculation."""

    symbol: str
    mid_price: Decimal | None = None
    l1_bid_depth: Decimal | None = None  # Size at best bid
    l1_ask_depth: Decimal | None = None  # Size at best ask
    l10_bid_depth: Decimal | None = None  # Total size top 10 bid levels
    l10_ask_depth: Decimal | None = None  # Total size top 10 ask levels

    @property
    def is_valid(self) -> bool:
        """Check if depth data is complete."""
        return all(
            [
                self.mid_price is not None,
                self.l1_bid_depth is not None,
                self.l1_ask_depth is not None,
                self.l10_bid_depth is not None,
                self.l10_ask_depth is not None,
            ]
        )


class SlippageGuard:
    """Guard against excessive slippage using order book depth."""

    def __init__(self, config: SlippageConfig) -> None:
        self.config = config

    def _get_available_depth(self, depth: MarketDepth, side: str) -> Decimal:
        """Get available depth for order side."""
        if side.lower() == "buy":
            # Buy orders consume ask liquidity
            if self.config.use_l1_only:
                return depth.l1_ask_depth or Decimal("0")
            else:
                return depth.l10_ask_depth or Decimal("0")
        else:
            # Sell orders consume bid liquidity
            if self.config.use_l1_only:
                return depth.l1_bid_depth or Decimal("0")
            else:
                return depth.l10_bid_depth or Decimal("0")

    def _apply_safety_buffer(self, depth: Decimal) -> Decimal:
        """Apply safety buffer to available depth."""
        buffer_factor = Decimal(str(1.0 - self.config.safety_buffer_pct / 100.0))
        return depth * buffer_factor

    def calculate_max_quantity_for_impact(
        self, depth: MarketDepth, side: str, target_impact_bps: float
    ) -> Decimal | None:
        """
        Calculate maximum order quantity for target market impact.

        Simplified model: impact_bps = (order_quantity / available_depth) * 10000
        So: max_quantity = (target_impact_bps / 10000) * available_depth

        Args:
            depth: Market depth data
            side: Order side ("buy" or "sell")
            target_impact_bps: Target impact in basis points

        Returns:
            Maximum quantity or None if insufficient data
        """
        if not depth.is_valid or depth.mid_price is None:
            return None

        available_depth = self._get_available_depth(depth, side)
        if available_depth <= 0:
            return None

        # Apply safety buffer
        effective_depth = self._apply_safety_buffer(available_depth)

        # Calculate max quantity for target impact
        impact_ratio = target_impact_bps / 10000.0  # Convert bps to ratio
        max_quantity = effective_depth * Decimal(str(impact_ratio))

        return max_quantity

    def should_reject_order(
        self, depth: MarketDepth, side: str, quantity: Decimal
    ) -> tuple[bool, str]:
        """
        Check if order should be rejected due to excessive slippage impact.

        Args:
            depth: Current market depth
            side: Order side ("buy" or "sell")
            quantity: Desired order quantity

        Returns:
            (should_reject, reason)
        """
        if not depth.is_valid:
            return True, f"Insufficient market data for {depth.symbol}"

        # Calculate maximum allowed quantity
        max_quantity = self.calculate_max_quantity_for_impact(
            depth, side, self.config.max_impact_bps
        )

        if max_quantity is None:
            return True, f"Cannot calculate slippage impact for {depth.symbol}"

        if quantity > max_quantity:
            # Calculate actual impact
            available_depth = self._get_available_depth(depth, side)
            effective_depth = self._apply_safety_buffer(available_depth)

            if effective_depth > 0:
                impact_ratio = float(quantity / effective_depth)
                impact_bps = impact_ratio * 10000

                return True, (
                    f"Order quantity {quantity} would cause {impact_bps:.1f}bps impact "
                    f"(max {self.config.max_impact_bps}bps). Max safe quantity: {max_quantity}"
                )
            else:
                return True, f"No effective depth available for {side} on {depth.symbol}"

        return False, "Slippage within acceptable limits"

    def get_safe_quantity(
        self, depth: MarketDepth, side: str, desired_quantity: Decimal
    ) -> tuple[Decimal, str]:
        """
        Get safe order quantity that stays within slippage limits.

        Args:
            depth: Current market depth
            side: Order side
            desired_quantity: Desired order size

        Returns:
            (safe_quantity, reason)
        """
        should_reject, reason = self.should_reject_order(depth, side, desired_quantity)

        if not should_reject:
            return desired_quantity, "Desired quantity is safe"

        # Calculate maximum safe quantity
        max_safe_quantity = self.calculate_max_quantity_for_impact(
            depth, side, self.config.max_impact_bps
        )

        if max_safe_quantity is None or max_safe_quantity <= 0:
            return Decimal("0"), f"No safe quantity available: {reason}"

        return max_safe_quantity, (
            f"Reduced from {desired_quantity} to {max_safe_quantity} for slippage protection"
        )

    def log_depth_summary(self, depth: MarketDepth) -> None:
        """Log current depth conditions for debugging."""
        logger.info(f"Depth summary for {depth.symbol}:")
        logger.info(f"  Mid: {depth.mid_price}")
        logger.info(f"  L1 depth: {depth.l1_bid_depth} (bid) / {depth.l1_ask_depth} (ask)")
        logger.info(f"  L10 depth: {depth.l10_bid_depth} (bid) / {depth.l10_ask_depth} (ask)")

        # Calculate max safe quantities for both sides
        for side in ["buy", "sell"]:
            max_quantity = self.calculate_max_quantity_for_impact(
                depth, side, self.config.max_impact_bps
            )
            depth_type = "L1" if self.config.use_l1_only else "L10"
            logger.info(f"  Max safe {side} quantity ({depth_type}): {max_quantity}")


# Helper functions for testing
def create_test_depth(
    symbol: str = "BTC-PERP",
    mid_price: float = 50000.0,
    l1_bid: float = 100.0,
    l1_ask: float = 100.0,
    l10_bid: float = 1000.0,
    l10_ask: float = 1000.0,
) -> MarketDepth:
    """Create test market depth for unit tests."""
    return MarketDepth(
        symbol=symbol,
        mid_price=Decimal(str(mid_price)),
        l1_bid_depth=Decimal(str(l1_bid)),
        l1_ask_depth=Decimal(str(l1_ask)),
        l10_bid_depth=Decimal(str(l10_bid)),
        l10_ask_depth=Decimal(str(l10_ask)),
    )

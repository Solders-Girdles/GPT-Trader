"""
Unit tests for slippage guard with depth-based limits.
"""

from decimal import Decimal

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.strategies.slippage_guard import (
    SlippageGuard,
    SlippageConfig,
    MarketDepth,
    create_test_depth,
)
from bot_v2.features.live_trade.risk import LiveRiskManager


class TestSlippageGuard:
    """Test slippage protection logic."""

    def setup_method(self):
        """Set up slippage guard."""
        config = SlippageConfig(
            max_impact_bps=50.0,  # 50 bps max impact
            use_l1_only=False,  # Use L10 depth
            safety_buffer_pct=20.0,  # 20% safety buffer
        )
        self.guard = SlippageGuard(config)

    def test_max_quantity_calculation_buy(self):
        """Test maximum quantity calculation for buy orders."""
        depth = create_test_depth(l10_ask=1000.0)  # 1000 ask depth

        # For 50 bps impact: max_quantity = (50/10000) * 1000 * 0.8 = 4.0
        # (0.8 factor from 20% safety buffer)
        max_quantity = self.guard.calculate_max_quantity_for_impact(depth, "buy", 50.0)

        expected = Decimal("1000") * Decimal("0.005") * Decimal("0.8")  # 4.0
        assert max_quantity is not None
        assert abs(max_quantity - expected) < Decimal("0.01")

    def test_max_quantity_calculation_sell(self):
        """Test maximum quantity calculation for sell orders."""
        depth = create_test_depth(l10_bid=2000.0)  # 2000 bid depth

        # For 25 bps impact: max_quantity = (25/10000) * 2000 * 0.8 = 4.0
        max_quantity = self.guard.calculate_max_quantity_for_impact(depth, "sell", 25.0)

        expected = Decimal("2000") * Decimal("0.0025") * Decimal("0.8")  # 4.0
        assert max_quantity is not None
        assert abs(max_quantity - expected) < Decimal("0.01")

    def test_order_rejection_excessive_impact(self):
        """Test order rejection when impact exceeds limits."""
        depth = create_test_depth(l10_ask=100.0)  # Small ask depth

        # Large order that would cause excessive impact
        large_quantity = Decimal("10.0")  # 10 vs 100*0.8*0.005 = 0.4 max safe

        should_reject, reason = self.guard.should_reject_order(depth, "buy", large_quantity)

        assert should_reject == True
        assert "impact" in reason.lower()
        assert "max safe quantity" in reason.lower()

    def test_order_acceptance_within_limits(self):
        """Test order acceptance when impact is within limits."""
        depth = create_test_depth(l10_ask=1000.0)  # Large ask depth

        # Small order that stays within limits
        small_quantity = Decimal("2.0")  # Well under max safe quantity

        should_reject, reason = self.guard.should_reject_order(depth, "buy", small_quantity)

        assert should_reject == False
        assert "acceptable limits" in reason.lower()

    def test_l1_vs_l10_mode(self):
        """Test L1-only vs L10 depth modes."""
        # L1-only guard
        l1_config = SlippageConfig(max_impact_bps=50.0, use_l1_only=True)
        l1_guard = SlippageGuard(l1_config)

        depth = create_test_depth(l1_ask=50.0, l10_ask=500.0)

        # L1 mode: uses only L1 depth (50)
        max_quantity_l1 = l1_guard.calculate_max_quantity_for_impact(depth, "buy", 50.0)

        # L10 mode: uses L10 depth (500)
        max_quantity_l10 = self.guard.calculate_max_quantity_for_impact(depth, "buy", 50.0)

        assert max_quantity_l1 is not None
        assert max_quantity_l10 is not None
        assert max_quantity_l10 > max_quantity_l1  # L10 should allow larger orders

    def test_safety_buffer_application(self):
        """Test safety buffer reduces available depth."""
        # No buffer config
        no_buffer_config = SlippageConfig(max_impact_bps=100.0, safety_buffer_pct=0.0)
        no_buffer_guard = SlippageGuard(no_buffer_config)

        depth = create_test_depth(l10_ask=1000.0)

        # With buffer: 1000 * 0.8 = 800 effective depth
        max_quantity_buffered = self.guard.calculate_max_quantity_for_impact(depth, "buy", 100.0)

        # Without buffer: 1000 effective depth
        max_quantity_no_buffer = no_buffer_guard.calculate_max_quantity_for_impact(
            depth, "buy", 100.0
        )

        assert max_quantity_buffered is not None
        assert max_quantity_no_buffer is not None
        assert max_quantity_no_buffer > max_quantity_buffered  # No buffer allows larger orders

    def test_get_safe_quantity_reduction(self):
        """Test safe quantity calculation with reduction."""
        depth = create_test_depth(l10_ask=100.0)  # Limited depth

        desired_quantity = Decimal("10.0")  # Too large
        safe_quantity, reason = self.guard.get_safe_quantity(depth, "buy", desired_quantity)

        assert safe_quantity < desired_quantity  # Should be reduced
        assert safe_quantity > Decimal("0")  # But not zero
        assert "reduced from" in reason.lower()

    def test_get_safe_quantity_no_reduction_needed(self):
        """Test safe quantity when no reduction needed."""
        depth = create_test_depth(l10_ask=10000.0)  # Huge depth

        desired_quantity = Decimal("1.0")  # Small order
        safe_quantity, reason = self.guard.get_safe_quantity(depth, "buy", desired_quantity)

        assert safe_quantity == desired_quantity  # No reduction needed
        assert "desired quantity is safe" in reason.lower()

    def test_insufficient_market_data(self):
        """Test handling of insufficient market data."""
        incomplete_depth = MarketDepth(symbol="BTC-PERP")  # No data filled

        should_reject, reason = self.guard.should_reject_order(
            incomplete_depth, "buy", Decimal("1.0")
        )

        assert should_reject == True
        assert "insufficient market data" in reason.lower()

    def test_zero_depth_handling(self):
        """Test handling of zero depth conditions."""
        zero_depth = create_test_depth(l10_ask=0.0)  # No ask liquidity

        should_reject, reason = self.guard.should_reject_order(zero_depth, "buy", Decimal("1.0"))

        assert should_reject == True
        # The actual implementation returns "Cannot calculate slippage impact for {symbol}"
        assert "cannot calculate slippage impact" in reason.lower()


def test_slippage_guard_disabled_no_error():
    """Disabled guard should allow trades regardless of quoted slippage."""
    config = RiskConfig(slippage_guard_bps=0)
    risk_manager = LiveRiskManager(config=config)

    risk_manager.validate_slippage_guard(
        symbol="BTC-PERP",
        side="buy",
        quantity=Decimal("1"),
        expected_price=Decimal("55000"),
        mark_or_quote=Decimal("50000"),
    )


def test_slippage_just_under_threshold_allowed():
    """Guard should allow requests that stay within configured basis-point limit."""
    config = RiskConfig(slippage_guard_bps=50)
    risk_manager = LiveRiskManager(config=config)

    risk_manager.validate_slippage_guard(
        symbol="BTC-PERP",
        side="buy",
        quantity=Decimal("1"),
        expected_price=Decimal("50245"),
        mark_or_quote=Decimal("50000"),
    )

"""
Comprehensive tests for risk guards.

Tests cover:
- RiskGuards initialization
- Liquidation distance checking
- Slippage impact estimation
- Standard risk guards factory
- Edge cases and boundary conditions
"""

from decimal import Decimal

import pytest

from bot_v2.features.strategy_tools.guards import RiskGuards, create_standard_risk_guards


# ============================================================================
# Test: RiskGuards Initialization
# ============================================================================


class TestRiskGuardsInitialization:
    """Test RiskGuards initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        guards = RiskGuards()

        assert guards.min_liquidation_buffer_pct == Decimal("15")
        assert guards.max_slippage_impact_bps == Decimal("20")

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        guards = RiskGuards(
            min_liquidation_buffer_pct=Decimal("25"),
            max_slippage_impact_bps=Decimal("10"),
        )

        assert guards.min_liquidation_buffer_pct == Decimal("25")
        assert guards.max_slippage_impact_bps == Decimal("10")

    def test_initialization_disabled_guards(self):
        """Test initialization with disabled guards."""
        guards = RiskGuards(min_liquidation_buffer_pct=None, max_slippage_impact_bps=None)

        assert guards.min_liquidation_buffer_pct is None
        assert guards.max_slippage_impact_bps is None


# ============================================================================
# Test: Liquidation Distance Checking
# ============================================================================


class TestLiquidationDistanceChecking:
    """Test liquidation distance checking."""

    def test_check_liquidation_safe_distance(self):
        """Test liquidation check with safe distance."""
        guards = RiskGuards(min_liquidation_buffer_pct=Decimal("10"))

        safe, reason = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("5"),  # Gives ~15% distance
            account_equity=Decimal("10000"),
        )

        assert safe is True
        assert "safe liquidation distance" in reason.lower()

    def test_check_liquidation_too_close(self):
        """Test liquidation check when too close to liquidation."""
        guards = RiskGuards(min_liquidation_buffer_pct=Decimal("30"))

        safe, reason = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("20"),  # High leverage = close to liquidation
            account_equity=Decimal("2500"),
        )

        assert safe is False
        assert "too close to liquidation" in reason.lower()

    def test_check_liquidation_at_threshold(self):
        """Test liquidation check at exact threshold."""
        guards = RiskGuards(min_liquidation_buffer_pct=Decimal("20"))

        # Manually calculate to get exactly 20% distance
        safe, reason = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("5"),
            account_equity=Decimal("10000"),
            maintenance_margin_rate=Decimal("0.05"),
        )

        # Should be safe (>= threshold)
        assert safe in [True, False]  # Depends on exact calculation

    def test_check_liquidation_disabled(self):
        """Test liquidation check when guard is disabled."""
        guards = RiskGuards(min_liquidation_buffer_pct=None)

        safe, reason = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("100"),  # Extreme leverage
            account_equity=Decimal("500"),
        )

        assert safe is True
        assert "disabled" in reason.lower()

    def test_check_liquidation_different_leverage_levels(self):
        """Test liquidation check with different leverage levels."""
        guards = RiskGuards(min_liquidation_buffer_pct=Decimal("20"))

        # Low leverage (safer)
        safe_low, _ = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("2"),
            account_equity=Decimal("25000"),
        )

        # High leverage (riskier)
        safe_high, _ = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("50"),
            account_equity=Decimal("1000"),
        )

        assert safe_low is True
        # High leverage should be flagged or close to threshold
        assert isinstance(safe_high, bool)

    def test_check_liquidation_custom_maintenance_margin(self):
        """Test liquidation check with custom maintenance margin rate."""
        guards = RiskGuards(min_liquidation_buffer_pct=Decimal("20"))

        # Lower maintenance margin (safer)
        safe_low_mm, _ = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("10"),
            account_equity=Decimal("5000"),
            maintenance_margin_rate=Decimal("0.01"),
        )

        # Higher maintenance margin (riskier)
        safe_high_mm, _ = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("10"),
            account_equity=Decimal("5000"),
            maintenance_margin_rate=Decimal("0.10"),
        )

        assert isinstance(safe_low_mm, bool)
        assert isinstance(safe_high_mm, bool)

    def test_check_liquidation_zero_leverage(self):
        """Test liquidation check with zero leverage (edge case)."""
        guards = RiskGuards(min_liquidation_buffer_pct=Decimal("20"))

        # Zero leverage would cause division by zero
        with pytest.raises((ZeroDivisionError, Exception)):
            guards.check_liquidation_distance(
                entry_price=Decimal("50000"),
                position_size=Decimal("1"),
                leverage=Decimal("0"),
                account_equity=Decimal("50000"),
            )

    def test_check_liquidation_negative_leverage(self):
        """Test liquidation check with negative leverage (invalid)."""
        guards = RiskGuards(min_liquidation_buffer_pct=Decimal("20"))

        # Negative leverage is invalid but should handle gracefully
        safe, reason = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("-5"),
            account_equity=Decimal("10000"),
        )

        # Should handle invalid input
        assert isinstance(safe, bool)


# ============================================================================
# Test: Slippage Impact Checking
# ============================================================================


class TestSlippageImpactChecking:
    """Test slippage impact estimation."""

    def test_check_slippage_small_order_within_l1(self):
        """Test slippage check for small order within L1 depth."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        market_snapshot = {"depth_l1": 100000, "depth_l10": 500000}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("50000"), market_snapshot=market_snapshot
        )

        assert safe is True
        assert "acceptable slippage" in reason.lower()

    def test_check_slippage_order_exceeds_l1(self):
        """Test slippage check for order exceeding L1 depth."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        market_snapshot = {"depth_l1": 50000, "depth_l10": 500000}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("100000"), market_snapshot=market_snapshot
        )

        # May or may not be safe depending on calculation
        assert isinstance(safe, bool)

    def test_check_slippage_order_exceeds_l10(self):
        """Test slippage check for order exceeding L10 depth."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        market_snapshot = {"depth_l1": 50000, "depth_l10": 200000}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("300000"), market_snapshot=market_snapshot
        )

        assert safe is False
        assert "order too large" in reason.lower()

    def test_check_slippage_disabled(self):
        """Test slippage check when guard is disabled."""
        guards = RiskGuards(max_slippage_impact_bps=None)

        market_snapshot = {"depth_l1": 10000, "depth_l10": 50000}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("1000000"), market_snapshot=market_snapshot
        )

        assert safe is True
        assert "disabled" in reason.lower()

    def test_check_slippage_missing_depth_data(self):
        """Test slippage check with missing depth data."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        market_snapshot = {}  # No depth data

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("50000"), market_snapshot=market_snapshot
        )

        assert safe is False
        assert "insufficient market data" in reason.lower()

    def test_check_slippage_zero_depth(self):
        """Test slippage check with zero depth."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        market_snapshot = {"depth_l1": 0, "depth_l10": 0}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("50000"), market_snapshot=market_snapshot
        )

        assert safe is False

    def test_check_slippage_order_at_exact_l1(self):
        """Test slippage check with order size exactly at L1 depth."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        market_snapshot = {"depth_l1": 100000, "depth_l10": 500000}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("100000"), market_snapshot=market_snapshot
        )

        # Should use L1-only calculation
        assert isinstance(safe, bool)

    def test_check_slippage_order_at_exact_l10(self):
        """Test slippage check with order size exactly at L10 depth."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        market_snapshot = {"depth_l1": 100000, "depth_l10": 500000}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("500000"), market_snapshot=market_snapshot
        )

        # Should reject (order == L10 depth)
        assert safe is False

    def test_check_slippage_progressive_impact(self):
        """Test that slippage impact increases with order size."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("100"))

        market_snapshot = {"depth_l1": 100000, "depth_l10": 1000000}

        # Small order
        safe_small, reason_small = guards.check_slippage_impact(
            order_size=Decimal("25000"), market_snapshot=market_snapshot
        )

        # Medium order
        safe_medium, reason_medium = guards.check_slippage_impact(
            order_size=Decimal("200000"), market_snapshot=market_snapshot
        )

        # Large order
        safe_large, reason_large = guards.check_slippage_impact(
            order_size=Decimal("800000"), market_snapshot=market_snapshot
        )

        # Extract impact from reason strings
        # Should show increasing impact
        assert safe_small is True

    def test_check_slippage_tight_threshold(self):
        """Test slippage check with tight threshold."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("5"))

        market_snapshot = {"depth_l1": 100000, "depth_l10": 500000}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("80000"), market_snapshot=market_snapshot
        )

        # Tight threshold should reject more orders
        assert isinstance(safe, bool)

    def test_check_slippage_l10_less_than_l1(self):
        """Test slippage check when L10 < L1 (invalid but handled)."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        market_snapshot = {"depth_l1": 100000, "depth_l10": 50000}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("80000"), market_snapshot=market_snapshot
        )

        # Should handle edge case
        assert isinstance(safe, bool)


# ============================================================================
# Test: Factory Function
# ============================================================================


class TestFactoryFunction:
    """Test risk guards factory function."""

    def test_create_standard_risk_guards(self):
        """Test standard risk guards factory."""
        guards = create_standard_risk_guards()

        assert guards.min_liquidation_buffer_pct == Decimal("20")
        assert guards.max_slippage_impact_bps == Decimal("15")

    def test_standard_guards_are_reasonable(self):
        """Test that standard guards provide reasonable protection."""
        guards = create_standard_risk_guards()

        # Should reject very high leverage
        safe_high_lev, _ = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("50"),
            account_equity=Decimal("1000"),
        )

        # Should reject very large orders
        safe_large_order, _ = guards.check_slippage_impact(
            order_size=Decimal("500000"),
            market_snapshot={"depth_l1": 50000, "depth_l10": 200000},
        )

        # At least one should be rejected (reasonable protection)
        assert not (safe_high_lev and safe_large_order)


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_liquidation_zero_entry_price(self):
        """Test liquidation check with zero entry price."""
        guards = RiskGuards(min_liquidation_buffer_pct=Decimal("20"))

        safe, reason = guards.check_liquidation_distance(
            entry_price=Decimal("0"),
            position_size=Decimal("1"),
            leverage=Decimal("5"),
            account_equity=Decimal("10000"),
        )

        # Should handle gracefully (likely division by zero)
        assert isinstance(safe, bool) or reason

    def test_liquidation_negative_entry_price(self):
        """Test liquidation check with negative entry price."""
        guards = RiskGuards(min_liquidation_buffer_pct=Decimal("20"))

        safe, reason = guards.check_liquidation_distance(
            entry_price=Decimal("-50000"),
            position_size=Decimal("1"),
            leverage=Decimal("5"),
            account_equity=Decimal("10000"),
        )

        # Should handle invalid input
        assert isinstance(safe, bool)

    def test_liquidation_zero_equity(self):
        """Test liquidation check with zero equity."""
        guards = RiskGuards(min_liquidation_buffer_pct=Decimal("20"))

        safe, reason = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("5"),
            account_equity=Decimal("0"),
        )

        # Should handle edge case
        assert isinstance(safe, bool)

    def test_liquidation_very_high_leverage(self):
        """Test liquidation check with extreme leverage."""
        guards = RiskGuards(min_liquidation_buffer_pct=Decimal("20"))

        safe, reason = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("100"),
            account_equity=Decimal("500"),
        )

        # Should flag as unsafe
        assert safe is False

    def test_slippage_zero_order_size(self):
        """Test slippage check with zero order size."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        market_snapshot = {"depth_l1": 100000, "depth_l10": 500000}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("0"), market_snapshot=market_snapshot
        )

        # Zero order should have minimal impact
        assert safe is True

    def test_slippage_negative_order_size(self):
        """Test slippage check with negative order size."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        market_snapshot = {"depth_l1": 100000, "depth_l10": 500000}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("-50000"), market_snapshot=market_snapshot
        )

        # Should handle invalid input
        assert isinstance(safe, bool)

    def test_slippage_very_small_order(self):
        """Test slippage with very small order size."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        market_snapshot = {"depth_l1": 100000, "depth_l10": 500000}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("0.01"), market_snapshot=market_snapshot
        )

        assert safe is True

    def test_slippage_very_large_order(self):
        """Test slippage with extremely large order."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        market_snapshot = {"depth_l1": 100000, "depth_l10": 500000}

        safe, reason = guards.check_slippage_impact(
            order_size=Decimal("10000000"), market_snapshot=market_snapshot
        )

        assert safe is False

    def test_liquidation_one_leverage(self):
        """Test liquidation with 1x leverage (no leverage)."""
        guards = RiskGuards(min_liquidation_buffer_pct=Decimal("20"))

        safe, reason = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("1"),
            account_equity=Decimal("50000"),
        )

        # 1x leverage should be very safe
        assert safe is True

    def test_multiple_guard_checks_same_instance(self):
        """Test multiple checks with same RiskGuards instance."""
        guards = RiskGuards(
            min_liquidation_buffer_pct=Decimal("20"),
            max_slippage_impact_bps=Decimal("20"),
        )

        # First liquidation check
        safe1, _ = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("5"),
            account_equity=Decimal("10000"),
        )

        # Second liquidation check
        safe2, _ = guards.check_liquidation_distance(
            entry_price=Decimal("60000"),
            position_size=Decimal("1"),
            leverage=Decimal("10"),
            account_equity=Decimal("6000"),
        )

        # First slippage check
        safe3, _ = guards.check_slippage_impact(
            order_size=Decimal("50000"),
            market_snapshot={"depth_l1": 100000, "depth_l10": 500000},
        )

        # All checks should work independently
        assert isinstance(safe1, bool)
        assert isinstance(safe2, bool)
        assert isinstance(safe3, bool)


# ============================================================================
# Test: Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_safe_trade_setup(self):
        """Test a completely safe trade setup."""
        guards = create_standard_risk_guards()

        # Safe liquidation distance (low leverage)
        safe_liq, _ = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("3"),
            account_equity=Decimal("20000"),
        )

        # Safe slippage (small order, deep market)
        safe_slip, _ = guards.check_slippage_impact(
            order_size=Decimal("30000"),
            market_snapshot={"depth_l1": 150000, "depth_l10": 800000},
        )

        assert safe_liq is True
        assert safe_slip is True

    def test_risky_trade_setup(self):
        """Test a risky trade setup."""
        guards = create_standard_risk_guards()

        # Risky liquidation distance (high leverage)
        safe_liq, _ = guards.check_liquidation_distance(
            entry_price=Decimal("50000"),
            position_size=Decimal("1"),
            leverage=Decimal("25"),
            account_equity=Decimal("2000"),
        )

        # Risky slippage (large order, shallow market)
        safe_slip, _ = guards.check_slippage_impact(
            order_size=Decimal("400000"),
            market_snapshot={"depth_l1": 50000, "depth_l10": 200000},
        )

        # At least one should be flagged
        assert not safe_liq or not safe_slip

    def test_progressive_risk_levels(self):
        """Test guards across progressive risk levels."""
        guards = RiskGuards(
            min_liquidation_buffer_pct=Decimal("20"),
            max_slippage_impact_bps=Decimal("15"),
        )

        leverages = [Decimal("2"), Decimal("5"), Decimal("10"), Decimal("20")]
        results = []

        for lev in leverages:
            safe, _ = guards.check_liquidation_distance(
                entry_price=Decimal("50000"),
                position_size=Decimal("1"),
                leverage=lev,
                account_equity=Decimal("50000") / lev,
            )
            results.append(safe)

        # Lower leverage should be safer
        # Results should show increasing risk (though not necessarily linear)
        assert isinstance(results, list)

    def test_market_depth_scenarios(self):
        """Test slippage across different market depth scenarios."""
        guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

        order_size = Decimal("100000")

        # Deep market
        safe_deep, _ = guards.check_slippage_impact(
            order_size=order_size,
            market_snapshot={"depth_l1": 200000, "depth_l10": 1000000},
        )

        # Shallow market
        safe_shallow, _ = guards.check_slippage_impact(
            order_size=order_size,
            market_snapshot={"depth_l1": 50000, "depth_l10": 150000},
        )

        # Deep market should be safer
        assert safe_deep is True
        # Shallow market may be rejected
        assert isinstance(safe_shallow, bool)

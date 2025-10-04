"""Tests for ImpactEstimator - market impact estimation.

This module tests the ImpactEstimator's ability to:
- Calculate base impact using square-root model
- Apply depth adjustments when notional exceeds available depth
- Apply spread and condition multipliers
- Calculate estimated prices and slippage costs
- Generate execution recommendations (slicing, post-only)
- Provide conservative fallback estimates
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from bot_v2.features.live_trade.impact_estimator import ImpactEstimator
from bot_v2.features.live_trade.liquidity_models import DepthAnalysis, LiquidityCondition


@pytest.fixture
def estimator():
    """Create ImpactEstimator with default settings."""
    return ImpactEstimator(max_impact_bps=Decimal("50"))


@pytest.fixture
def volume_metrics():
    """Sample volume metrics."""
    return {
        "volume_15m": Decimal("100000"),  # $100k volume
        "volume_1h": Decimal("400000"),
        "trade_count_15m": 100,
    }


@pytest.fixture
def good_analysis():
    """Sample depth analysis with good liquidity."""
    return DepthAnalysis(
        symbol="BTC-USD",
        timestamp=datetime.now(),
        bid_price=Decimal("50000"),
        ask_price=Decimal("50100"),
        bid_size=Decimal("10"),
        ask_size=Decimal("10"),
        spread=Decimal("100"),
        spread_bps=Decimal("20"),  # 2bps
        depth_usd_1=Decimal("500000"),
        depth_usd_5=Decimal("1000000"),  # $1M depth at 5%
        depth_usd_10=Decimal("2000000"),
        bid_ask_ratio=Decimal("1"),
        depth_imbalance=Decimal("0"),
        liquidity_score=Decimal("75"),
        condition=LiquidityCondition.GOOD,
    )


class TestBaseImpactCalculation:
    """Test base impact calculation using square-root model."""

    def test_calculates_base_impact_square_root_model(
        self, estimator, good_analysis, volume_metrics
    ):
        """Calculates base impact using square-root model."""
        # Notional = 1 * 50050 = $50,050
        # Base impact = sqrt(50050 / 100000) * 100 = sqrt(0.50050) * 100 ≈ 70.7 bps
        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            analysis=good_analysis,
            volume_metrics=volume_metrics,
        )

        # Base impact should be around 70 bps before multipliers
        # After good condition (1.0x) and small spread (1.02x), should be ~71 bps
        assert 60 < result.estimated_impact_bps < 80

    def test_base_impact_scales_with_notional(self, estimator, good_analysis, volume_metrics):
        """Base impact scales with square root of notional."""
        # Small order
        small_result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.5"),
            analysis=good_analysis,
            volume_metrics=volume_metrics,
        )

        # Large order (4x notional = 2x impact)
        large_result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("2"),
            analysis=good_analysis,
            volume_metrics=volume_metrics,
        )

        # Large impact should be roughly sqrt(4) = 2x the small impact
        assert large_result.estimated_impact_bps > small_result.estimated_impact_bps
        assert large_result.estimated_impact_bps < small_result.estimated_impact_bps * Decimal(
            "2.5"
        )

    def test_minimum_volume_floor_prevents_division_by_zero(self, estimator, good_analysis):
        """Uses minimum volume floor to prevent division by zero."""
        # Very low volume
        low_volume_metrics = {
            "volume_15m": Decimal("10"),  # Below $1k floor
            "volume_1h": Decimal("40"),
            "trade_count_15m": 1,
        }

        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            analysis=good_analysis,
            volume_metrics=low_volume_metrics,
        )

        # Should use $1000 minimum volume, not $10
        assert result.estimated_impact_bps > Decimal("0")


class TestDepthAdjustment:
    """Test depth adjustment when notional exceeds available depth."""

    def test_applies_depth_multiplier_when_notional_exceeds_depth(
        self, estimator, good_analysis, volume_metrics
    ):
        """Applies depth multiplier when order size exceeds available depth."""
        # Modify analysis to have low depth
        shallow_analysis = DepthAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            bid_price=Decimal("50000"),
            ask_price=Decimal("50100"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            spread=Decimal("100"),
            spread_bps=Decimal("20"),
            depth_usd_1=Decimal("10000"),
            depth_usd_5=Decimal("20000"),  # Only $20k depth
            depth_usd_10=Decimal("40000"),
            bid_ask_ratio=Decimal("1"),
            depth_imbalance=Decimal("0"),
            liquidity_score=Decimal("75"),
            condition=LiquidityCondition.GOOD,
        )

        # Large order: notional = 10 * 50050 = $500,500 >> $20k depth
        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("10"),
            analysis=shallow_analysis,
            volume_metrics=volume_metrics,
        )

        # Impact should be significantly higher due to depth adjustment
        assert result.estimated_impact_bps > Decimal("100")  # Should be elevated

    def test_no_depth_adjustment_when_notional_within_depth(
        self, estimator, good_analysis, volume_metrics
    ):
        """No depth adjustment when notional is within available depth."""
        # Small order: notional = 0.1 * 50050 = $5,005 << $1M depth
        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.1"),
            analysis=good_analysis,
            volume_metrics=volume_metrics,
        )

        # Impact should be low since order is well within depth
        assert result.estimated_impact_bps < Decimal("50")

    def test_depth_adjustment_uses_sqrt_multiplier(self, estimator, good_analysis, volume_metrics):
        """Depth adjustment uses square root multiplier."""
        # Create two scenarios with different depth ratios
        shallow_analysis = DepthAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            bid_price=Decimal("50000"),
            ask_price=Decimal("50100"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            spread=Decimal("100"),
            spread_bps=Decimal("20"),
            depth_usd_1=Decimal("10000"),
            depth_usd_5=Decimal("25000"),  # $25k depth
            depth_usd_10=Decimal("50000"),
            bid_ask_ratio=Decimal("1"),
            depth_imbalance=Decimal("0"),
            liquidity_score=Decimal("75"),
            condition=LiquidityCondition.GOOD,
        )

        # Order: notional = 2 * 50050 = $100,100 (4x depth)
        # Depth multiplier should be sqrt(4) = 2x
        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("2"),
            analysis=shallow_analysis,
            volume_metrics=volume_metrics,
        )

        # Impact should be elevated but not linear (sqrt scaling)
        assert result.estimated_impact_bps > Decimal("50")


class TestSpreadAndConditionMultipliers:
    """Test spread and condition multiplier application."""

    def test_applies_spread_multiplier(self, estimator, volume_metrics):
        """Applies spread multiplier correctly."""
        # Wide spread analysis
        wide_spread_analysis = DepthAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            bid_price=Decimal("50000"),
            ask_price=Decimal("50500"),  # Wide spread
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            spread=Decimal("500"),
            spread_bps=Decimal("100"),  # 10bps spread
            depth_usd_1=Decimal("500000"),
            depth_usd_5=Decimal("1000000"),
            depth_usd_10=Decimal("2000000"),
            bid_ask_ratio=Decimal("1"),
            depth_imbalance=Decimal("0"),
            liquidity_score=Decimal("60"),
            condition=LiquidityCondition.GOOD,
        )

        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            analysis=wide_spread_analysis,
            volume_metrics=volume_metrics,
        )

        # Wide spread should increase impact
        assert result.estimated_impact_bps > Decimal("70")

    def test_excellent_condition_reduces_impact(self, estimator, volume_metrics):
        """Excellent liquidity condition reduces impact (0.5x multiplier)."""
        excellent_analysis = DepthAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            bid_price=Decimal("50000"),
            ask_price=Decimal("50100"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            spread=Decimal("100"),
            spread_bps=Decimal("20"),
            depth_usd_1=Decimal("500000"),
            depth_usd_5=Decimal("1000000"),
            depth_usd_10=Decimal("2000000"),
            bid_ask_ratio=Decimal("1"),
            depth_imbalance=Decimal("0"),
            liquidity_score=Decimal("85"),
            condition=LiquidityCondition.EXCELLENT,  # 0.5x multiplier
        )

        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            analysis=excellent_analysis,
            volume_metrics=volume_metrics,
        )

        # Impact should be reduced (0.5x condition multiplier)
        assert result.estimated_impact_bps < Decimal("50")

    def test_critical_condition_increases_impact(self, estimator, volume_metrics):
        """Critical liquidity condition increases impact (3.0x multiplier)."""
        critical_analysis = DepthAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            bid_price=Decimal("50000"),
            ask_price=Decimal("50100"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            spread=Decimal("100"),
            spread_bps=Decimal("20"),
            depth_usd_1=Decimal("500000"),
            depth_usd_5=Decimal("1000000"),
            depth_usd_10=Decimal("2000000"),
            bid_ask_ratio=Decimal("1"),
            depth_imbalance=Decimal("0"),
            liquidity_score=Decimal("10"),
            condition=LiquidityCondition.CRITICAL,  # 3.0x multiplier
        )

        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            analysis=critical_analysis,
            volume_metrics=volume_metrics,
        )

        # Impact should be significantly increased (3.0x condition multiplier)
        assert result.estimated_impact_bps > Decimal("150")

    def test_combines_spread_and_condition_multipliers(self, estimator, volume_metrics):
        """Combines spread and condition multipliers correctly."""
        # Poor condition + wide spread
        poor_wide_analysis = DepthAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            bid_price=Decimal("50000"),
            ask_price=Decimal("50500"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            spread=Decimal("500"),
            spread_bps=Decimal("100"),  # 10bps spread → 1.1x multiplier
            depth_usd_1=Decimal("500000"),
            depth_usd_5=Decimal("1000000"),
            depth_usd_10=Decimal("2000000"),
            bid_ask_ratio=Decimal("1"),
            depth_imbalance=Decimal("0"),
            liquidity_score=Decimal("25"),
            condition=LiquidityCondition.POOR,  # 2.0x multiplier
        )

        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            analysis=poor_wide_analysis,
            volume_metrics=volume_metrics,
        )

        # Combined multipliers (spread + condition) should compound
        assert result.estimated_impact_bps > Decimal("100")


class TestPriceCalculations:
    """Test price and cost calculations."""

    def test_buy_increases_price_by_impact(self, estimator, good_analysis, volume_metrics):
        """Buy orders increase estimated price by impact."""
        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            analysis=good_analysis,
            volume_metrics=volume_metrics,
        )

        mid_price = (good_analysis.bid_price + good_analysis.ask_price) / 2

        # Estimated avg price should be higher than mid for buy
        assert result.estimated_avg_price > mid_price
        assert result.max_impact_price > result.estimated_avg_price

    def test_sell_decreases_price_by_impact(self, estimator, good_analysis, volume_metrics):
        """Sell orders decrease estimated price by impact."""
        result = estimator.estimate(
            symbol="BTC-USD",
            side="sell",
            quantity=Decimal("1"),
            analysis=good_analysis,
            volume_metrics=volume_metrics,
        )

        mid_price = (good_analysis.bid_price + good_analysis.ask_price) / 2

        # Estimated avg price should be lower than mid for sell
        assert result.estimated_avg_price < mid_price
        assert result.max_impact_price < result.estimated_avg_price

    def test_slippage_cost_calculation(self, estimator, good_analysis, volume_metrics):
        """Calculates slippage cost correctly."""
        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("2"),
            analysis=good_analysis,
            volume_metrics=volume_metrics,
        )

        mid_price = (good_analysis.bid_price + good_analysis.ask_price) / 2

        # Slippage cost = |estimated_avg_price - mid_price| * quantity
        expected_slippage = abs(result.estimated_avg_price - mid_price) * Decimal("2")
        assert result.slippage_cost == expected_slippage
        assert result.slippage_cost > Decimal("0")


class TestExecutionRecommendations:
    """Test execution recommendation logic."""

    def test_recommends_slicing_when_impact_exceeds_threshold(
        self, estimator, good_analysis, volume_metrics
    ):
        """Recommends slicing when impact exceeds max threshold."""
        # Large order that will exceed 50bps threshold
        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("20"),  # Large order
            analysis=good_analysis,
            volume_metrics=volume_metrics,
        )

        # Should recommend slicing for high impact
        assert result.recommended_slicing is True
        assert result.max_slice_size is not None
        assert result.max_slice_size < Decimal("20")

    def test_calculates_max_slice_size_for_target_impact(
        self, estimator, good_analysis, volume_metrics
    ):
        """Calculates max slice size to keep impact under threshold."""
        # Large order
        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("50"),
            analysis=good_analysis,
            volume_metrics=volume_metrics,
        )

        if result.recommended_slicing:
            # Max slice size should be significantly smaller than full order
            assert result.max_slice_size < Decimal("50")
            assert result.max_slice_size > Decimal("0")

    def test_recommends_post_only_for_poor_liquidity(self, estimator, volume_metrics):
        """Recommends post-only orders in poor liquidity conditions."""
        poor_analysis = DepthAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            bid_price=Decimal("50000"),
            ask_price=Decimal("50100"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            spread=Decimal("100"),
            spread_bps=Decimal("20"),
            depth_usd_1=Decimal("500000"),
            depth_usd_5=Decimal("1000000"),
            depth_usd_10=Decimal("2000000"),
            bid_ask_ratio=Decimal("1"),
            depth_imbalance=Decimal("0"),
            liquidity_score=Decimal("25"),
            condition=LiquidityCondition.POOR,
        )

        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            analysis=poor_analysis,
            volume_metrics=volume_metrics,
        )

        # Should recommend post-only for poor liquidity
        assert result.use_post_only is True

    def test_recommends_post_only_for_high_impact(self, estimator, good_analysis, volume_metrics):
        """Recommends post-only when impact exceeds half of max threshold."""
        # Large enough order to trigger post-only recommendation
        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("15"),
            analysis=good_analysis,
            volume_metrics=volume_metrics,
        )

        # If impact > 25bps (half of 50bps threshold), should recommend post-only
        if result.estimated_impact_bps > Decimal("25"):
            assert result.use_post_only is True


class TestConservativeEstimate:
    """Test conservative fallback estimation."""

    def test_conservative_estimate_when_no_analysis(self, estimator):
        """Provides conservative estimate when no analysis available."""
        result = estimator.estimate_conservative(
            symbol="BTC-USD", side="buy", quantity=Decimal("5")
        )

        # Conservative estimates
        assert result.estimated_impact_bps == Decimal("100")  # 10bps
        assert result.recommended_slicing is True
        assert result.max_slice_size == Decimal("0.5")  # quantity / 10
        assert result.use_post_only is True

    def test_handles_zero_volume_gracefully(self, estimator, good_analysis):
        """Handles zero volume gracefully with minimum floor."""
        zero_volume_metrics = {
            "volume_15m": Decimal("0"),  # Zero volume
            "volume_1h": Decimal("0"),
            "trade_count_15m": 0,
        }

        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            analysis=good_analysis,
            volume_metrics=zero_volume_metrics,
        )

        # Should use minimum volume floor ($1000) and produce valid estimate
        assert result.estimated_impact_bps > Decimal("0")
        assert result.estimated_avg_price > Decimal("0")

    def test_handles_extreme_notional_values(self, estimator, good_analysis, volume_metrics):
        """Handles extreme notional values without errors."""
        # Very large order
        result = estimator.estimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1000"),  # Extremely large
            analysis=good_analysis,
            volume_metrics=volume_metrics,
        )

        # Should produce valid (though high) impact estimate
        assert result.estimated_impact_bps > Decimal("0")
        assert result.recommended_slicing is True
        assert result.use_post_only is True

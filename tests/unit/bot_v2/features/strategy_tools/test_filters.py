"""
Comprehensive tests for market condition filters.

Tests cover:
- MarketConditionFilters initialization
- Long entry filtering (spread, depth, volume, RSI)
- Short entry filtering
- RSI confirmation logic
- Factory functions (conservative/aggressive)
- Edge cases and boundary conditions
"""

from decimal import Decimal

import pytest

from bot_v2.features.strategy_tools.filters import (
    MarketConditionFilters,
    create_aggressive_filters,
    create_conservative_filters,
)


# ============================================================================
# Test: MarketConditionFilters Initialization
# ============================================================================


class TestMarketConditionFiltersInitialization:
    """Test MarketConditionFilters initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        filters = MarketConditionFilters()

        assert filters.max_spread_bps is None
        assert filters.min_depth_l1 is None
        assert filters.min_depth_l10 is None
        assert filters.min_volume_1m is None
        assert filters.min_volume_5m is None
        assert filters.rsi_oversold == Decimal("30")
        assert filters.rsi_overbought == Decimal("70")
        assert filters.require_rsi_confirmation is False

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        filters = MarketConditionFilters(
            max_spread_bps=Decimal("15"),
            min_depth_l1=Decimal("100000"),
            min_depth_l10=Decimal("500000"),
            min_volume_1m=Decimal("200000"),
            min_volume_5m=Decimal("1000000"),
            rsi_oversold=Decimal("25"),
            rsi_overbought=Decimal("75"),
            require_rsi_confirmation=True,
        )

        assert filters.max_spread_bps == Decimal("15")
        assert filters.min_depth_l1 == Decimal("100000")
        assert filters.min_depth_l10 == Decimal("500000")
        assert filters.min_volume_1m == Decimal("200000")
        assert filters.min_volume_5m == Decimal("1000000")
        assert filters.rsi_oversold == Decimal("25")
        assert filters.rsi_overbought == Decimal("75")
        assert filters.require_rsi_confirmation is True

    def test_initialization_partial_params(self):
        """Test initialization with partial parameters."""
        filters = MarketConditionFilters(
            max_spread_bps=Decimal("20"), require_rsi_confirmation=True
        )

        assert filters.max_spread_bps == Decimal("20")
        assert filters.min_depth_l1 is None
        assert filters.require_rsi_confirmation is True


# ============================================================================
# Test: Long Entry Filtering
# ============================================================================


class TestLongEntryFiltering:
    """Test long entry filtering logic."""

    def test_long_entry_all_conditions_pass(self):
        """Test long entry when all conditions pass."""
        filters = MarketConditionFilters(
            max_spread_bps=Decimal("20"),
            min_depth_l1=Decimal("50000"),
            min_depth_l10=Decimal("200000"),
            min_volume_1m=Decimal("100000"),
            min_volume_5m=Decimal("500000"),
        )

        market_snapshot = {
            "spread_bps": 15,
            "depth_l1": 60000,
            "depth_l10": 250000,
            "vol_1m": 150000,
            "vol_5m": 600000,
        }

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is True
        assert "acceptable" in reason.lower()

    def test_long_entry_spread_too_wide(self):
        """Test long entry rejection when spread is too wide."""
        filters = MarketConditionFilters(max_spread_bps=Decimal("10"))

        market_snapshot = {"spread_bps": 15}

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is False
        assert "spread too wide" in reason.lower()

    def test_long_entry_l1_depth_insufficient(self):
        """Test long entry rejection when L1 depth is insufficient."""
        filters = MarketConditionFilters(min_depth_l1=Decimal("100000"))

        market_snapshot = {"spread_bps": 5, "depth_l1": 50000}

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is False
        assert "l1 depth insufficient" in reason.lower()

    def test_long_entry_l10_depth_insufficient(self):
        """Test long entry rejection when L10 depth is insufficient."""
        filters = MarketConditionFilters(
            min_depth_l1=Decimal("50000"), min_depth_l10=Decimal("500000")
        )

        market_snapshot = {"spread_bps": 5, "depth_l1": 60000, "depth_l10": 300000}

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is False
        assert "l10 depth insufficient" in reason.lower()

    def test_long_entry_1m_volume_too_low(self):
        """Test long entry rejection when 1m volume is too low."""
        filters = MarketConditionFilters(min_volume_1m=Decimal("200000"))

        market_snapshot = {
            "spread_bps": 5,
            "depth_l1": 100000,
            "depth_l10": 500000,
            "vol_1m": 100000,
        }

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is False
        assert "1m volume too low" in reason.lower()

    def test_long_entry_5m_volume_too_low(self):
        """Test long entry rejection when 5m volume is too low."""
        filters = MarketConditionFilters(
            min_volume_1m=Decimal("100000"), min_volume_5m=Decimal("1000000")
        )

        market_snapshot = {
            "spread_bps": 5,
            "depth_l1": 100000,
            "depth_l10": 500000,
            "vol_1m": 150000,
            "vol_5m": 500000,
        }

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is False
        assert "5m volume too low" in reason.lower()

    def test_long_entry_rsi_too_high(self):
        """Test long entry rejection when RSI is too high."""
        filters = MarketConditionFilters(
            rsi_overbought=Decimal("70"), require_rsi_confirmation=True
        )

        market_snapshot = {"spread_bps": 5}

        allowed, reason = filters.should_allow_long_entry(market_snapshot, rsi=Decimal("75"))

        assert allowed is False
        assert "rsi too high" in reason.lower()

    def test_long_entry_rsi_acceptable(self):
        """Test long entry allowed when RSI is acceptable."""
        filters = MarketConditionFilters(
            rsi_overbought=Decimal("70"), require_rsi_confirmation=True
        )

        market_snapshot = {"spread_bps": 5}

        allowed, reason = filters.should_allow_long_entry(market_snapshot, rsi=Decimal("50"))

        assert allowed is True

    def test_long_entry_rsi_not_required(self):
        """Test long entry when RSI confirmation is not required."""
        filters = MarketConditionFilters(require_rsi_confirmation=False)

        market_snapshot = {"spread_bps": 5}

        allowed, reason = filters.should_allow_long_entry(market_snapshot, rsi=Decimal("80"))

        assert allowed is True

    def test_long_entry_no_filters_enabled(self):
        """Test long entry with no filters enabled."""
        filters = MarketConditionFilters()

        market_snapshot = {}

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is True

    def test_long_entry_missing_market_data(self):
        """Test long entry with missing market data."""
        filters = MarketConditionFilters(
            max_spread_bps=Decimal("20"), min_depth_l1=Decimal("50000")
        )

        market_snapshot = {}  # No data

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        # Should pass since data is missing (defaults to 0)
        assert allowed is False

    def test_long_entry_at_exact_threshold(self):
        """Test long entry at exact threshold values."""
        filters = MarketConditionFilters(
            max_spread_bps=Decimal("10"), min_depth_l1=Decimal("50000")
        )

        # Exactly at spread limit
        market_snapshot = {"spread_bps": 10, "depth_l1": 50000}

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        # Should reject (> not >=)
        assert allowed is False


# ============================================================================
# Test: Short Entry Filtering
# ============================================================================


class TestShortEntryFiltering:
    """Test short entry filtering logic."""

    def test_short_entry_all_conditions_pass(self):
        """Test short entry when all conditions pass."""
        filters = MarketConditionFilters(
            max_spread_bps=Decimal("20"), min_depth_l1=Decimal("50000")
        )

        market_snapshot = {"spread_bps": 15, "depth_l1": 60000}

        allowed, reason = filters.should_allow_short_entry(market_snapshot)

        assert allowed is True

    def test_short_entry_inherits_long_filters(self):
        """Test that short entry inherits long entry filters."""
        filters = MarketConditionFilters(max_spread_bps=Decimal("10"))

        market_snapshot = {"spread_bps": 15}

        allowed, reason = filters.should_allow_short_entry(market_snapshot)

        assert allowed is False
        assert "spread too wide" in reason.lower()

    def test_short_entry_rsi_too_low(self):
        """Test short entry rejection when RSI is too low."""
        filters = MarketConditionFilters(rsi_oversold=Decimal("30"), require_rsi_confirmation=True)

        market_snapshot = {"spread_bps": 5}

        allowed, reason = filters.should_allow_short_entry(market_snapshot, rsi=Decimal("25"))

        assert allowed is False
        assert "rsi too low" in reason.lower()

    def test_short_entry_rsi_acceptable(self):
        """Test short entry allowed when RSI is acceptable."""
        filters = MarketConditionFilters(rsi_oversold=Decimal("30"), require_rsi_confirmation=True)

        market_snapshot = {"spread_bps": 5}

        allowed, reason = filters.should_allow_short_entry(market_snapshot, rsi=Decimal("50"))

        assert allowed is True

    def test_short_entry_rsi_not_required(self):
        """Test short entry when RSI confirmation is not required."""
        filters = MarketConditionFilters(require_rsi_confirmation=False)

        market_snapshot = {"spread_bps": 5}

        allowed, reason = filters.should_allow_short_entry(market_snapshot, rsi=Decimal("20"))

        assert allowed is True

    def test_short_entry_no_rsi_provided(self):
        """Test short entry when no RSI is provided."""
        filters = MarketConditionFilters(require_rsi_confirmation=True)

        market_snapshot = {"spread_bps": 5}

        allowed, reason = filters.should_allow_short_entry(market_snapshot, rsi=None)

        assert allowed is True


# ============================================================================
# Test: Factory Functions
# ============================================================================


class TestFactoryFunctions:
    """Test filter factory functions."""

    def test_create_conservative_filters(self):
        """Test conservative filter factory."""
        filters = create_conservative_filters()

        assert filters.max_spread_bps == Decimal("10")
        assert filters.min_depth_l1 == Decimal("50000")
        assert filters.min_depth_l10 == Decimal("200000")
        assert filters.min_volume_1m == Decimal("100000")
        assert filters.require_rsi_confirmation is True

    def test_create_aggressive_filters(self):
        """Test aggressive filter factory."""
        filters = create_aggressive_filters()

        assert filters.max_spread_bps == Decimal("25")
        assert filters.min_depth_l1 == Decimal("20000")
        assert filters.min_depth_l10 == Decimal("100000")
        assert filters.min_volume_1m == Decimal("50000")
        assert filters.require_rsi_confirmation is False

    def test_conservative_stricter_than_aggressive(self):
        """Test that conservative filters are stricter than aggressive."""
        conservative = create_conservative_filters()
        aggressive = create_aggressive_filters()

        # Conservative should have tighter limits
        assert conservative.max_spread_bps < aggressive.max_spread_bps
        assert conservative.min_depth_l1 > aggressive.min_depth_l1
        assert conservative.min_depth_l10 > aggressive.min_depth_l10
        assert conservative.min_volume_1m > aggressive.min_volume_1m
        assert conservative.require_rsi_confirmation is True
        assert aggressive.require_rsi_confirmation is False


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_spread(self):
        """Test with zero spread."""
        filters = MarketConditionFilters(max_spread_bps=Decimal("10"))

        market_snapshot = {"spread_bps": 0}

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is True

    def test_negative_spread(self):
        """Test with negative spread (invalid but handled)."""
        filters = MarketConditionFilters(max_spread_bps=Decimal("10"))

        market_snapshot = {"spread_bps": -5}

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is True

    def test_very_large_spread(self):
        """Test with very large spread."""
        filters = MarketConditionFilters(max_spread_bps=Decimal("10"))

        market_snapshot = {"spread_bps": 10000}

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is False

    def test_zero_depth(self):
        """Test with zero depth."""
        filters = MarketConditionFilters(min_depth_l1=Decimal("50000"))

        market_snapshot = {"spread_bps": 5, "depth_l1": 0}

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is False

    def test_zero_volume(self):
        """Test with zero volume."""
        filters = MarketConditionFilters(min_volume_1m=Decimal("100000"))

        market_snapshot = {"spread_bps": 5, "vol_1m": 0}

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is False

    def test_rsi_boundary_values(self):
        """Test RSI at boundary values."""
        filters = MarketConditionFilters(
            rsi_overbought=Decimal("70"),
            rsi_oversold=Decimal("30"),
            require_rsi_confirmation=True,
        )

        market_snapshot = {"spread_bps": 5}

        # RSI exactly at overbought threshold
        allowed1, _ = filters.should_allow_long_entry(market_snapshot, rsi=Decimal("70"))
        assert allowed1 is True  # <= not <

        # RSI slightly above threshold
        allowed2, _ = filters.should_allow_long_entry(market_snapshot, rsi=Decimal("70.1"))
        assert allowed2 is False

        # RSI exactly at oversold threshold
        allowed3, _ = filters.should_allow_short_entry(market_snapshot, rsi=Decimal("30"))
        assert allowed3 is True  # >= not >

        # RSI slightly below threshold
        allowed4, _ = filters.should_allow_short_entry(market_snapshot, rsi=Decimal("29.9"))
        assert allowed4 is False

    def test_rsi_extreme_values(self):
        """Test RSI with extreme values."""
        filters = MarketConditionFilters(require_rsi_confirmation=True)

        market_snapshot = {"spread_bps": 5}

        # RSI = 0
        allowed1, _ = filters.should_allow_long_entry(market_snapshot, rsi=Decimal("0"))
        assert allowed1 is True

        # RSI = 100
        allowed2, _ = filters.should_allow_long_entry(market_snapshot, rsi=Decimal("100"))
        assert allowed2 is False

        # RSI > 100 (invalid)
        allowed3, _ = filters.should_allow_long_entry(market_snapshot, rsi=Decimal("150"))
        assert allowed3 is False

    def test_custom_rsi_thresholds(self):
        """Test custom RSI thresholds."""
        filters = MarketConditionFilters(
            rsi_overbought=Decimal("80"),
            rsi_oversold=Decimal("20"),
            require_rsi_confirmation=True,
        )

        market_snapshot = {"spread_bps": 5}

        # RSI 75 should be ok for long (below 80)
        allowed1, _ = filters.should_allow_long_entry(market_snapshot, rsi=Decimal("75"))
        assert allowed1 is True

        # RSI 25 should be ok for short (above 20)
        allowed2, _ = filters.should_allow_short_entry(market_snapshot, rsi=Decimal("25"))
        assert allowed2 is True

    def test_all_filters_at_limits(self):
        """Test all filters at exact limit values."""
        filters = MarketConditionFilters(
            max_spread_bps=Decimal("20"),
            min_depth_l1=Decimal("50000"),
            min_depth_l10=Decimal("200000"),
            min_volume_1m=Decimal("100000"),
            min_volume_5m=Decimal("500000"),
        )

        # All at exact thresholds
        market_snapshot = {
            "spread_bps": 20,
            "depth_l1": 50000,
            "depth_l10": 200000,
            "vol_1m": 100000,
            "vol_5m": 500000,
        }

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        # spread should fail (> not >=), others should pass (< not <=)
        assert allowed is False

    def test_float_vs_decimal_comparison(self):
        """Test that float values work with Decimal comparisons."""
        filters = MarketConditionFilters(max_spread_bps=Decimal("10"))

        market_snapshot = {"spread_bps": 5.5}  # float

        allowed, reason = filters.should_allow_long_entry(market_snapshot)

        assert allowed is True


# ============================================================================
# Test: Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_high_quality_market_conservative(self):
        """Test high-quality market passes conservative filters."""
        filters = create_conservative_filters()

        market_snapshot = {
            "spread_bps": 5,
            "depth_l1": 100000,
            "depth_l10": 300000,
            "vol_1m": 200000,
            "vol_5m": 1000000,
        }

        allowed, _ = filters.should_allow_long_entry(market_snapshot, rsi=Decimal("50"))

        assert allowed is True

    def test_low_quality_market_aggressive(self):
        """Test low-quality market rejected even by aggressive filters."""
        filters = create_aggressive_filters()

        market_snapshot = {
            "spread_bps": 50,  # Very wide
            "depth_l1": 5000,  # Very shallow
            "depth_l10": 10000,
            "vol_1m": 10000,  # Very low volume
        }

        allowed, _ = filters.should_allow_long_entry(market_snapshot)

        assert allowed is False

    def test_marginal_market_aggressive_pass(self):
        """Test marginal market passes aggressive but not conservative."""
        aggressive = create_aggressive_filters()
        conservative = create_conservative_filters()

        market_snapshot = {
            "spread_bps": 20,
            "depth_l1": 30000,
            "depth_l10": 150000,
            "vol_1m": 75000,
        }

        aggressive_allowed, _ = aggressive.should_allow_long_entry(market_snapshot)
        conservative_allowed, _ = conservative.should_allow_long_entry(market_snapshot)

        assert aggressive_allowed is True
        assert conservative_allowed is False

    def test_complete_filtering_workflow(self):
        """Test complete filtering workflow for both long and short."""
        filters = MarketConditionFilters(
            max_spread_bps=Decimal("15"),
            min_depth_l1=Decimal("50000"),
            min_volume_1m=Decimal("100000"),
            rsi_overbought=Decimal("70"),
            rsi_oversold=Decimal("30"),
            require_rsi_confirmation=True,
        )

        market_snapshot = {
            "spread_bps": 10,
            "depth_l1": 60000,
            "vol_1m": 150000,
        }

        # Neutral RSI - both should pass
        long_allowed, _ = filters.should_allow_long_entry(market_snapshot, rsi=Decimal("50"))
        short_allowed, _ = filters.should_allow_short_entry(market_snapshot, rsi=Decimal("50"))

        assert long_allowed is True
        assert short_allowed is True

        # High RSI - long should fail
        long_allowed, _ = filters.should_allow_long_entry(market_snapshot, rsi=Decimal("75"))
        assert long_allowed is False

        # Low RSI - short should fail
        short_allowed, _ = filters.should_allow_short_entry(market_snapshot, rsi=Decimal("25"))
        assert short_allowed is False

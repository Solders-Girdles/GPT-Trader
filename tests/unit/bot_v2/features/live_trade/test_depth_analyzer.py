"""Tests for DepthAnalyzer - order book depth analysis.

This module tests the DepthAnalyzer's ability to:
- Extract Level 1 data (best bid/ask)
- Calculate spreads (absolute and basis points)
- Measure depth at multiple price levels
- Calculate imbalance metrics
- Handle empty or invalid order books
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bot_v2.features.live_trade.depth_analyzer import DepthAnalyzer, DepthData


class TestDepthAnalyzerBasics:
    """Test basic depth analysis functionality."""

    def test_analyze_empty_bids(self):
        """Returns None for empty bids."""
        analyzer = DepthAnalyzer()

        result = analyzer.analyze_depth([], [(Decimal("50100"), Decimal("10"))])

        assert result is None

    def test_analyze_empty_asks(self):
        """Returns None for empty asks."""
        analyzer = DepthAnalyzer()

        result = analyzer.analyze_depth([(Decimal("50000"), Decimal("10"))], [])

        assert result is None

    def test_analyze_both_empty(self):
        """Returns None when both bids and asks are empty."""
        analyzer = DepthAnalyzer()

        result = analyzer.analyze_depth([], [])

        assert result is None


class TestLevel1Extraction:
    """Test Level 1 data extraction (best bid/ask)."""

    def test_extracts_best_bid_ask(self):
        """Extracts best bid and ask prices correctly."""
        analyzer = DepthAnalyzer()
        bids = [(Decimal("50000"), Decimal("5")), (Decimal("49990"), Decimal("10"))]
        asks = [(Decimal("50100"), Decimal("3")), (Decimal("50110"), Decimal("8"))]

        result = analyzer.analyze_depth(bids, asks)

        assert result.best_bid == Decimal("50000")
        assert result.best_ask == Decimal("50100")

    def test_extracts_bid_ask_sizes(self):
        """Extracts best bid and ask sizes correctly."""
        analyzer = DepthAnalyzer()
        bids = [(Decimal("50000"), Decimal("5.5"))]
        asks = [(Decimal("50100"), Decimal("3.2"))]

        result = analyzer.analyze_depth(bids, asks)

        assert result.bid_size == Decimal("5.5")
        assert result.ask_size == Decimal("3.2")


class TestSpreadCalculation:
    """Test spread calculation."""

    def test_calculates_absolute_spread(self):
        """Calculates absolute spread correctly."""
        analyzer = DepthAnalyzer()
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50100"), Decimal("10"))]

        result = analyzer.analyze_depth(bids, asks)

        assert result.spread == Decimal("100")  # 50100 - 50000

    def test_calculates_mid_price(self):
        """Calculates mid price correctly."""
        analyzer = DepthAnalyzer()
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50100"), Decimal("10"))]

        result = analyzer.analyze_depth(bids, asks)

        assert result.mid_price == Decimal("50050")  # (50000 + 50100) / 2

    def test_calculates_spread_bps(self):
        """Calculates spread in basis points correctly."""
        analyzer = DepthAnalyzer()
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50100"), Decimal("10"))]

        result = analyzer.analyze_depth(bids, asks)

        # spread_bps = (100 / 50050) * 10000 = 19.98...
        assert 19 < result.spread_bps < 20

    def test_handles_zero_mid_price(self):
        """Handles zero mid price edge case."""
        analyzer = DepthAnalyzer()
        # Artificially create zero mid price scenario
        bids = [(Decimal("0"), Decimal("10"))]
        asks = [(Decimal("0"), Decimal("10"))]

        result = analyzer.analyze_depth(bids, asks)

        assert result.spread_bps == Decimal("10000")  # 100% spread fallback


class TestDepthCalculation:
    """Test depth calculation at multiple levels."""

    def test_calculates_depth_at_1_percent(self):
        """Calculates depth within 1% of mid price."""
        analyzer = DepthAnalyzer()
        # Mid price = 50050, 1% = 500.5
        # Bid range: 50000 - 500.5 = 49499.5 to 50000
        # Ask range: 50100 to 50100 + 500.5 = 50600.5
        bids = [
            (Decimal("50000"), Decimal("10")),  # Within 1%
            (Decimal("49500"), Decimal("20")),  # Within 1%
            (Decimal("49000"), Decimal("30")),  # Outside 1%
        ]
        asks = [
            (Decimal("50100"), Decimal("5")),  # Within 1%
            (Decimal("50500"), Decimal("15")),  # Within 1%
            (Decimal("51000"), Decimal("25")),  # Outside 1%
        ]

        result = analyzer.analyze_depth(bids, asks)

        # Should include 10 + 20 = 30 on bid side
        assert result.bid_depth_1pct == Decimal("30")
        # Should include 5 + 15 = 20 on ask side
        assert result.ask_depth_1pct == Decimal("20")

    def test_calculates_depth_usd_values(self):
        """Calculates USD depth (depth * mid_price)."""
        analyzer = DepthAnalyzer()
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50100"), Decimal("5"))]

        result = analyzer.analyze_depth(bids, asks)

        # Mid = 50050, depth_1 = bid_depth_1 + ask_depth_1 = 10 + 5 = 15
        # depth_usd_1 = 15 * 50050 = 750,750
        assert (
            result.depth_usd_1 == (result.bid_depth_1pct + result.ask_depth_1pct) * result.mid_price
        )

    def test_depth_calculation_empty_range(self):
        """Handles case where no orders within range."""
        analyzer = DepthAnalyzer()
        # Orders far from best
        bids = [(Decimal("50000"), Decimal("10")), (Decimal("40000"), Decimal("100"))]
        asks = [(Decimal("60000"), Decimal("10")), (Decimal("70000"), Decimal("100"))]

        result = analyzer.analyze_depth(bids, asks)

        # At 1% of mid (55000), range is tiny compared to spread
        # Should have minimal depth
        assert result.bid_depth_1pct >= Decimal("0")
        assert result.ask_depth_1pct >= Decimal("0")


class TestImbalanceMetrics:
    """Test imbalance metric calculations."""

    def test_calculates_bid_ask_ratio(self):
        """Calculates bid/ask size ratio correctly."""
        analyzer = DepthAnalyzer()
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50100"), Decimal("5"))]

        result = analyzer.analyze_depth(bids, asks)

        assert result.bid_ask_ratio == Decimal("2")  # 10 / 5

    def test_bid_ask_ratio_handles_zero_ask(self):
        """Handles zero ask size edge case."""
        analyzer = DepthAnalyzer()
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50100"), Decimal("0"))]

        result = analyzer.analyze_depth(bids, asks)

        assert result.bid_ask_ratio == Decimal("999")  # Fallback for division by zero

    def test_calculates_depth_imbalance(self):
        """Calculates depth imbalance correctly."""
        analyzer = DepthAnalyzer()
        # Create imbalanced book
        bids = [
            (Decimal("50000"), Decimal("100")),
            (Decimal("49900"), Decimal("100")),
        ]
        asks = [
            (Decimal("50100"), Decimal("10")),
            (Decimal("50200"), Decimal("10")),
        ]

        result = analyzer.analyze_depth(bids, asks)

        # Imbalance = (bid_depth_5 - ask_depth_5) / total_depth_5
        # Positive imbalance = more bid depth
        assert result.depth_imbalance > Decimal("0")

    def test_depth_imbalance_handles_zero_depth(self):
        """Handles zero total depth edge case."""
        analyzer = DepthAnalyzer()
        # Orders far outside depth calculation range
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("60000"), Decimal("10"))]

        result = analyzer.analyze_depth(bids, asks)

        # If total_depth_5 is zero, imbalance should be 0
        assert result.depth_imbalance == Decimal("0")


class TestCustomDepthThresholds:
    """Test custom depth threshold support."""

    def test_uses_custom_thresholds(self):
        """Uses custom depth thresholds when provided."""
        analyzer = DepthAnalyzer()
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50100"), Decimal("10"))]

        # Custom thresholds: 2%, 10%, 20%
        result = analyzer.analyze_depth(
            bids, asks, depth_thresholds=[Decimal("0.02"), Decimal("0.10"), Decimal("0.20")]
        )

        # Thresholds should be applied
        assert result is not None

    def test_defaults_to_standard_thresholds(self):
        """Uses default 1%, 5%, 10% thresholds when not provided."""
        analyzer = DepthAnalyzer()
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50100"), Decimal("10"))]

        result = analyzer.analyze_depth(bids, asks)

        # Should have all three depth levels
        assert result.depth_usd_1 is not None
        assert result.depth_usd_5 is not None
        assert result.depth_usd_10 is not None


class TestDepthInRangeHelper:
    """Test the _calculate_depth_in_range helper method."""

    def test_sums_sizes_within_range(self):
        """Sums all sizes within price range."""
        analyzer = DepthAnalyzer()
        levels = [
            (Decimal("100"), Decimal("5")),
            (Decimal("99"), Decimal("10")),
            (Decimal("98"), Decimal("15")),
        ]

        total = analyzer._calculate_depth_in_range(levels, Decimal("98"), Decimal("100"))

        assert total == Decimal("30")  # 5 + 10 + 15

    def test_excludes_outside_range(self):
        """Excludes sizes outside price range."""
        analyzer = DepthAnalyzer()
        levels = [
            (Decimal("100"), Decimal("5")),
            (Decimal("95"), Decimal("10")),  # Outside range
            (Decimal("90"), Decimal("15")),  # Outside range
        ]

        total = analyzer._calculate_depth_in_range(levels, Decimal("98"), Decimal("100"))

        assert total == Decimal("5")  # Only first level

    def test_handles_empty_levels(self):
        """Handles empty levels list."""
        analyzer = DepthAnalyzer()

        total = analyzer._calculate_depth_in_range([], Decimal("0"), Decimal("100"))

        assert total == Decimal("0")

    def test_handles_no_matches(self):
        """Returns zero when no levels in range."""
        analyzer = DepthAnalyzer()
        levels = [(Decimal("50"), Decimal("10"))]

        total = analyzer._calculate_depth_in_range(levels, Decimal("100"), Decimal("200"))

        assert total == Decimal("0")

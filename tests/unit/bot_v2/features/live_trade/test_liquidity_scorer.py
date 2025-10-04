"""Tests for LiquidityScorer - liquidity quality scoring.

This module tests the LiquidityScorer's ability to:
- Score spread quality (tight vs wide spreads)
- Score market depth (deep vs shallow)
- Score order flow imbalance (balanced vs imbalanced)
- Calculate composite liquidity scores
- Map scores to liquidity conditions
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bot_v2.features.live_trade.liquidity_models import LiquidityCondition
from bot_v2.features.live_trade.liquidity_scorer import LiquidityScorer


@pytest.fixture
def scorer():
    """Create LiquidityScorer instance."""
    return LiquidityScorer()


class TestSpreadScoring:
    """Test spread quality scoring."""

    def test_scores_excellent_spread(self, scorer):
        """Very tight spread (≤1bps) scores 100."""
        score = scorer.score_spread(Decimal("0.5"))
        assert score == Decimal("100")

        score = scorer.score_spread(Decimal("1"))
        assert score == Decimal("100")

    def test_scores_good_spread(self, scorer):
        """Good spread (≤5bps) scores 80."""
        score = scorer.score_spread(Decimal("3"))
        assert score == Decimal("80")

        score = scorer.score_spread(Decimal("5"))
        assert score == Decimal("80")

    def test_scores_moderate_spread(self, scorer):
        """Moderate spread (≤10bps) scores 60."""
        score = scorer.score_spread(Decimal("8"))
        assert score == Decimal("60")

        score = scorer.score_spread(Decimal("10"))
        assert score == Decimal("60")

    def test_scores_fair_spread(self, scorer):
        """Fair spread (≤20bps) scores 40."""
        score = scorer.score_spread(Decimal("15"))
        assert score == Decimal("40")

    def test_scores_poor_spread(self, scorer):
        """Poor spread (≤50bps) scores 20."""
        score = scorer.score_spread(Decimal("30"))
        assert score == Decimal("20")

    def test_scores_very_wide_spread(self, scorer):
        """Very wide spread (>50bps) scores 0."""
        score = scorer.score_spread(Decimal("100"))
        assert score == Decimal("0")

        score = scorer.score_spread(Decimal("1000"))
        assert score == Decimal("0")


class TestDepthScoring:
    """Test market depth scoring."""

    def test_scores_deep_liquidity(self, scorer):
        """Deep liquidity (≥$10k) scores 100."""
        score = scorer.score_depth(Decimal("10000"), Decimal("50000"))
        assert score == Decimal("100")

        score = scorer.score_depth(Decimal("50000"), Decimal("50000"))
        assert score == Decimal("100")

    def test_scores_moderate_liquidity(self, scorer):
        """Moderate liquidity scores proportionally."""
        # $5k depth = 50% of $10k baseline = score of 50
        score = scorer.score_depth(Decimal("5000"), Decimal("50000"))
        assert score == Decimal("50")

        # $2.5k depth = 25% of baseline = score of 25
        score = scorer.score_depth(Decimal("2500"), Decimal("50000"))
        assert score == Decimal("25")

    def test_scores_shallow_liquidity(self, scorer):
        """Shallow liquidity scores proportionally low."""
        score = scorer.score_depth(Decimal("1000"), Decimal("50000"))
        assert score == Decimal("10")

    def test_scores_zero_depth(self, scorer):
        """Zero depth scores 0."""
        score = scorer.score_depth(Decimal("0"), Decimal("50000"))
        assert score == Decimal("0")

    def test_caps_depth_score_at_100(self, scorer):
        """Depth score caps at 100 even for very deep liquidity."""
        score = scorer.score_depth(Decimal("100000"), Decimal("50000"))
        assert score == Decimal("100")


class TestImbalanceScoring:
    """Test order flow imbalance scoring."""

    def test_scores_perfect_balance(self, scorer):
        """Zero imbalance scores 100 (perfect balance)."""
        score = scorer.score_imbalance(Decimal("0"))
        assert score == Decimal("100")

    def test_scores_slight_imbalance(self, scorer):
        """Slight imbalance (0.1) scores 80."""
        score = scorer.score_imbalance(Decimal("0.1"))
        assert score == Decimal("80")

    def test_scores_moderate_imbalance(self, scorer):
        """Moderate imbalance (0.3) scores 40."""
        score = scorer.score_imbalance(Decimal("0.3"))
        assert score == Decimal("40")

    def test_scores_high_imbalance(self, scorer):
        """High imbalance (≥0.5) scores 0."""
        score = scorer.score_imbalance(Decimal("0.5"))
        assert score == Decimal("0")

        score = scorer.score_imbalance(Decimal("1.0"))
        assert score == Decimal("0")

    def test_handles_extreme_imbalance(self, scorer):
        """Extreme imbalance doesn't produce negative scores."""
        score = scorer.score_imbalance(Decimal("10"))
        assert score == Decimal("0")


class TestCompositeScoring:
    """Test composite liquidity score calculation."""

    def test_calculates_composite_from_components(self, scorer):
        """Composite score is average of component scores."""
        # Perfect conditions: tight spread, deep liquidity, balanced
        composite = scorer.calculate_composite_score(
            spread_bps=Decimal("1"),  # 100
            depth_usd_1=Decimal("10000"),  # 100
            depth_usd_5=Decimal("20000"),  # 100
            depth_imbalance=Decimal("0"),  # 100
            mid_price=Decimal("50000"),
        )

        # Average of [100, 100, 100, 100] = 100
        assert composite == Decimal("100")

    def test_composite_with_mixed_components(self, scorer):
        """Composite score averages mixed component scores."""
        # Mixed: good spread (80), moderate depth (50, 100), slight imbalance (80)
        composite = scorer.calculate_composite_score(
            spread_bps=Decimal("3"),  # 80
            depth_usd_1=Decimal("5000"),  # 50
            depth_usd_5=Decimal("10000"),  # 100
            depth_imbalance=Decimal("0.1"),  # 80
            mid_price=Decimal("50000"),
        )

        # Average of [80, 50, 100, 80] = 77.5
        assert composite == Decimal("77.5")

    def test_composite_with_poor_components(self, scorer):
        """Composite score handles poor conditions."""
        # Poor: wide spread (0), shallow depth (10, 20), high imbalance (0)
        composite = scorer.calculate_composite_score(
            spread_bps=Decimal("100"),  # 0
            depth_usd_1=Decimal("1000"),  # 10
            depth_usd_5=Decimal("2000"),  # 20
            depth_imbalance=Decimal("0.8"),  # 0
            mid_price=Decimal("50000"),
        )

        # Average of [0, 10, 20, 0] = 7.5
        assert composite == Decimal("7.5")


class TestConditionDetermination:
    """Test liquidity condition mapping from scores."""

    def test_determines_excellent_condition(self, scorer):
        """Score ≥80 maps to EXCELLENT."""
        condition = scorer.determine_condition(Decimal("80"))
        assert condition == LiquidityCondition.EXCELLENT

        condition = scorer.determine_condition(Decimal("100"))
        assert condition == LiquidityCondition.EXCELLENT

    def test_determines_good_condition(self, scorer):
        """Score ≥60 and <80 maps to GOOD."""
        condition = scorer.determine_condition(Decimal("60"))
        assert condition == LiquidityCondition.GOOD

        condition = scorer.determine_condition(Decimal("75"))
        assert condition == LiquidityCondition.GOOD

    def test_determines_fair_condition(self, scorer):
        """Score ≥40 and <60 maps to FAIR."""
        condition = scorer.determine_condition(Decimal("40"))
        assert condition == LiquidityCondition.FAIR

        condition = scorer.determine_condition(Decimal("55"))
        assert condition == LiquidityCondition.FAIR

    def test_determines_poor_condition(self, scorer):
        """Score ≥20 and <40 maps to POOR."""
        condition = scorer.determine_condition(Decimal("20"))
        assert condition == LiquidityCondition.POOR

        condition = scorer.determine_condition(Decimal("35"))
        assert condition == LiquidityCondition.POOR

    def test_determines_critical_condition(self, scorer):
        """Score <20 maps to CRITICAL."""
        condition = scorer.determine_condition(Decimal("19"))
        assert condition == LiquidityCondition.CRITICAL

        condition = scorer.determine_condition(Decimal("0"))
        assert condition == LiquidityCondition.CRITICAL


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_handles_zero_values(self, scorer):
        """Handles zero values gracefully."""
        composite = scorer.calculate_composite_score(
            spread_bps=Decimal("0"),
            depth_usd_1=Decimal("0"),
            depth_usd_5=Decimal("0"),
            depth_imbalance=Decimal("0"),
            mid_price=Decimal("0"),
        )

        # spread=100, depth_1=0, depth_5=0, imbalance=100
        # Average = (100 + 0 + 0 + 100) / 4 = 50
        assert composite == Decimal("50")

    def test_handles_boundary_scores(self, scorer):
        """Handles boundary values for condition thresholds."""
        # Test exact boundaries
        assert scorer.determine_condition(Decimal("79.99")) == LiquidityCondition.GOOD
        assert scorer.determine_condition(Decimal("80.00")) == LiquidityCondition.EXCELLENT
        assert scorer.determine_condition(Decimal("59.99")) == LiquidityCondition.FAIR
        assert scorer.determine_condition(Decimal("60.00")) == LiquidityCondition.GOOD

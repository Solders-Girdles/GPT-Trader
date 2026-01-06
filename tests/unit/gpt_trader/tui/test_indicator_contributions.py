"""Tests for indicator contributions functionality."""

from gpt_trader.tui.types import DecisionData, IndicatorContribution


class TestIndicatorContribution:
    """Tests for IndicatorContribution dataclass."""

    def test_is_bullish_positive_contribution(self):
        """Test is_bullish returns True for positive contribution."""
        contrib = IndicatorContribution(name="RSI", value=30.0, contribution=0.3)
        assert contrib.is_bullish is True
        assert contrib.is_bearish is False

    def test_is_bearish_negative_contribution(self):
        """Test is_bearish returns True for negative contribution."""
        contrib = IndicatorContribution(name="MACD", value=-25.0, contribution=-0.4)
        assert contrib.is_bullish is False
        assert contrib.is_bearish is True

    def test_neutral_contribution(self):
        """Test neutral contribution is neither bullish nor bearish."""
        contrib = IndicatorContribution(name="Trend", value=0.0, contribution=0.0)
        assert contrib.is_bullish is False
        assert contrib.is_bearish is False

    def test_abs_contribution(self):
        """Test abs_contribution returns absolute value."""
        positive = IndicatorContribution(name="A", value=0.0, contribution=0.5)
        negative = IndicatorContribution(name="B", value=0.0, contribution=-0.5)

        assert positive.abs_contribution == 0.5
        assert negative.abs_contribution == 0.5


class TestDecisionDataContributions:
    """Tests for DecisionData contributions methods."""

    def test_top_contributors_sorted_by_magnitude(self):
        """Test top_contributors returns top 3 by absolute contribution."""
        decision = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            reason="Test",
            confidence=0.75,
            contributions=[
                IndicatorContribution(name="RSI", value=30.0, contribution=0.2),
                IndicatorContribution(name="MACD", value=10.0, contribution=-0.5),
                IndicatorContribution(name="Trend", value=0.0, contribution=0.3),
                IndicatorContribution(name="Volume", value=100.0, contribution=0.1),
            ],
        )

        top = decision.top_contributors
        assert len(top) == 3
        # Should be sorted by abs value: MACD (0.5), Trend (0.3), RSI (0.2)
        assert top[0].name == "MACD"
        assert top[1].name == "Trend"
        assert top[2].name == "RSI"

    def test_top_contributors_less_than_three(self):
        """Test top_contributors with fewer than 3 contributions."""
        decision = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            reason="Test",
            confidence=0.75,
            contributions=[
                IndicatorContribution(name="RSI", value=30.0, contribution=0.2),
            ],
        )

        top = decision.top_contributors
        assert len(top) == 1
        assert top[0].name == "RSI"

    def test_bullish_contributors(self):
        """Test bullish_contributors filters positive contributions."""
        decision = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            reason="Test",
            confidence=0.75,
            contributions=[
                IndicatorContribution(name="RSI", value=30.0, contribution=0.2),
                IndicatorContribution(name="MACD", value=10.0, contribution=-0.5),
                IndicatorContribution(name="Trend", value=0.0, contribution=0.3),
            ],
        )

        bullish = decision.bullish_contributors
        assert len(bullish) == 2
        assert all(c.is_bullish for c in bullish)

    def test_bearish_contributors(self):
        """Test bearish_contributors filters negative contributions."""
        decision = DecisionData(
            symbol="BTC-USD",
            action="SELL",
            reason="Test",
            confidence=0.75,
            contributions=[
                IndicatorContribution(name="RSI", value=70.0, contribution=-0.3),
                IndicatorContribution(name="MACD", value=-10.0, contribution=-0.5),
                IndicatorContribution(name="Trend", value=0.0, contribution=0.1),
            ],
        )

        bearish = decision.bearish_contributors
        assert len(bearish) == 2
        assert all(c.is_bearish for c in bearish)

    def test_empty_contributions(self):
        """Test methods work with empty contributions."""
        decision = DecisionData(
            symbol="BTC-USD",
            action="HOLD",
            reason="Test",
            confidence=0.5,
            contributions=[],
        )

        assert decision.top_contributors == []
        assert decision.bullish_contributors == []
        assert decision.bearish_contributors == []

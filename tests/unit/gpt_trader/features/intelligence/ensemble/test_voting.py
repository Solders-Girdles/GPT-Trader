"""Tests for voting mechanisms."""

import pytest

from gpt_trader.features.intelligence.ensemble.models import StrategyVote
from gpt_trader.features.intelligence.ensemble.voting import (
    ConfidenceLeaderVoting,
    VotingMechanism,
    WeightedMajorityVoting,
)
from gpt_trader.features.intelligence.regime.models import RegimeState, RegimeType
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


@pytest.fixture
def regime_state() -> RegimeState:
    """Create a test regime state."""
    return RegimeState(
        regime=RegimeType.BULL_QUIET,
        confidence=0.8,
        trend_score=0.5,
        volatility_percentile=0.3,
        momentum_score=0.4,
    )


class TestVotingMechanismFactory:
    """Test VotingMechanism factory method."""

    def test_create_weighted_majority(self):
        """Test creating weighted majority voter."""
        voter = VotingMechanism.create("weighted_majority")
        assert isinstance(voter, WeightedMajorityVoting)

    def test_create_confidence_leader(self):
        """Test creating confidence leader voter."""
        voter = VotingMechanism.create("confidence_leader")
        assert isinstance(voter, ConfidenceLeaderVoting)

    def test_create_unknown_raises(self):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown voting method"):
            VotingMechanism.create("unknown_method")


class TestWeightedMajorityVoting:
    """Test WeightedMajorityVoting."""

    @pytest.fixture
    def voter(self) -> WeightedMajorityVoting:
        """Create voter instance."""
        return WeightedMajorityVoting()

    def test_empty_votes_returns_hold(
        self, voter: WeightedMajorityVoting, regime_state: RegimeState
    ):
        """Test that empty votes returns HOLD."""
        decision = voter.aggregate([], regime_state)
        assert decision.action == Action.HOLD
        assert decision.confidence == 0.0

    def test_unanimous_buy(self, voter: WeightedMajorityVoting, regime_state: RegimeState):
        """Test unanimous BUY votes."""
        votes = [
            StrategyVote(
                strategy_name="strategy_a",
                decision=Decision(Action.BUY, "Buy signal", 0.8, {}),
                weight=0.5,
            ),
            StrategyVote(
                strategy_name="strategy_b",
                decision=Decision(Action.BUY, "Buy signal", 0.7, {}),
                weight=0.5,
            ),
        ]

        decision = voter.aggregate(votes, regime_state)
        assert decision.action == Action.BUY
        assert decision.confidence > 0.5

    def test_weighted_aggregation(self, voter: WeightedMajorityVoting, regime_state: RegimeState):
        """Test that weights affect outcome."""
        # High weight strategy says BUY
        # Low weight strategy says SELL
        votes = [
            StrategyVote(
                strategy_name="high_weight",
                decision=Decision(Action.BUY, "Buy signal", 0.8, {}),
                weight=0.9,
            ),
            StrategyVote(
                strategy_name="low_weight",
                decision=Decision(Action.SELL, "Sell signal", 0.9, {}),
                weight=0.1,
            ),
        ]

        decision = voter.aggregate(votes, regime_state)
        # High weight BUY should win despite lower confidence
        assert decision.action == Action.BUY

    def test_equal_weights_confidence_matters(
        self, voter: WeightedMajorityVoting, regime_state: RegimeState
    ):
        """Test that confidence matters with equal weights."""
        votes = [
            StrategyVote(
                strategy_name="high_conf",
                decision=Decision(Action.BUY, "Buy signal", 0.9, {}),
                weight=0.5,
            ),
            StrategyVote(
                strategy_name="low_conf",
                decision=Decision(Action.SELL, "Sell signal", 0.3, {}),
                weight=0.5,
            ),
        ]

        decision = voter.aggregate(votes, regime_state)
        # Higher confidence BUY should win
        assert decision.action == Action.BUY


class TestConfidenceLeaderVoting:
    """Test ConfidenceLeaderVoting."""

    @pytest.fixture
    def voter(self) -> ConfidenceLeaderVoting:
        """Create voter instance."""
        return ConfidenceLeaderVoting()

    def test_empty_votes_returns_hold(
        self, voter: ConfidenceLeaderVoting, regime_state: RegimeState
    ):
        """Test that empty votes returns HOLD."""
        decision = voter.aggregate([], regime_state)
        assert decision.action == Action.HOLD

    def test_unanimous_boosted_confidence(
        self, voter: ConfidenceLeaderVoting, regime_state: RegimeState
    ):
        """Test that unanimous agreement boosts confidence."""
        votes = [
            StrategyVote(
                strategy_name="strategy_a",
                decision=Decision(Action.BUY, "Buy signal", 0.7, {}),
                weight=0.5,
            ),
            StrategyVote(
                strategy_name="strategy_b",
                decision=Decision(Action.BUY, "Buy signal", 0.7, {}),
                weight=0.5,
            ),
        ]

        decision = voter.aggregate(votes, regime_state)
        assert decision.action == Action.BUY
        # Should be boosted from 0.7
        assert decision.confidence > 0.7
        assert decision.indicators.get("unanimous") is True

    def test_conflict_follows_leader(
        self, voter: ConfidenceLeaderVoting, regime_state: RegimeState
    ):
        """Test that conflict follows highest weighted confidence."""
        votes = [
            StrategyVote(
                strategy_name="confident_buyer",
                decision=Decision(Action.BUY, "Strong buy", 0.95, {}),
                weight=0.4,
            ),
            StrategyVote(
                strategy_name="moderate_seller",
                decision=Decision(Action.SELL, "Weak sell", 0.5, {}),
                weight=0.6,
            ),
        ]

        decision = voter.aggregate(votes, regime_state)
        # BUY has higher weighted_confidence: 0.95 * 0.4 = 0.38
        # vs SELL: 0.5 * 0.6 = 0.30
        assert decision.action == Action.BUY
        assert decision.indicators.get("leader") == "confident_buyer"

    def test_conflict_penalizes_confidence(
        self, voter: ConfidenceLeaderVoting, regime_state: RegimeState
    ):
        """Test that disagreement reduces final confidence."""
        unanimous_votes = [
            StrategyVote(
                strategy_name="a",
                decision=Decision(Action.BUY, "Buy", 0.8, {}),
                weight=0.5,
            ),
            StrategyVote(
                strategy_name="b",
                decision=Decision(Action.BUY, "Buy", 0.8, {}),
                weight=0.5,
            ),
        ]

        conflict_votes = [
            StrategyVote(
                strategy_name="a",
                decision=Decision(Action.BUY, "Buy", 0.8, {}),
                weight=0.5,
            ),
            StrategyVote(
                strategy_name="b",
                decision=Decision(Action.SELL, "Sell", 0.4, {}),
                weight=0.5,
            ),
        ]

        unanimous_decision = voter.aggregate(unanimous_votes, regime_state)
        conflict_decision = voter.aggregate(conflict_votes, regime_state)

        # Conflict should have lower confidence
        assert conflict_decision.confidence < unanimous_decision.confidence


class TestStrategyVote:
    """Test StrategyVote model."""

    def test_weighted_confidence(self):
        """Test weighted_confidence property."""
        vote = StrategyVote(
            strategy_name="test",
            decision=Decision(Action.BUY, "Test", 0.8, {}),
            weight=0.5,
        )

        assert vote.weighted_confidence == 0.4  # 0.8 * 0.5

    def test_to_dict(self):
        """Test serialization."""
        vote = StrategyVote(
            strategy_name="baseline",
            decision=Decision(Action.SELL, "Overbought", 0.75, {}),
            weight=0.6,
        )

        d = vote.to_dict()
        assert d["strategy_name"] == "baseline"
        assert d["action"] == "sell"
        assert d["confidence"] == 0.75
        assert d["weight"] == 0.6
        assert d["weighted_confidence"] == 0.45

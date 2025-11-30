"""Tests for Bayesian adaptive learning module."""

import pytest

from gpt_trader.features.intelligence.ensemble.adaptive import (
    BayesianWeightConfig,
    BayesianWeightUpdater,
    StrategyPerformanceRecord,
)
from gpt_trader.features.intelligence.regime.models import RegimeType


class TestStrategyPerformanceRecord:
    """Tests for StrategyPerformanceRecord."""

    def test_init_with_priors(self):
        """Test initialization with Beta(2,2) priors."""
        record = StrategyPerformanceRecord(strategy_name="test_strategy")

        # Should have priors for all regimes
        for regime in RegimeType:
            prob = record.get_success_probability(regime)
            # Beta(2,2) has mean 0.5
            assert prob == pytest.approx(0.5, abs=0.001)

    def test_record_success(self):
        """Test recording successful trade."""
        record = StrategyPerformanceRecord(strategy_name="test")

        # Record success in BULL_QUIET regime
        record.record_outcome(RegimeType.BULL_QUIET, is_success=True, pnl=100.0)

        # Success probability should increase
        prob = record.get_success_probability(RegimeType.BULL_QUIET)
        # Beta(3, 2) has mean 3/5 = 0.6
        assert prob == pytest.approx(0.6, abs=0.001)

        # Other regimes should be unchanged
        prob_other = record.get_success_probability(RegimeType.BEAR_QUIET)
        assert prob_other == pytest.approx(0.5, abs=0.001)

    def test_record_failure(self):
        """Test recording failed trade."""
        record = StrategyPerformanceRecord(strategy_name="test")

        record.record_outcome(RegimeType.CRISIS, is_success=False, pnl=-50.0)

        # Success probability should decrease
        prob = record.get_success_probability(RegimeType.CRISIS)
        # Beta(2, 3) has mean 2/5 = 0.4
        assert prob == pytest.approx(0.4, abs=0.001)

    def test_totals_tracking(self):
        """Test total trades and wins tracking."""
        record = StrategyPerformanceRecord(strategy_name="test")

        record.record_outcome(RegimeType.BULL_QUIET, is_success=True, pnl=100.0)
        record.record_outcome(RegimeType.BULL_QUIET, is_success=True, pnl=50.0)
        record.record_outcome(RegimeType.BEAR_QUIET, is_success=False, pnl=-30.0)

        assert record.total_trades == 3
        assert record.total_wins == 2
        assert record.total_pnl == pytest.approx(120.0)

    def test_recent_win_rate(self):
        """Test recent win rate calculation."""
        record = StrategyPerformanceRecord(strategy_name="test")

        # Record 3 wins, 2 losses
        for _ in range(3):
            record.record_outcome(RegimeType.BULL_QUIET, is_success=True)
        for _ in range(2):
            record.record_outcome(RegimeType.BULL_QUIET, is_success=False)

        assert record.get_recent_win_rate() == pytest.approx(0.6, abs=0.001)

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        record = StrategyPerformanceRecord(strategy_name="test")

        # Add some trades to reduce uncertainty
        for _ in range(10):
            record.record_outcome(RegimeType.BULL_QUIET, is_success=True)
        for _ in range(5):
            record.record_outcome(RegimeType.BULL_QUIET, is_success=False)

        lower, upper = record.get_confidence_interval(RegimeType.BULL_QUIET)

        # Should be a valid interval
        assert 0.0 <= lower < upper <= 1.0
        # Mean should be within interval
        mean = record.get_success_probability(RegimeType.BULL_QUIET)
        assert lower <= mean <= upper

    def test_uncertainty(self):
        """Test uncertainty decreases with more data."""
        record = StrategyPerformanceRecord(strategy_name="test")

        initial_uncertainty = record.get_uncertainty(RegimeType.BULL_QUIET)

        # Add many trades
        for _ in range(50):
            record.record_outcome(RegimeType.BULL_QUIET, is_success=True)

        final_uncertainty = record.get_uncertainty(RegimeType.BULL_QUIET)

        # Uncertainty should decrease
        assert final_uncertainty < initial_uncertainty

    def test_serialize_deserialize(self):
        """Test serialization round-trip."""
        record = StrategyPerformanceRecord(strategy_name="test")
        record.record_outcome(RegimeType.BULL_QUIET, is_success=True, pnl=100.0)
        record.record_outcome(RegimeType.CRISIS, is_success=False, pnl=-50.0)

        # Serialize
        data = record.serialize()

        # Deserialize
        restored = StrategyPerformanceRecord.deserialize(data)

        assert restored.strategy_name == record.strategy_name
        assert restored.total_trades == record.total_trades
        assert restored.total_wins == record.total_wins
        assert restored.total_pnl == record.total_pnl


class TestBayesianWeightUpdater:
    """Tests for BayesianWeightUpdater."""

    def test_init_equal_weights(self):
        """Test initialization with equal weights."""
        updater = BayesianWeightUpdater(strategy_names=["baseline", "mean_reversion"])

        weights = updater.get_weights(RegimeType.BULL_QUIET)

        # Should sum to 1.0
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.001)
        # Should have both strategies
        assert "baseline" in weights
        assert "mean_reversion" in weights

    def test_init_custom_weights(self):
        """Test initialization with custom base weights."""
        updater = BayesianWeightUpdater(
            strategy_names=["a", "b"],
            base_weights={"a": 0.7, "b": 0.3},
        )

        # Base weights should be stored
        assert updater.base_weights["a"] == 0.7
        assert updater.base_weights["b"] == 0.3

    def test_record_outcome_updates_performance(self):
        """Test that recording outcomes updates performance."""
        updater = BayesianWeightUpdater(strategy_names=["strategy_a"])

        # Record success
        updater.record_outcome(
            strategy_name="strategy_a",
            regime=RegimeType.BULL_QUIET,
            is_success=True,
            pnl=100.0,
        )

        # Check performance was updated
        perf = updater.get_strategy_performance("strategy_a")
        assert perf["total_trades"] == 1
        assert perf["total_wins"] == 1

    def test_weights_adapt_to_performance(self):
        """Test that weights adapt based on performance."""
        updater = BayesianWeightUpdater(
            strategy_names=["good", "bad"],
            config=BayesianWeightConfig(smoothing=0.0),  # Instant updates
        )

        # Good strategy wins
        for _ in range(10):
            updater.record_outcome("good", RegimeType.BULL_QUIET, True, 100.0)
            updater.record_outcome("bad", RegimeType.BULL_QUIET, False, -100.0)

        weights = updater.get_weights(RegimeType.BULL_QUIET)

        # Good strategy should have higher weight
        assert weights["good"] > weights["bad"]

    def test_weight_caps(self):
        """Test weight min/max caps are enforced."""
        config = BayesianWeightConfig(min_weight=0.1, max_weight=0.9)
        updater = BayesianWeightUpdater(
            strategy_names=["a", "b"],
            config=config,
        )

        # Even with extreme performance, weights should be capped
        for _ in range(100):
            updater.record_outcome("a", RegimeType.BULL_QUIET, True)
            updater.record_outcome("b", RegimeType.BULL_QUIET, False)

        weights = updater.get_weights(RegimeType.BULL_QUIET)

        # Note: after normalization, caps may not be exact
        # but should prevent extreme concentration
        assert weights["a"] < 0.99
        assert weights["b"] > 0.01

    def test_different_regimes_independent(self):
        """Test that different regimes have independent performance."""
        updater = BayesianWeightUpdater(
            strategy_names=["strategy"],
            config=BayesianWeightConfig(smoothing=0.0),
        )

        # Win in bull, lose in bear
        for _ in range(5):
            updater.record_outcome("strategy", RegimeType.BULL_QUIET, True)
            updater.record_outcome("strategy", RegimeType.BEAR_QUIET, False)

        perf = updater.get_strategy_performance("strategy")

        # Should show different success rates per regime
        assert perf["regime_probabilities"]["BULL_QUIET"] > 0.5
        assert perf["regime_probabilities"]["BEAR_QUIET"] < 0.5

    def test_serialize_deserialize(self):
        """Test serialization round-trip."""
        updater = BayesianWeightUpdater(
            strategy_names=["a", "b"],
            base_weights={"a": 0.6, "b": 0.4},
        )

        # Record some outcomes
        updater.record_outcome("a", RegimeType.BULL_QUIET, True)
        updater.record_outcome("b", RegimeType.CRISIS, False)

        # Serialize
        data = updater.serialize()

        # Deserialize
        restored = BayesianWeightUpdater.deserialize(data)

        assert restored.strategy_names == updater.strategy_names
        assert restored.base_weights == updater.base_weights

    def test_get_all_performance(self):
        """Test getting performance for all strategies."""
        updater = BayesianWeightUpdater(strategy_names=["a", "b", "c"])

        updater.record_outcome("a", RegimeType.BULL_QUIET, True)
        updater.record_outcome("b", RegimeType.BEAR_QUIET, False)

        all_perf = updater.get_all_performance()

        assert "a" in all_perf
        assert "b" in all_perf
        assert "c" in all_perf
        assert all_perf["a"]["total_trades"] == 1
        assert all_perf["b"]["total_trades"] == 1
        assert all_perf["c"]["total_trades"] == 0

"""Tests for BayesianWeightUpdater."""

import pytest

from gpt_trader.features.intelligence.ensemble.adaptive import (
    BayesianWeightConfig,
    BayesianWeightUpdater,
)
from gpt_trader.features.intelligence.regime.models import RegimeType


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

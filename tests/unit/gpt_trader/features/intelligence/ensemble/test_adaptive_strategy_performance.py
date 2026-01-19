"""Tests for StrategyPerformanceRecord."""

import pytest

from gpt_trader.features.intelligence.ensemble.adaptive import StrategyPerformanceRecord
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

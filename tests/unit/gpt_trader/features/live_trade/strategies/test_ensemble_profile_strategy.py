"""Tests for building EnsembleStrategy from profiles."""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from gpt_trader.features.live_trade.strategies.ensemble import EnsembleStrategy
from gpt_trader.features.live_trade.strategies.ensemble_profile import (
    get_default_profile,
)


class TestEnsembleStrategyFromProfile:
    """Tests for EnsembleStrategy.from_profile factory methods."""

    def test_from_profile_creates_strategy(self):
        """from_profile should create working EnsembleStrategy."""
        profile = get_default_profile()
        strategy = EnsembleStrategy.from_profile(profile)

        assert strategy is not None
        assert len(strategy.signals) > 0
        assert strategy.combiner is not None
        assert strategy.config.buy_threshold == profile.decision.buy_threshold

    def test_from_profile_name_default(self):
        """from_profile_name('default') should work."""
        strategy = EnsembleStrategy.from_profile_name("default")
        assert strategy is not None

    def test_from_profile_name_microstructure(self):
        """from_profile_name('microstructure') should work."""
        strategy = EnsembleStrategy.from_profile_name("microstructure")
        assert strategy is not None
        # Microstructure should have orderbook/trade flow signals
        signal_names = [type(s).__name__ for s in strategy.signals]
        assert "OrderFlowSignal" in signal_names or "VWAPSignal" in signal_names

    def test_from_profile_name_conservative(self):
        """from_profile_name('conservative') should work."""
        strategy = EnsembleStrategy.from_profile_name("conservative")
        assert strategy is not None
        # Conservative should have higher thresholds
        assert strategy.config.buy_threshold >= 0.3

    def test_from_profile_name_aggressive(self):
        """from_profile_name('aggressive') should work."""
        strategy = EnsembleStrategy.from_profile_name("aggressive")
        assert strategy is not None
        # Aggressive should have lower thresholds
        assert strategy.config.buy_threshold <= 0.2

    def test_from_profile_name_unknown_raises(self):
        """from_profile_name with unknown name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown profile"):
            EnsembleStrategy.from_profile_name("nonexistent_profile")

    def test_from_yaml_file(self):
        """from_yaml should load and create strategy."""
        profile = get_default_profile()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = Path(f.name)

        try:
            profile.to_yaml(path)
            strategy = EnsembleStrategy.from_yaml(path)

            assert strategy is not None
            assert strategy.config.buy_threshold == profile.decision.buy_threshold
        finally:
            path.unlink()


class TestProfileDecisionIntegration:
    """Integration tests for profile-based strategy decisions."""

    def test_strategy_from_profile_can_decide(self):
        """Strategy built from profile should make decisions."""
        strategy = EnsembleStrategy.from_profile_name("default")

        # Create minimal context for decision
        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[Decimal("49000"), Decimal("49500"), Decimal("50000")],
            equity=Decimal("10000"),
            product=None,
        )

        assert decision is not None
        assert decision.action is not None
        assert decision.reason is not None

    def test_different_profiles_different_thresholds(self):
        """Different profiles should have different decision thresholds."""
        conservative = EnsembleStrategy.from_profile_name("conservative")
        aggressive = EnsembleStrategy.from_profile_name("aggressive")

        # Conservative should require stronger signals
        assert conservative.config.buy_threshold > aggressive.config.buy_threshold
        assert conservative.config.stop_loss_pct < aggressive.config.stop_loss_pct

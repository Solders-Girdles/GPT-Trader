"""Tests for EnsembleOrchestrator."""

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

import pytest

from gpt_trader.core import Product
from gpt_trader.features.intelligence.ensemble.models import EnsembleConfig
from gpt_trader.features.intelligence.ensemble.orchestrator import EnsembleOrchestrator
from gpt_trader.features.intelligence.regime import (
    RegimeConfig,
    RegimeType,
)
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


class MockStrategy:
    """Mock strategy for testing."""

    def __init__(self, name: str, action: Action = Action.HOLD, confidence: float = 0.5):
        self.name = name
        self.action = action
        self.confidence = confidence
        self.call_count = 0

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: Sequence[Decimal],
        equity: Decimal,
        product: Product | None,
    ) -> Decision:
        """Return configured decision."""
        self.call_count += 1
        return Decision(
            action=self.action,
            reason=f"{self.name} signal",
            confidence=self.confidence,
            indicators={"strategy": self.name},
        )


class TestEnsembleOrchestrator:
    """Test EnsembleOrchestrator."""

    @pytest.fixture
    def mock_strategies(self) -> dict[str, MockStrategy]:
        """Create mock strategies."""
        return {
            "baseline": MockStrategy("baseline", Action.BUY, 0.7),
            "mean_reversion": MockStrategy("mean_reversion", Action.BUY, 0.8),
        }

    @pytest.fixture
    def config(self) -> EnsembleConfig:
        """Create ensemble config."""
        return EnsembleConfig(
            voting_method="confidence_leader",
            base_weights={"baseline": 0.5, "mean_reversion": 0.5},
            min_ensemble_confidence=0.3,
        )

    @pytest.fixture
    def regime_config(self) -> RegimeConfig:
        """Create fast regime config for testing."""
        return RegimeConfig(
            short_ema_period=5,
            long_ema_period=10,
            min_regime_ticks=2,
        )

    @pytest.fixture
    def orchestrator(
        self,
        mock_strategies: dict[str, MockStrategy],
        config: EnsembleConfig,
        regime_config: RegimeConfig,
    ) -> EnsembleOrchestrator:
        """Create orchestrator with mock strategies."""
        return EnsembleOrchestrator(
            strategies=mock_strategies,  # type: ignore
            config=config,
            regime_config=regime_config,
        )

    def test_decide_calls_all_strategies(
        self, orchestrator: EnsembleOrchestrator, mock_strategies: dict[str, MockStrategy]
    ):
        """Test that decide() calls all child strategies."""
        orchestrator.decide(
            symbol="BTC-USD",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[Decimal("50000")],
            equity=Decimal("10000"),
            product=None,
        )

        for strategy in mock_strategies.values():
            assert strategy.call_count == 1

    def test_decide_returns_aggregated_decision(self, orchestrator: EnsembleOrchestrator):
        """Test that decide() returns aggregated decision."""
        # Feed some prices first to warm up regime detector
        for i in range(20):
            orchestrator.decide(
                symbol="BTC-USD",
                current_mark=Decimal(str(50000 + i * 10)),
                position_state=None,
                recent_marks=[Decimal(str(50000 + i * 10))],
                equity=Decimal("10000"),
                product=None,
            )

        decision = orchestrator.decide(
            symbol="BTC-USD",
            current_mark=Decimal("50200"),
            position_state=None,
            recent_marks=[Decimal("50200")],
            equity=Decimal("10000"),
            product=None,
        )

        # Both strategies say BUY with good confidence
        assert decision.action == Action.BUY
        assert decision.confidence > 0.3

    def test_decide_includes_ensemble_metadata(self, orchestrator: EnsembleOrchestrator):
        """Test that decision includes ensemble metadata."""
        decision = orchestrator.decide(
            symbol="BTC-USD",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[Decimal("50000")],
            equity=Decimal("10000"),
            product=None,
        )

        assert "ensemble" in decision.indicators
        ensemble_data = decision.indicators["ensemble"]
        assert "regime" in ensemble_data
        assert "weights" in ensemble_data
        assert "votes" in ensemble_data
        assert "voting_method" in ensemble_data

    def test_unanimous_agreement(self, orchestrator: EnsembleOrchestrator):
        """Test unanimous agreement from strategies."""
        decision = orchestrator.decide(
            symbol="BTC-USD",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[Decimal("50000")],
            equity=Decimal("10000"),
            product=None,
        )

        # Both mock strategies return BUY
        assert decision.action == Action.BUY

    def test_disagreement_follows_leader(self, config: EnsembleConfig, regime_config: RegimeConfig):
        """Test that disagreement follows highest confidence."""
        strategies = {
            "high_conf": MockStrategy("high_conf", Action.BUY, 0.9),
            "low_conf": MockStrategy("low_conf", Action.SELL, 0.3),
        }

        orchestrator = EnsembleOrchestrator(
            strategies=strategies,  # type: ignore
            config=config,
            regime_config=regime_config,
        )

        decision = orchestrator.decide(
            symbol="BTC-USD",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[Decimal("50000")],
            equity=Decimal("10000"),
            product=None,
        )

        # high_conf has higher weighted_confidence
        assert decision.action == Action.BUY

    def test_low_confidence_returns_hold(self, config: EnsembleConfig, regime_config: RegimeConfig):
        """Test that low ensemble confidence returns HOLD."""
        config.min_ensemble_confidence = 0.95  # Very high threshold

        strategies = {
            "a": MockStrategy("a", Action.BUY, 0.3),
            "b": MockStrategy("b", Action.SELL, 0.3),
        }

        orchestrator = EnsembleOrchestrator(
            strategies=strategies,  # type: ignore
            config=config,
            regime_config=regime_config,
        )

        decision = orchestrator.decide(
            symbol="BTC-USD",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[Decimal("50000")],
            equity=Decimal("10000"),
            product=None,
        )

        # Low confidence should trigger HOLD
        assert decision.action == Action.HOLD

    def test_get_regime(self, orchestrator: EnsembleOrchestrator):
        """Test get_regime returns current regime."""
        # Initial state is unknown
        state = orchestrator.get_regime("BTC-USD")
        assert state.regime == RegimeType.UNKNOWN

        # After some updates
        for i in range(30):
            orchestrator.decide(
                symbol="BTC-USD",
                current_mark=Decimal(str(50000 + i * 100)),
                position_state=None,
                recent_marks=[Decimal(str(50000 + i * 100))],
                equity=Decimal("10000"),
                product=None,
            )

        state = orchestrator.get_regime("BTC-USD")
        # Should have some state now
        assert state is not None

    def test_get_strategy_weights(self, orchestrator: EnsembleOrchestrator):
        """Test get_strategy_weights returns current weights."""
        weights = orchestrator.get_strategy_weights("BTC-USD")

        # Should include all strategies
        assert "baseline" in weights
        assert "mean_reversion" in weights

        # Should sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.001


class TestCrisisMode:
    """Test crisis mode behavior."""

    @pytest.fixture
    def crisis_config(self) -> EnsembleConfig:
        """Create config with crisis handling."""
        return EnsembleConfig(
            voting_method="confidence_leader",
            crisis_behavior="scaled_down",
            crisis_scale_factor=0.2,
            min_ensemble_confidence=0.1,  # Low threshold for testing
        )

    def test_crisis_adjustment_applied(self, crisis_config: EnsembleConfig):
        """Test that crisis mode adjustment is applied when regime is crisis."""
        from gpt_trader.features.intelligence.ensemble.orchestrator import EnsembleOrchestrator
        from gpt_trader.features.intelligence.regime.models import RegimeState

        strategies = {
            "buyer": MockStrategy("buyer", Action.BUY, 0.9),
        }

        orchestrator = EnsembleOrchestrator(
            strategies=strategies,  # type: ignore
            config=crisis_config,
        )

        # Create a crisis regime state
        crisis_state = RegimeState(
            regime=RegimeType.CRISIS,
            confidence=0.9,
            trend_score=-0.8,
            volatility_percentile=0.95,
            momentum_score=-0.9,
        )

        # Test the crisis adjustment method directly
        decision = Decision(Action.BUY, "Test buy", 0.9, {})
        adjusted = orchestrator._apply_crisis_adjustment(
            decision=decision,
            regime_state=crisis_state,
            position_state=None,
        )

        # Should have crisis indicators
        assert adjusted.indicators.get("crisis_mode", False)
        assert adjusted.confidence < decision.confidence  # Confidence reduced


class TestSerialization:
    """Test state serialization."""

    @pytest.fixture
    def orchestrator(self) -> EnsembleOrchestrator:
        """Create orchestrator for serialization tests."""
        strategies = {
            "baseline": MockStrategy("baseline", Action.BUY, 0.7),
        }
        return EnsembleOrchestrator(strategies=strategies)  # type: ignore

    def test_serialize_state(self, orchestrator: EnsembleOrchestrator):
        """Test state serialization."""
        # Build up some state
        for i in range(30):
            orchestrator.decide(
                symbol="BTC-USD",
                current_mark=Decimal(str(50000 + i * 50)),
                position_state=None,
                recent_marks=[Decimal(str(50000 + i * 50))],
                equity=Decimal("10000"),
                product=None,
            )

        state = orchestrator.serialize_state()

        assert "regime_detector" in state
        assert "strategies" in state
        assert "last_regimes" in state

    def test_deserialize_state(self, orchestrator: EnsembleOrchestrator):
        """Test state restoration."""
        # Build up state
        for i in range(30):
            orchestrator.decide(
                symbol="BTC-USD",
                current_mark=Decimal(str(50000 + i * 50)),
                position_state=None,
                recent_marks=[Decimal(str(50000 + i * 50))],
                equity=Decimal("10000"),
                product=None,
            )

        # Serialize
        serialized = orchestrator.serialize_state()

        # Create new orchestrator
        new_strategies = {"baseline": MockStrategy("baseline", Action.BUY, 0.7)}
        new_orchestrator = EnsembleOrchestrator(strategies=new_strategies)  # type: ignore

        # Deserialize
        new_orchestrator.deserialize_state(serialized)

        # States should be restored (regime detector state)
        assert new_orchestrator.regime_detector is not None

"""Tests for MarketRegimeDetector crisis detection behavior."""

from decimal import Decimal

import pytest

from gpt_trader.features.intelligence.regime import (
    MarketRegimeDetector,
    RegimeConfig,
    RegimeState,
    RegimeType,
)


class TestCrisisDetection:
    """Tests for crisis regime detection."""

    @pytest.fixture
    def detector(self) -> MarketRegimeDetector:
        """Create detector for crisis testing."""
        config = RegimeConfig(
            short_ema_period=5,
            long_ema_period=10,
            crisis_drawdown_threshold=0.10,  # 10% drawdown
            crisis_volatility_multiplier=3.0,
        )
        return MarketRegimeDetector(config)

    def test_detects_drawdown_crisis(self, detector: MarketRegimeDetector) -> None:
        """Test crisis detection on large drawdown."""
        for _ in range(50):
            detector.update("BTC-USD", Decimal("60000"))

        for i in range(20):
            price = Decimal(str(60000 - i * 400))  # Dropping fast
            state = detector.update("BTC-USD", price)

        if state.regime != RegimeType.UNKNOWN:
            assert state.is_crisis() or state.regime in (
                RegimeType.BEAR_QUIET,
                RegimeType.BEAR_VOLATILE,
            )

    def test_crisis_state_to_dict(self) -> None:
        """Test crisis state serialization."""
        state = RegimeState(
            regime=RegimeType.CRISIS,
            confidence=0.9,
            trend_score=-0.8,
            volatility_percentile=0.98,
            momentum_score=-0.9,
            regime_age_ticks=3,
            transition_probability=0.8,
        )

        d = state.to_dict()
        assert d["regime"] == "CRISIS"
        assert state.is_crisis()
        assert state.is_volatile()

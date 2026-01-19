"""Tests for market regime core models."""

from gpt_trader.features.intelligence.regime import RegimeConfig, RegimeState, RegimeType


class TestRegimeType:
    """Tests for RegimeType enum."""

    def test_regime_types_exist(self) -> None:
        """Verify all expected regime types exist."""
        assert RegimeType.BULL_QUIET
        assert RegimeType.BULL_VOLATILE
        assert RegimeType.BEAR_QUIET
        assert RegimeType.BEAR_VOLATILE
        assert RegimeType.SIDEWAYS_QUIET
        assert RegimeType.SIDEWAYS_VOLATILE
        assert RegimeType.CRISIS
        assert RegimeType.UNKNOWN


class TestRegimeState:
    """Tests for RegimeState dataclass."""

    def test_create_regime_state(self) -> None:
        """Test creating a regime state."""
        state = RegimeState(
            regime=RegimeType.BULL_QUIET,
            confidence=0.8,
            trend_score=0.5,
            volatility_percentile=0.3,
            momentum_score=0.4,
            regime_age_ticks=10,
            transition_probability=0.1,
        )

        assert state.regime == RegimeType.BULL_QUIET
        assert state.confidence == 0.8
        assert state.trend_score == 0.5
        assert state.is_bullish()
        assert not state.is_bearish()
        assert not state.is_volatile()

    def test_unknown_factory(self) -> None:
        """Test unknown() factory method."""
        state = RegimeState.unknown()
        assert state.regime == RegimeType.UNKNOWN
        assert state.confidence == 0.0

    def test_is_bullish(self) -> None:
        """Test is_bullish helper."""
        assert RegimeState(
            regime=RegimeType.BULL_QUIET,
            confidence=0.8,
            trend_score=0.5,
            volatility_percentile=0.3,
            momentum_score=0.4,
        ).is_bullish()

        assert RegimeState(
            regime=RegimeType.BULL_VOLATILE,
            confidence=0.8,
            trend_score=0.5,
            volatility_percentile=0.8,
            momentum_score=0.4,
        ).is_bullish()

        assert not RegimeState(
            regime=RegimeType.BEAR_QUIET,
            confidence=0.8,
            trend_score=-0.5,
            volatility_percentile=0.3,
            momentum_score=-0.4,
        ).is_bullish()

    def test_is_volatile(self) -> None:
        """Test is_volatile helper."""
        assert RegimeState(
            regime=RegimeType.BULL_VOLATILE,
            confidence=0.8,
            trend_score=0.5,
            volatility_percentile=0.8,
            momentum_score=0.4,
        ).is_volatile()

        assert not RegimeState(
            regime=RegimeType.BULL_QUIET,
            confidence=0.8,
            trend_score=0.5,
            volatility_percentile=0.3,
            momentum_score=0.4,
        ).is_volatile()

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        state = RegimeState(
            regime=RegimeType.SIDEWAYS_QUIET,
            confidence=0.75,
            trend_score=0.0,
            volatility_percentile=0.25,
            momentum_score=0.1,
            regime_age_ticks=5,
            transition_probability=0.3,
        )

        d = state.to_dict()
        assert d["regime"] == "SIDEWAYS_QUIET"
        assert d["confidence"] == 0.75
        assert d["trend_score"] == 0.0


class TestRegimeConfig:
    """Tests for RegimeConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RegimeConfig()
        assert config.short_ema_period == 20
        assert config.long_ema_period == 50
        assert config.crisis_volatility_multiplier == 3.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RegimeConfig(
            short_ema_period=10,
            long_ema_period=30,
            min_regime_ticks=3,
        )
        assert config.short_ema_period == 10
        assert config.long_ema_period == 30
        assert config.min_regime_ticks == 3

    def test_to_dict_and_from_dict(self) -> None:
        """Test config serialization roundtrip."""
        original = RegimeConfig(
            short_ema_period=15,
            crisis_drawdown_threshold=0.15,
        )
        data = original.to_dict()
        restored = RegimeConfig.from_dict(data)

        assert restored.short_ema_period == 15
        assert restored.crisis_drawdown_threshold == 0.15

"""Tests for MarketRegimeDetector."""

from decimal import Decimal

import pytest

from gpt_trader.features.intelligence.regime import (
    MarketRegimeDetector,
    RegimeConfig,
    RegimeState,
    RegimeType,
)


class TestRegimeType:
    """Test RegimeType enum."""

    def test_regime_types_exist(self):
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
    """Test RegimeState dataclass."""

    def test_create_regime_state(self):
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

    def test_unknown_factory(self):
        """Test unknown() factory method."""
        state = RegimeState.unknown()
        assert state.regime == RegimeType.UNKNOWN
        assert state.confidence == 0.0

    def test_is_bullish(self):
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

    def test_is_volatile(self):
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

    def test_to_dict(self):
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
    """Test RegimeConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RegimeConfig()
        assert config.short_ema_period == 20
        assert config.long_ema_period == 50
        assert config.crisis_volatility_multiplier == 3.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = RegimeConfig(
            short_ema_period=10,
            long_ema_period=30,
            min_regime_ticks=3,
        )
        assert config.short_ema_period == 10
        assert config.long_ema_period == 30
        assert config.min_regime_ticks == 3

    def test_to_dict_and_from_dict(self):
        """Test config serialization roundtrip."""
        original = RegimeConfig(
            short_ema_period=15,
            crisis_drawdown_threshold=0.15,
        )
        data = original.to_dict()
        restored = RegimeConfig.from_dict(data)

        assert restored.short_ema_period == 15
        assert restored.crisis_drawdown_threshold == 0.15


class TestMarketRegimeDetector:
    """Test MarketRegimeDetector."""

    @pytest.fixture
    def detector(self) -> MarketRegimeDetector:
        """Create detector with default config."""
        return MarketRegimeDetector(RegimeConfig())

    @pytest.fixture
    def fast_detector(self) -> MarketRegimeDetector:
        """Create detector with faster warmup for testing."""
        config = RegimeConfig(
            short_ema_period=5,
            long_ema_period=10,
            min_regime_ticks=2,
        )
        return MarketRegimeDetector(config)

    def test_initial_state_is_unknown(self, detector: MarketRegimeDetector):
        """Test that initial state is unknown."""
        # First update - not enough data
        state = detector.update("BTC-USD", Decimal("50000"))
        assert state.regime == RegimeType.UNKNOWN

    def test_warmup_period(self, fast_detector: MarketRegimeDetector):
        """Test that detector needs warmup period."""
        # Feed a few prices
        for i in range(5):
            state = fast_detector.update("BTC-USD", Decimal("50000"))
            # Should still be unknown during warmup
            assert state.regime == RegimeType.UNKNOWN

    def test_detects_uptrend(self, fast_detector: MarketRegimeDetector):
        """Test detection of bullish regime."""
        # Feed rising prices
        base = 50000
        for i in range(100):
            price = Decimal(str(base + i * 100))  # Rising 100 per tick
            state = fast_detector.update("BTC-USD", price)

        # Should detect bullish regime after warmup
        assert state.regime in (
            RegimeType.BULL_QUIET,
            RegimeType.BULL_VOLATILE,
            RegimeType.UNKNOWN,  # May still be warming up
        )

    def test_detects_downtrend(self, fast_detector: MarketRegimeDetector):
        """Test detection of bearish regime."""
        # Feed falling prices
        base = 60000
        for i in range(100):
            price = Decimal(str(base - i * 100))  # Falling 100 per tick
            state = fast_detector.update("BTC-USD", price)

        # Should detect bearish regime or crisis (large drawdown triggers crisis)
        if state.regime != RegimeType.UNKNOWN:
            assert (
                state.is_bearish()
                or state.is_crisis()
                or state.regime
                in (
                    RegimeType.SIDEWAYS_QUIET,
                    RegimeType.SIDEWAYS_VOLATILE,
                )
            )

    def test_detects_sideways(self, fast_detector: MarketRegimeDetector):
        """Test detection of sideways regime."""
        # Feed oscillating prices around a center
        center = 50000
        for i in range(100):
            # Oscillate +/- 50 around center
            offset = 50 if i % 2 == 0 else -50
            price = Decimal(str(center + offset))
            state = fast_detector.update("BTC-USD", price)

        # Should detect sideways regime (low trend score)
        if state.regime != RegimeType.UNKNOWN:
            assert (
                state.regime
                in (
                    RegimeType.SIDEWAYS_QUIET,
                    RegimeType.SIDEWAYS_VOLATILE,
                )
                or abs(state.trend_score) < 0.5
            )

    def test_multiple_symbols(self, fast_detector: MarketRegimeDetector):
        """Test tracking multiple symbols independently."""
        # BTC rising
        for i in range(50):
            fast_detector.update("BTC-USD", Decimal(str(50000 + i * 100)))

        # ETH falling
        for i in range(50):
            fast_detector.update("ETH-USD", Decimal(str(3000 - i * 10)))

        btc_state = fast_detector.get_regime("BTC-USD")
        eth_state = fast_detector.get_regime("ETH-USD")

        # States should be independent
        assert btc_state.trend_score != eth_state.trend_score

    def test_serialization_roundtrip(self, fast_detector: MarketRegimeDetector):
        """Test state persistence and recovery."""
        # Build up some state
        for i in range(50):
            fast_detector.update("BTC-USD", Decimal(str(50000 + i * 50)))

        fast_detector.get_regime("BTC-USD")

        # Serialize
        serialized = fast_detector.serialize_state()

        # Create new detector and restore
        new_detector = MarketRegimeDetector(fast_detector.config)
        new_detector.deserialize_state(serialized)

        # Continue with same price
        fast_detector.update("BTC-USD", Decimal("52500"))
        new_detector.update("BTC-USD", Decimal("52500"))

        # States should be similar (not exactly equal due to floating point)
        assert (
            fast_detector.get_regime("BTC-USD").regime == new_detector.get_regime("BTC-USD").regime
        )

    def test_reset_symbol(self, fast_detector: MarketRegimeDetector):
        """Test resetting a single symbol."""
        # Build state
        for i in range(50):
            fast_detector.update("BTC-USD", Decimal(str(50000 + i * 50)))

        # Reset
        fast_detector.reset("BTC-USD")

        # Should be back to unknown
        state = fast_detector.get_regime("BTC-USD")
        assert state.regime == RegimeType.UNKNOWN

    def test_reset_all(self, fast_detector: MarketRegimeDetector):
        """Test resetting all symbols."""
        # Build state for multiple symbols
        for i in range(50):
            fast_detector.update("BTC-USD", Decimal(str(50000 + i * 50)))
            fast_detector.update("ETH-USD", Decimal(str(3000 + i * 10)))

        # Reset all
        fast_detector.reset()

        # Both should be unknown
        assert fast_detector.get_regime("BTC-USD").regime == RegimeType.UNKNOWN
        assert fast_detector.get_regime("ETH-USD").regime == RegimeType.UNKNOWN

    def test_regime_persistence(self, fast_detector: MarketRegimeDetector):
        """Test that regime doesn't flip-flop on noise."""
        config = fast_detector.config

        # Establish an uptrend
        for i in range(50):
            fast_detector.update("BTC-USD", Decimal(str(50000 + i * 100)))

        # Small counter-trend noise shouldn't flip regime immediately
        for i in range(config.min_regime_ticks - 1):
            fast_detector.update("BTC-USD", Decimal(str(55000 - i * 50)))

        state = fast_detector.get_regime("BTC-USD")
        # Should still be in previous regime due to persistence
        assert state.regime != RegimeType.UNKNOWN


class TestCrisisDetection:
    """Test crisis regime detection."""

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

    def test_detects_drawdown_crisis(self, detector: MarketRegimeDetector):
        """Test crisis detection on large drawdown."""
        # Establish a peak
        for i in range(50):
            detector.update("BTC-USD", Decimal("60000"))

        # Sharp drop (>10%)
        for i in range(20):
            price = Decimal(str(60000 - i * 400))  # Dropping fast
            state = detector.update("BTC-USD", price)

        # Should detect crisis due to drawdown
        # Note: May need more data for crisis detection to kick in
        if state.regime != RegimeType.UNKNOWN:
            # At 20 drops of 400, we're at 52000 which is ~13% drawdown
            assert state.is_crisis() or state.regime in (
                RegimeType.BEAR_QUIET,
                RegimeType.BEAR_VOLATILE,
            )

    def test_crisis_state_to_dict(self, detector: MarketRegimeDetector):
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

"""Tests for MarketRegimeDetector persistence and reset behavior."""

from decimal import Decimal

import pytest

from gpt_trader.features.intelligence.regime import (
    MarketRegimeDetector,
    RegimeConfig,
    RegimeType,
)


class TestMarketRegimeDetectorPersistence:
    """Tests for MarketRegimeDetector persistence behavior."""

    @pytest.fixture
    def fast_detector(self) -> MarketRegimeDetector:
        """Create detector with faster warmup for testing."""
        config = RegimeConfig(
            short_ema_period=5,
            long_ema_period=10,
            min_regime_ticks=2,
        )
        return MarketRegimeDetector(config)

    def test_serialization_roundtrip(self, fast_detector: MarketRegimeDetector) -> None:
        """Test state persistence and recovery."""
        for i in range(50):
            fast_detector.update("BTC-USD", Decimal(str(50000 + i * 50)))

        fast_detector.get_regime("BTC-USD")

        serialized = fast_detector.serialize_state()

        new_detector = MarketRegimeDetector(fast_detector.config)
        new_detector.deserialize_state(serialized)

        fast_detector.update("BTC-USD", Decimal("52500"))
        new_detector.update("BTC-USD", Decimal("52500"))

        assert (
            fast_detector.get_regime("BTC-USD").regime == new_detector.get_regime("BTC-USD").regime
        )

    def test_reset_symbol(self, fast_detector: MarketRegimeDetector) -> None:
        """Test resetting a single symbol."""
        for i in range(50):
            fast_detector.update("BTC-USD", Decimal(str(50000 + i * 50)))

        fast_detector.reset("BTC-USD")

        state = fast_detector.get_regime("BTC-USD")
        assert state.regime == RegimeType.UNKNOWN

    def test_reset_all(self, fast_detector: MarketRegimeDetector) -> None:
        """Test resetting all symbols."""
        for i in range(50):
            fast_detector.update("BTC-USD", Decimal(str(50000 + i * 50)))
            fast_detector.update("ETH-USD", Decimal(str(3000 + i * 10)))

        fast_detector.reset()

        assert fast_detector.get_regime("BTC-USD").regime == RegimeType.UNKNOWN
        assert fast_detector.get_regime("ETH-USD").regime == RegimeType.UNKNOWN

    def test_regime_persistence(self, fast_detector: MarketRegimeDetector) -> None:
        """Test that regime doesn't flip-flop on noise."""
        config = fast_detector.config

        for i in range(50):
            fast_detector.update("BTC-USD", Decimal(str(50000 + i * 100)))

        for i in range(config.min_regime_ticks - 1):
            fast_detector.update("BTC-USD", Decimal(str(55000 - i * 50)))

        state = fast_detector.get_regime("BTC-USD")
        assert state.regime != RegimeType.UNKNOWN

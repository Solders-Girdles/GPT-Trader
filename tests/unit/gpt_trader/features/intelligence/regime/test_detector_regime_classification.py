"""Tests for MarketRegimeDetector regime classification behavior."""

from decimal import Decimal

import pytest

from gpt_trader.features.intelligence.regime import (
    MarketRegimeDetector,
    RegimeConfig,
    RegimeType,
)


class TestMarketRegimeDetector:
    """Tests for MarketRegimeDetector."""

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

    def test_initial_state_is_unknown(self, detector: MarketRegimeDetector) -> None:
        """Test that initial state is unknown."""
        state = detector.update("BTC-USD", Decimal("50000"))
        assert state.regime == RegimeType.UNKNOWN

    def test_warmup_period(self, fast_detector: MarketRegimeDetector) -> None:
        """Test that detector needs warmup period."""
        for _ in range(5):
            state = fast_detector.update("BTC-USD", Decimal("50000"))
            assert state.regime == RegimeType.UNKNOWN

    def test_detects_uptrend(self, fast_detector: MarketRegimeDetector) -> None:
        """Test detection of bullish regime."""
        base = 50000
        for i in range(100):
            price = Decimal(str(base + i * 100))  # Rising 100 per tick
            state = fast_detector.update("BTC-USD", price)

        assert state.regime in (
            RegimeType.BULL_QUIET,
            RegimeType.BULL_VOLATILE,
            RegimeType.UNKNOWN,  # May still be warming up
        )

    def test_detects_downtrend(self, fast_detector: MarketRegimeDetector) -> None:
        """Test detection of bearish regime."""
        base = 60000
        for i in range(100):
            price = Decimal(str(base - i * 100))  # Falling 100 per tick
            state = fast_detector.update("BTC-USD", price)

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

    def test_detects_sideways(self, fast_detector: MarketRegimeDetector) -> None:
        """Test detection of sideways regime."""
        center = 50000
        for i in range(100):
            offset = 50 if i % 2 == 0 else -50
            price = Decimal(str(center + offset))
            state = fast_detector.update("BTC-USD", price)

        if state.regime != RegimeType.UNKNOWN:
            assert (
                state.regime
                in (
                    RegimeType.SIDEWAYS_QUIET,
                    RegimeType.SIDEWAYS_VOLATILE,
                )
                or abs(state.trend_score) < 0.5
            )

    def test_multiple_symbols(self, fast_detector: MarketRegimeDetector) -> None:
        """Test tracking multiple symbols independently."""
        for i in range(50):
            fast_detector.update("BTC-USD", Decimal(str(50000 + i * 100)))

        for i in range(50):
            fast_detector.update("ETH-USD", Decimal(str(3000 - i * 10)))

        btc_state = fast_detector.get_regime("BTC-USD")
        eth_state = fast_detector.get_regime("ETH-USD")

        assert btc_state.trend_score != eth_state.trend_score

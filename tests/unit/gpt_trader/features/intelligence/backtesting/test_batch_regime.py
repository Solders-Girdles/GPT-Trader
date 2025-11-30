"""Tests for batch regime detection module."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.features.intelligence.backtesting.batch_regime import (
    BatchRegimeDetector,
    RegimeHistory,
    RegimeSnapshot,
)
from gpt_trader.features.intelligence.regime.models import RegimeType


class TestRegimeSnapshot:
    """Tests for RegimeSnapshot."""

    def test_create_snapshot(self):
        """Test creating a regime snapshot."""
        snapshot = RegimeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0),
            price=Decimal("50000"),
            regime=RegimeType.BULL_QUIET,
            confidence=0.85,
            volatility_percentile=0.3,
            trend_percentile=0.7,
        )

        assert snapshot.regime == RegimeType.BULL_QUIET
        assert snapshot.confidence == 0.85

    def test_snapshot_with_advanced_indicators(self):
        """Test snapshot with advanced indicator values."""
        snapshot = RegimeSnapshot(
            timestamp=datetime(2024, 1, 1),
            price=Decimal("50000"),
            regime=RegimeType.BULL_VOLATILE,
            confidence=0.75,
            volatility_percentile=0.8,
            trend_percentile=0.6,
            atr_value=1500.0,
            atr_percentile=0.7,
            adx_value=45.0,
            squeeze_score=0.2,
        )

        assert snapshot.atr_value == 1500.0
        assert snapshot.adx_value == 45.0

    def test_to_dict(self):
        """Test snapshot serialization."""
        snapshot = RegimeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0),
            price=Decimal("50000"),
            regime=RegimeType.CRISIS,
            confidence=0.9,
            volatility_percentile=0.95,
            trend_percentile=0.1,
        )

        data = snapshot.to_dict()

        assert data["regime"] == "CRISIS"
        assert data["price"] == "50000"
        assert "timestamp" in data


class TestRegimeHistory:
    """Tests for RegimeHistory."""

    @pytest.fixture
    def sample_history(self):
        """Create sample regime history."""
        history = RegimeHistory(symbol="BTC-USD")

        base_time = datetime(2024, 1, 1)
        regimes = [
            RegimeType.BULL_QUIET,
            RegimeType.BULL_QUIET,
            RegimeType.BULL_VOLATILE,
            RegimeType.CRISIS,
            RegimeType.CRISIS,
            RegimeType.BEAR_QUIET,
        ]

        for i, regime in enumerate(regimes):
            snapshot = RegimeSnapshot(
                timestamp=base_time + timedelta(hours=i),
                price=Decimal(str(50000 - i * 100)),
                regime=regime,
                confidence=0.8,
                volatility_percentile=0.5,
                trend_percentile=0.5,
            )
            history.snapshots.append(snapshot)

        return history

    def test_len(self, sample_history):
        """Test history length."""
        assert len(sample_history) == 6

    def test_iter(self, sample_history):
        """Test history iteration."""
        count = sum(1 for _ in sample_history)
        assert count == 6

    def test_get_regime_at(self, sample_history):
        """Test getting regime at specific time."""
        base_time = datetime(2024, 1, 1)

        # Exact match
        snapshot = sample_history.get_regime_at(base_time + timedelta(hours=2))
        assert snapshot is not None
        assert snapshot.regime == RegimeType.BULL_VOLATILE

        # Between snapshots - should return previous
        snapshot = sample_history.get_regime_at(base_time + timedelta(hours=2, minutes=30))
        assert snapshot is not None
        assert snapshot.regime == RegimeType.BULL_VOLATILE

    def test_get_regime_at_before_start(self, sample_history):
        """Test getting regime before history start."""
        result = sample_history.get_regime_at(datetime(2023, 1, 1))
        assert result is None

    def test_get_regime_distribution(self, sample_history):
        """Test regime distribution calculation."""
        distribution = sample_history.get_regime_distribution()

        # 2 BULL_QUIET, 1 BULL_VOLATILE, 2 CRISIS, 1 BEAR_QUIET
        assert distribution["BULL_QUIET"] == pytest.approx(33.33, abs=0.1)
        assert distribution["CRISIS"] == pytest.approx(33.33, abs=0.1)

    def test_get_regime_transitions(self, sample_history):
        """Test regime transition detection."""
        transitions = sample_history.get_regime_transitions()

        # BULL_QUIET -> BULL_VOLATILE -> CRISIS -> BEAR_QUIET
        assert len(transitions) == 3

        # First transition should be BULL_QUIET -> BULL_VOLATILE
        ts, from_regime, to_regime = transitions[0]
        assert from_regime == RegimeType.BULL_QUIET
        assert to_regime == RegimeType.BULL_VOLATILE

    def test_get_crisis_periods(self, sample_history):
        """Test crisis period detection."""
        periods = sample_history.get_crisis_periods()

        assert len(periods) == 1
        start, end = periods[0]
        assert start == datetime(2024, 1, 1, 3, 0)  # Hour 3

    def test_get_average_confidence(self, sample_history):
        """Test average confidence calculation."""
        avg_conf = sample_history.get_average_confidence()
        assert avg_conf == pytest.approx(0.8)

    def test_get_volatility_summary(self, sample_history):
        """Test volatility summary."""
        summary = sample_history.get_volatility_summary()

        assert "mean" in summary
        assert "min" in summary
        assert "max" in summary

    def test_summary(self, sample_history):
        """Test comprehensive summary."""
        summary = sample_history.summary()

        assert summary["symbol"] == "BTC-USD"
        assert summary["total_bars"] == 6
        assert summary["total_transitions"] == 3
        assert summary["crisis_periods"] == 1

    def test_to_dataframe_rows(self, sample_history):
        """Test DataFrame row generation."""
        rows = sample_history.to_dataframe_rows()

        assert len(rows) == 6
        assert all("timestamp" in row for row in rows)
        assert all("regime" in row for row in rows)


class TestBatchRegimeDetector:
    """Tests for BatchRegimeDetector."""

    def test_process_prices(self):
        """Test processing price sequence."""
        detector = BatchRegimeDetector(warmup_bars=10)

        # Generate 100 prices with some trend
        prices = [Decimal(str(50000 + i * 10)) for i in range(100)]

        history = detector.process(
            symbol="BTC-USD",
            prices=prices,
        )

        # Should have 90 snapshots (100 - 10 warmup)
        assert len(history) == 90
        assert history.symbol == "BTC-USD"

    def test_process_with_timestamps(self):
        """Test processing with explicit timestamps."""
        detector = BatchRegimeDetector(warmup_bars=5)

        base_time = datetime(2024, 1, 1)
        prices = [Decimal(str(50000 + i * 5)) for i in range(50)]
        timestamps = [base_time + timedelta(minutes=i) for i in range(50)]

        history = detector.process(
            symbol="TEST",
            prices=prices,
            timestamps=timestamps,
        )

        # Snapshots should have correct timestamps
        assert history.snapshots[0].timestamp == base_time + timedelta(minutes=5)

    def test_process_volatile_prices(self):
        """Test processing volatile price data."""
        detector = BatchRegimeDetector(warmup_bars=30)

        # Generate prices with high volatility
        import math

        prices = []
        for i in range(100):
            # Large oscillations
            price = 50000 + 2000 * math.sin(i / 5)
            prices.append(Decimal(str(int(price))))

        history = detector.process(symbol="VOLATILE", prices=prices)

        # Should generate a valid history
        assert len(history) == 70  # 100 - 30 warmup

        # Should have regime distribution (regardless of specific regime types)
        distribution = history.get_regime_distribution()
        assert len(distribution) > 0
        # Sum of distribution should be 100%
        assert sum(distribution.values()) == pytest.approx(100.0, abs=0.1)

    def test_process_trending_prices(self):
        """Test processing trending price data."""
        detector = BatchRegimeDetector(warmup_bars=30)

        # Strong uptrend
        prices = [Decimal(str(50000 + i * 100)) for i in range(100)]

        history = detector.process(symbol="TRENDING", prices=prices)

        # Should generate valid history
        assert len(history) == 70  # 100 - 30 warmup

        # Should have regime distribution
        distribution = history.get_regime_distribution()
        assert len(distribution) > 0
        # All snapshots should have positive trend percentile (uptrend)
        avg_trend = sum(s.trend_percentile for s in history) / len(history)
        assert avg_trend >= 0.5  # Should be bullish on average

    def test_reset(self):
        """Test detector reset."""
        detector = BatchRegimeDetector(warmup_bars=5)

        prices = [Decimal(str(50000 + i)) for i in range(20)]
        detector.process("TEST", prices)

        # Reset
        detector.reset()

        # Process again should work
        history = detector.process("TEST2", prices)
        assert history.symbol == "TEST2"

    def test_process_candles_interface(self):
        """Test process_candles convenience method."""
        detector = BatchRegimeDetector(warmup_bars=5)

        # Create mock candle objects
        class MockCandle:
            def __init__(self, close, ts):
                self.close = close
                self.ts = ts

        candles = [
            MockCandle(
                close=Decimal(str(50000 + i * 10)),
                ts=datetime(2024, 1, 1) + timedelta(minutes=i),
            )
            for i in range(50)
        ]

        history = detector.process_candles(
            symbol="TEST",
            candles=candles,
            price_field="close",
        )

        assert len(history) == 45  # 50 - 5 warmup

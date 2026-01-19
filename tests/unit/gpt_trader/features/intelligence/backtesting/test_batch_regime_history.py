"""Tests for batch regime history model."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.features.intelligence.backtesting.batch_regime import (
    RegimeHistory,
    RegimeSnapshot,
)
from gpt_trader.features.intelligence.regime.models import RegimeType


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

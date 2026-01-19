"""Tests for batch regime snapshot model."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from gpt_trader.features.intelligence.backtesting.batch_regime import RegimeSnapshot
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

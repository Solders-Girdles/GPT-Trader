"""Tests for MetricsTracker - liquidity metrics tracking and aggregation.

This module tests the MetricsTracker's ability to:
- Track trade and spread data within rolling time windows
- Calculate volume metrics across multiple timeframes
- Aggregate spread statistics
- Manage per-symbol metrics independently
- Clean expired data
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.features.live_trade.liquidity_metrics_tracker import LiquidityMetrics, MetricsTracker


class TestLiquidityMetrics:
    """Test LiquidityMetrics time-series tracking."""

    def test_initialization_creates_empty_metrics(self):
        """Creates metrics with empty data structures."""
        metrics = LiquidityMetrics(window_minutes=15)

        assert metrics.window_duration == timedelta(minutes=15)
        volume_metrics = metrics.get_volume_metrics()
        assert volume_metrics["volume_15m"] == Decimal("0")
        assert volume_metrics["trade_count"] == 0

    def test_add_trade_updates_volume_data(self):
        """Adds trade and calculates volume correctly."""
        metrics = LiquidityMetrics(window_minutes=15)

        # Add trade: $50,000 price × 0.5 size = $25,000 notional
        metrics.add_trade(Decimal("50000"), Decimal("0.5"))

        volume_metrics = metrics.get_volume_metrics()
        assert volume_metrics["volume_15m"] == Decimal("25000")
        assert volume_metrics["trade_count"] == 1
        assert volume_metrics["avg_trade_size"] == Decimal("0.5")

    def test_add_multiple_trades_aggregates_volume(self):
        """Aggregates multiple trades within window."""
        metrics = LiquidityMetrics(window_minutes=15)
        now = datetime.now()

        # Add three trades
        metrics.add_trade(Decimal("50000"), Decimal("0.5"), now - timedelta(minutes=1))
        metrics.add_trade(Decimal("50000"), Decimal("0.3"), now - timedelta(minutes=2))
        metrics.add_trade(Decimal("50000"), Decimal("0.2"), now - timedelta(minutes=3))

        volume_metrics = metrics.get_volume_metrics()
        # Total: (0.5 + 0.3 + 0.2) * 50000 = 50,000
        assert volume_metrics["volume_15m"] == Decimal("50000")
        assert volume_metrics["trade_count"] == 3
        assert volume_metrics["avg_trade_size"] == Decimal("1.0") / Decimal("3")

    def test_volume_metrics_respects_time_windows(self):
        """Calculates volume for 1m, 5m, 15m windows correctly."""
        metrics = LiquidityMetrics(window_minutes=15)
        now = datetime.now()

        # Trade within 1 minute
        metrics.add_trade(Decimal("50000"), Decimal("0.1"), now - timedelta(seconds=30))
        # Trade within 5 minutes
        metrics.add_trade(Decimal("50000"), Decimal("0.2"), now - timedelta(minutes=3))
        # Trade within 15 minutes
        metrics.add_trade(Decimal("50000"), Decimal("0.3"), now - timedelta(minutes=10))

        volume_metrics = metrics.get_volume_metrics()

        # 1m window: only first trade
        assert volume_metrics["volume_1m"] == Decimal("5000")  # 0.1 * 50000
        # 5m window: first two trades
        assert volume_metrics["volume_5m"] == Decimal("15000")  # (0.1 + 0.2) * 50000
        # 15m window: all three trades
        assert volume_metrics["volume_15m"] == Decimal("30000")  # (0.1 + 0.2 + 0.3) * 50000

    def test_add_spread_updates_spread_data(self):
        """Adds spread and calculates metrics correctly."""
        metrics = LiquidityMetrics(window_minutes=15)

        metrics.add_spread(Decimal("10"))  # 10 bps spread

        spread_metrics = metrics.get_spread_metrics()
        assert spread_metrics["avg_spread_bps"] == Decimal("10")
        assert spread_metrics["min_spread_bps"] == Decimal("10")
        assert spread_metrics["max_spread_bps"] == Decimal("10")

    def test_spread_metrics_calculates_avg_min_max(self):
        """Calculates average, min, and max spread over 5m window."""
        metrics = LiquidityMetrics(window_minutes=15)
        now = datetime.now()

        # Add spreads within 5m window
        metrics.add_spread(Decimal("10"), now - timedelta(minutes=1))
        metrics.add_spread(Decimal("20"), now - timedelta(minutes=2))
        metrics.add_spread(Decimal("15"), now - timedelta(minutes=3))

        spread_metrics = metrics.get_spread_metrics()
        assert spread_metrics["avg_spread_bps"] == Decimal("15")  # (10 + 20 + 15) / 3
        assert spread_metrics["min_spread_bps"] == Decimal("10")
        assert spread_metrics["max_spread_bps"] == Decimal("20")

    def test_spread_metrics_only_uses_last_5_minutes(self):
        """Spread metrics only include last 5 minutes of data."""
        metrics = LiquidityMetrics(window_minutes=15)
        now = datetime.now()

        # Add old spread (outside 5m window but inside 15m window)
        metrics.add_spread(Decimal("100"), now - timedelta(minutes=10))
        # Add recent spread (inside 5m window)
        metrics.add_spread(Decimal("20"), now - timedelta(minutes=2))

        spread_metrics = metrics.get_spread_metrics()
        # Should only include recent spread
        assert spread_metrics["avg_spread_bps"] == Decimal("20")
        assert spread_metrics["min_spread_bps"] == Decimal("20")
        assert spread_metrics["max_spread_bps"] == Decimal("20")

    def test_cleans_old_data_beyond_window(self):
        """Removes data older than window duration."""
        metrics = LiquidityMetrics(window_minutes=15)
        now = datetime.now()

        # Add old trade (outside 15m window)
        metrics.add_trade(Decimal("50000"), Decimal("1.0"), now - timedelta(minutes=20))
        # Add recent trade (inside 15m window)
        metrics.add_trade(Decimal("50000"), Decimal("0.5"), now - timedelta(minutes=5))

        volume_metrics = metrics.get_volume_metrics()
        # Should only count recent trade
        assert volume_metrics["volume_15m"] == Decimal("25000")  # 0.5 * 50000
        assert volume_metrics["trade_count"] == 1

    def test_empty_metrics_returns_zeros(self):
        """Returns zero values when no data available."""
        metrics = LiquidityMetrics(window_minutes=15)

        volume_metrics = metrics.get_volume_metrics()
        assert volume_metrics["volume_1m"] == Decimal("0")
        assert volume_metrics["volume_5m"] == Decimal("0")
        assert volume_metrics["volume_15m"] == Decimal("0")
        assert volume_metrics["trade_count"] == 0
        assert volume_metrics["avg_trade_size"] == Decimal("0")

        spread_metrics = metrics.get_spread_metrics()
        assert spread_metrics["avg_spread_bps"] == Decimal("0")
        assert spread_metrics["min_spread_bps"] == Decimal("0")
        assert spread_metrics["max_spread_bps"] == Decimal("0")

    def test_spread_metrics_returns_zeros_when_no_recent_data(self):
        """Returns zeros when spreads exist but outside 5m window."""
        metrics = LiquidityMetrics(window_minutes=15)
        now = datetime.now()

        # Add old spread (outside 5m window but inside 15m window)
        metrics.add_spread(Decimal("50"), now - timedelta(minutes=10))

        spread_metrics = metrics.get_spread_metrics()
        # No spreads in last 5 minutes
        assert spread_metrics["avg_spread_bps"] == Decimal("0")
        assert spread_metrics["min_spread_bps"] == Decimal("0")
        assert spread_metrics["max_spread_bps"] == Decimal("0")


class TestMetricsTracker:
    """Test MetricsTracker per-symbol management."""

    def test_initialization_creates_empty_tracker(self):
        """Creates tracker with no symbols."""
        tracker = MetricsTracker(window_minutes=15)

        assert not tracker.has_symbol("BTC-USD")

    def test_add_trade_creates_symbol_metrics_lazily(self):
        """Creates metrics for symbol on first trade."""
        tracker = MetricsTracker(window_minutes=15)

        assert not tracker.has_symbol("BTC-USD")

        tracker.add_trade("BTC-USD", Decimal("50000"), Decimal("0.5"))

        assert tracker.has_symbol("BTC-USD")
        volume_metrics = tracker.get_volume_metrics("BTC-USD")
        assert volume_metrics["volume_15m"] == Decimal("25000")

    def test_add_spread_creates_symbol_metrics_lazily(self):
        """Creates metrics for symbol on first spread update."""
        tracker = MetricsTracker(window_minutes=15)

        tracker.add_spread("ETH-USD", Decimal("15"))

        assert tracker.has_symbol("ETH-USD")
        spread_metrics = tracker.get_spread_metrics("ETH-USD")
        assert spread_metrics["avg_spread_bps"] == Decimal("15")

    def test_tracks_multiple_symbols_independently(self):
        """Tracks metrics for multiple symbols without interference."""
        tracker = MetricsTracker(window_minutes=15)

        # Add trades for different symbols
        tracker.add_trade("BTC-USD", Decimal("50000"), Decimal("1.0"))
        tracker.add_trade("ETH-USD", Decimal("3000"), Decimal("5.0"))

        btc_metrics = tracker.get_volume_metrics("BTC-USD")
        eth_metrics = tracker.get_volume_metrics("ETH-USD")

        # Each symbol has independent metrics
        assert btc_metrics["volume_15m"] == Decimal("50000")
        assert eth_metrics["volume_15m"] == Decimal("15000")
        assert btc_metrics["trade_count"] == 1
        assert eth_metrics["trade_count"] == 1

    def test_get_volume_metrics_creates_symbol_if_missing(self):
        """Returns zero metrics for unknown symbol."""
        tracker = MetricsTracker(window_minutes=15)

        volume_metrics = tracker.get_volume_metrics("UNKNOWN-SYMBOL")

        assert volume_metrics["volume_15m"] == Decimal("0")
        assert volume_metrics["trade_count"] == 0
        # Symbol is now tracked (lazy creation)
        assert tracker.has_symbol("UNKNOWN-SYMBOL")

    def test_get_spread_metrics_creates_symbol_if_missing(self):
        """Returns zero spread metrics for unknown symbol."""
        tracker = MetricsTracker(window_minutes=15)

        spread_metrics = tracker.get_spread_metrics("UNKNOWN-SYMBOL")

        assert spread_metrics["avg_spread_bps"] == Decimal("0")
        # Symbol is now tracked (lazy creation)
        assert tracker.has_symbol("UNKNOWN-SYMBOL")

    def test_add_trade_with_custom_timestamp(self):
        """Adds trade with custom timestamp correctly."""
        tracker = MetricsTracker(window_minutes=15)
        timestamp = datetime.now() - timedelta(minutes=3)  # Safely within 5m window

        tracker.add_trade("BTC-USD", Decimal("50000"), Decimal("1.0"), timestamp)

        volume_metrics = tracker.get_volume_metrics("BTC-USD")
        # Trade should be counted in 15m and 5m windows
        assert volume_metrics["volume_15m"] == Decimal("50000")
        assert volume_metrics["volume_5m"] == Decimal("50000")

    def test_add_spread_with_custom_timestamp(self):
        """Adds spread with custom timestamp correctly."""
        tracker = MetricsTracker(window_minutes=15)
        timestamp = datetime.now() - timedelta(minutes=2)

        tracker.add_spread("BTC-USD", Decimal("20"), timestamp)

        spread_metrics = tracker.get_spread_metrics("BTC-USD")
        assert spread_metrics["avg_spread_bps"] == Decimal("20")

    def test_window_duration_propagates_to_symbol_metrics(self):
        """Window duration from tracker is used for symbol metrics."""
        tracker = MetricsTracker(window_minutes=30)  # 30-minute window

        tracker.add_trade("BTC-USD", Decimal("50000"), Decimal("1.0"))

        # Verify the underlying metrics has correct window
        metrics = tracker._get_metrics("BTC-USD")
        assert metrics.window_duration == timedelta(minutes=30)

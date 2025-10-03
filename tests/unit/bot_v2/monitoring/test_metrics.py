"""Tests for monitoring.metrics module."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.monitoring.metrics import MetricsCalculator
from bot_v2.persistence.event_store import EventStore


@pytest.fixture
def event_store(tmp_path):
    """Create an event store for testing."""
    store = EventStore(str(tmp_path / "test_events.jsonl"))
    return store


@pytest.fixture
def metrics_calculator(event_store):
    """Create a MetricsCalculator instance."""
    return MetricsCalculator(event_store=event_store)


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    def test_coerce_decimal_with_none(self):
        """Test decimal coercion with None value."""
        result = MetricsCalculator._coerce_decimal(None)
        assert result is None

    def test_coerce_decimal_with_valid_values(self):
        """Test decimal coercion with valid numeric values."""
        assert MetricsCalculator._coerce_decimal(100) == Decimal("100")
        assert MetricsCalculator._coerce_decimal(100.5) == Decimal("100.5")
        assert MetricsCalculator._coerce_decimal("99.99") == Decimal("99.99")
        assert MetricsCalculator._coerce_decimal(Decimal("50.25")) == Decimal("50.25")

    def test_coerce_decimal_with_invalid_values(self):
        """Test decimal coercion with invalid values."""
        assert MetricsCalculator._coerce_decimal("invalid") is None
        assert MetricsCalculator._coerce_decimal({}) is None
        assert MetricsCalculator._coerce_decimal([]) is None

    def test_get_equity_curve_empty_store(self, metrics_calculator):
        """Test equity curve with empty event store."""
        curve = metrics_calculator.get_equity_curve(days=30)
        assert curve == []

    def test_get_equity_curve_with_events(self, metrics_calculator, event_store):
        """Test equity curve with valid events."""
        now = datetime.utcnow()

        # Add events with equity data
        for i in range(5):
            timestamp = now - timedelta(days=i)
            event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    "timestamp": timestamp.isoformat(),
                    "equity": str(1000 + i * 100),
                },
            )

        curve = metrics_calculator.get_equity_curve(days=30)
        assert len(curve) == 5
        assert all(isinstance(ts, datetime) for ts, _ in curve)
        assert all(isinstance(eq, Decimal) for _, eq in curve)

    def test_get_equity_series_filters_by_cutoff(self, metrics_calculator, event_store):
        """Test that equity series filters events by cutoff date."""
        now = datetime.utcnow()

        # Add old event (should be filtered out)
        old_timestamp = now - timedelta(days=40)
        event_store.append_metric(
            bot_id="risk_engine",
            metrics={
                "timestamp": old_timestamp.isoformat(),
                "equity": "500",
            },
        )

        # Add recent event (should be included)
        recent_timestamp = now - timedelta(days=5)
        event_store.append_metric(
            bot_id="risk_engine",
            metrics={
                "timestamp": recent_timestamp.isoformat(),
                "equity": "1000",
            },
        )

        curve = metrics_calculator.get_equity_curve(days=30)
        assert len(curve) == 1
        assert curve[0][1] == Decimal("1000")

    def test_get_equity_series_with_timestamp_formats(self, metrics_calculator, event_store):
        """Test handling of different timestamp formats."""
        now = datetime.utcnow()

        # ISO format string
        event_store.append_metric(
            bot_id="risk_engine",
            metrics={
                "timestamp": now.isoformat(),
                "equity": "1000",
            },
        )

        # Unix timestamp (float)
        event_store.append_metric(
            bot_id="risk_engine",
            metrics={
                "timestamp": now.timestamp(),
                "equity": "2000",
            },
        )

        # Unix timestamp (int)
        event_store.append_metric(
            bot_id="risk_engine",
            metrics={
                "timestamp": int(now.timestamp()),
                "equity": "3000",
            },
        )

        curve = metrics_calculator.get_equity_curve(days=1)
        assert len(curve) == 3

    def test_get_equity_series_with_time_field(self, metrics_calculator, event_store):
        """Test that 'time' field is used as fallback for timestamp."""
        now = datetime.utcnow()

        event_store.append_metric(
            bot_id="risk_engine",
            metrics={
                "time": now.isoformat(),
                "equity": "1000",
            },
        )

        curve = metrics_calculator.get_equity_curve(days=1)
        assert len(curve) == 1
        assert curve[0][1] == Decimal("1000")

    def test_get_equity_series_skips_invalid_timestamps(self, metrics_calculator, event_store):
        """Test that invalid timestamp formats are skipped."""
        now = datetime.utcnow()

        # Invalid string timestamp
        event_store.append_metric(
            bot_id="risk_engine",
            metrics={
                "timestamp": "invalid",
                "equity": "1000",
            },
        )

        # Valid timestamp
        event_store.append_metric(
            bot_id="risk_engine",
            metrics={
                "timestamp": now.isoformat(),
                "equity": "2000",
            },
        )

        curve = metrics_calculator.get_equity_curve(days=1)
        assert len(curve) == 1
        assert curve[0][1] == Decimal("2000")

    def test_get_equity_series_skips_missing_equity(self, metrics_calculator, event_store):
        """Test that events without equity are skipped."""
        now = datetime.utcnow()

        # Event without equity
        event_store.append_metric(
            bot_id="risk_engine",
            metrics={
                "timestamp": now.isoformat(),
            },
        )

        # Event with equity
        event_store.append_metric(
            bot_id="risk_engine",
            metrics={
                "timestamp": now.isoformat(),
                "equity": "1000",
            },
        )

        curve = metrics_calculator.get_equity_curve(days=1)
        assert len(curve) == 1

    def test_daily_returns_empty_series(self, metrics_calculator):
        """Test daily returns calculation with empty series."""
        returns = metrics_calculator._daily_returns([])
        assert returns == []

    def test_daily_returns_single_point(self, metrics_calculator):
        """Test daily returns with single data point."""
        now = datetime.utcnow()
        series = [(now, Decimal("1000"))]
        returns = metrics_calculator._daily_returns(series)
        assert returns == []

    def test_daily_returns_calculation(self, metrics_calculator):
        """Test daily returns calculation."""
        now = datetime.utcnow()
        series = [
            (now - timedelta(days=2), Decimal("1000")),
            (now - timedelta(days=1), Decimal("1100")),
            (now, Decimal("1050")),
        ]

        returns = metrics_calculator._daily_returns(series)
        assert len(returns) == 2
        # Day 1: (1100 - 1000) / 1000 = 0.1
        assert abs(returns[0] - Decimal("0.1")) < Decimal("0.0001")
        # Day 2: (1050 - 1100) / 1100 ≈ -0.0454
        assert abs(returns[1] - Decimal("-0.045454545")) < Decimal("0.001")

    def test_daily_returns_aggregates_by_day(self, metrics_calculator):
        """Test that returns aggregates multiple points per day to end-of-day."""
        now = datetime.utcnow()
        day1 = now - timedelta(days=1)

        # Multiple data points on same day - should use latest
        series = [
            (day1.replace(hour=9), Decimal("1000")),
            (day1.replace(hour=12), Decimal("1050")),
            (day1.replace(hour=16), Decimal("1100")),  # End of day
            (now.replace(hour=16), Decimal("1200")),
        ]

        returns = metrics_calculator._daily_returns(series)
        assert len(returns) == 1
        # Return should be from 1100 to 1200
        assert abs(returns[0] - Decimal("0.090909")) < Decimal("0.0001")

    def test_calculate_sharpe_insufficient_data(self, metrics_calculator, event_store):
        """Test Sharpe ratio with insufficient data."""
        # No data
        sharpe = metrics_calculator.calculate_sharpe(window_days=30)
        assert sharpe == Decimal("0")

        # Single point
        now = datetime.utcnow()
        event_store.append_metric(
            bot_id="risk_engine",
            metrics={
                "timestamp": now.isoformat(),
                "equity": "1000",
            },
        )
        sharpe = metrics_calculator.calculate_sharpe(window_days=30)
        assert sharpe == Decimal("0")

    def test_calculate_sharpe_with_returns(self, metrics_calculator, event_store):
        """Test Sharpe ratio calculation with valid returns."""
        now = datetime.utcnow()

        # Create equity curve with returns
        equities = [1000, 1050, 1100, 1080, 1150, 1200]
        for i, equity in enumerate(equities):
            timestamp = now - timedelta(days=len(equities) - i - 1)
            event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    "timestamp": timestamp.isoformat(),
                    "equity": str(equity),
                },
            )

        sharpe = metrics_calculator.calculate_sharpe(window_days=30)
        assert isinstance(sharpe, Decimal)
        assert sharpe > 0  # Positive returns should have positive Sharpe

    def test_calculate_sharpe_zero_variance(self, metrics_calculator, event_store):
        """Test Sharpe ratio when variance is zero (flat equity)."""
        now = datetime.utcnow()

        # Flat equity curve
        for i in range(5):
            timestamp = now - timedelta(days=4 - i)
            event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    "timestamp": timestamp.isoformat(),
                    "equity": "1000",
                },
            )

        sharpe = metrics_calculator.calculate_sharpe(window_days=30)
        assert sharpe == Decimal("0")

    def test_calculate_max_drawdown_empty_data(self, metrics_calculator):
        """Test max drawdown with no data."""
        dd, peak_date, trough_date = metrics_calculator.calculate_max_drawdown(window_days=90)
        assert dd == Decimal("0")
        assert isinstance(peak_date, datetime)
        assert isinstance(trough_date, datetime)

    def test_calculate_max_drawdown_no_drawdown(self, metrics_calculator, event_store):
        """Test max drawdown with only increasing equity."""
        now = datetime.utcnow()

        # Only gains
        for i in range(5):
            timestamp = now - timedelta(days=4 - i)
            event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    "timestamp": timestamp.isoformat(),
                    "equity": str(1000 + i * 100),
                },
            )

        dd, peak_date, trough_date = metrics_calculator.calculate_max_drawdown(window_days=90)
        assert dd == Decimal("0")

    def test_calculate_max_drawdown_with_drawdown(self, metrics_calculator, event_store):
        """Test max drawdown calculation with actual drawdown."""
        now = datetime.utcnow()

        # Equity curve: 1000 -> 1200 (peak) -> 900 (trough) -> 1100
        equities = [
            (now - timedelta(days=3), "1000"),
            (now - timedelta(days=2), "1200"),  # Peak
            (now - timedelta(days=1), "900"),  # Trough (25% drawdown)
            (now, "1100"),
        ]

        for timestamp, equity in equities:
            event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    "timestamp": timestamp.isoformat(),
                    "equity": equity,
                },
            )

        dd, peak_date, trough_date = metrics_calculator.calculate_max_drawdown(window_days=90)

        # Drawdown = (1200 - 900) / 1200 = 0.25
        assert abs(dd - Decimal("0.25")) < Decimal("0.0001")
        assert peak_date.date() == (now - timedelta(days=2)).date()
        assert trough_date.date() == (now - timedelta(days=1)).date()

    def test_calculate_max_drawdown_multiple_peaks(self, metrics_calculator, event_store):
        """Test max drawdown with multiple peaks and troughs."""
        now = datetime.utcnow()

        # Multiple peaks, largest drawdown in middle
        equities = [
            (now - timedelta(days=6), "1000"),
            (now - timedelta(days=5), "1100"),  # First peak
            (now - timedelta(days=4), "1050"),  # Small drawdown
            (now - timedelta(days=3), "1200"),  # Highest peak
            (now - timedelta(days=2), "800"),  # Largest drawdown (33.3%)
            (now - timedelta(days=1), "1000"),
            (now, "1150"),
        ]

        for timestamp, equity in equities:
            event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    "timestamp": timestamp.isoformat(),
                    "equity": equity,
                },
            )

        dd, peak_date, trough_date = metrics_calculator.calculate_max_drawdown(window_days=90)

        # Max drawdown should be from 1200 to 800
        expected_dd = (Decimal("1200") - Decimal("800")) / Decimal("1200")
        assert abs(dd - expected_dd) < Decimal("0.0001")

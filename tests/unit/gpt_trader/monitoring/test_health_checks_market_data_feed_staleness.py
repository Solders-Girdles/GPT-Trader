"""Tests for market data feed staleness health check wrapper."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpt_trader.monitoring.health_checks import check_market_data_feed_staleness
from gpt_trader.utilities.time_provider import FakeClock


class TestCheckMarketDataFeedStaleness:
    """Tests for check_market_data_feed_staleness wrapper behavior."""

    def test_unknown_signal_is_treated_as_skipped(self) -> None:
        healthy, details = check_market_data_feed_staleness(None)

        assert healthy is True
        assert details["status"] == "UNKNOWN"
        assert details["skipped"] is True
        assert details["reason"] == "market_data_timestamp_unavailable"

    @pytest.mark.parametrize(
        ("age_seconds", "expected_healthy", "expected_status", "expected_severity"),
        [
            (5.0, True, "OK", "ok"),
            (20.0, False, "WARN", "warning"),
            (40.0, False, "CRIT", "critical"),
        ],
    )
    def test_status_and_severity_follow_staleness_thresholds(
        self,
        age_seconds: float,
        expected_healthy: bool,
        expected_status: str,
        expected_severity: str,
    ) -> None:
        from gpt_trader.monitoring.health_signals import HealthThresholds

        class TimestampService:
            def __init__(self, last_update: datetime) -> None:
                self._last_update = last_update

            def get_last_ticker_timestamp(self) -> datetime:
                return self._last_update

        clock = FakeClock(start_time=1000.0)
        service = TimestampService(
            datetime.fromtimestamp(clock.time() - age_seconds, tz=timezone.utc)
        )
        thresholds = HealthThresholds(
            market_data_staleness_seconds_warn=10.0,
            market_data_staleness_seconds_crit=30.0,
        )

        healthy, details = check_market_data_feed_staleness(
            service,
            thresholds=thresholds,
            time_provider=clock,
        )

        assert healthy is expected_healthy
        assert details["status"] == expected_status
        assert details["severity"] == expected_severity

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from gpt_trader.core import Candle
from gpt_trader.features.data.quality import DataQualityChecker


def _make_candle(
    ts: datetime,
    open_: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    volume: Decimal,
) -> Candle:
    return Candle(ts=ts, open=open_, high=high, low=low, close=close, volume=volume)


class TestCheckCandles:
    def test_detects_gaps_spikes_and_volume_anomalies(self) -> None:
        checker = DataQualityChecker(
            spike_threshold_pct=10.0,
            volume_anomaly_std=2.0,
            min_acceptable_score=0.8,
        )
        base = datetime(2024, 1, 1, 0, 0)
        candles = [
            _make_candle(
                base + timedelta(minutes=i),
                Decimal("100"),
                Decimal("101"),
                Decimal("99"),
                Decimal("100"),
                Decimal("100"),
            )
            for i in range(9)
        ]
        candles.append(
            _make_candle(
                base + timedelta(minutes=16),
                Decimal("130"),
                Decimal("160"),
                Decimal("120"),
                Decimal("150"),
                Decimal("1000"),
            )
        )

        report = checker.check_candles(candles, timedelta(minutes=1))

        assert report.gaps_detected
        assert report.spikes_detected
        assert report.volume_anomalies
        assert report.overall_score < 0.8
        assert report.is_acceptable is False
        assert any(issue.severity == "error" for issue in report.gaps_detected)

    def test_short_series_skips_volume_anomalies(self) -> None:
        checker = DataQualityChecker(volume_anomaly_std=2.0)
        base = datetime(2024, 1, 1, 0, 0)
        candles = [
            _make_candle(
                base + timedelta(minutes=i),
                Decimal("100"),
                Decimal("101"),
                Decimal("99"),
                Decimal("100"),
                Decimal("100"),
            )
            for i in range(5)
        ]

        report = checker.check_candles(candles, timedelta(minutes=1))

        assert report.volume_anomalies == []

    def test_empty_candles_returns_unacceptable_report(self) -> None:
        checker = DataQualityChecker()
        report = checker.check_candles([], timedelta(minutes=1))

        assert report.total_candles == 0
        assert report.overall_score == 0.0
        assert report.is_acceptable is False

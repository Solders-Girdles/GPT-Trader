"""Tests for Prometheus metrics exporter."""

from __future__ import annotations

from gpt_trader.monitoring.metrics_exporter import (
    _format_labels,
    _parse_metric_key,
    format_prometheus,
)
from gpt_trader.monitoring.status_reporter import (
    HEARTBEAT_LAG_BUCKETS,
    HEARTBEAT_LAG_METRIC,
)


class TestParseMetricKey:
    """Tests for _parse_metric_key helper."""

    def test_no_labels(self):
        """Test parsing metric name without labels."""
        name, labels = _parse_metric_key("gpt_trader_equity_dollars")
        assert name == "gpt_trader_equity_dollars"
        assert labels == {}

    def test_single_label(self):
        """Test parsing metric with single label."""
        name, labels = _parse_metric_key("gpt_trader_order_total{result=success}")
        assert name == "gpt_trader_order_total"
        assert labels == {"result": "success"}

    def test_multiple_labels(self):
        """Test parsing metric with multiple labels."""
        name, labels = _parse_metric_key(
            "gpt_trader_order_total{reason=none,result=success,side=buy}"
        )
        assert name == "gpt_trader_order_total"
        assert labels == {"reason": "none", "result": "success", "side": "buy"}


class TestFormatLabels:
    """Tests for _format_labels helper."""

    def test_empty_labels(self):
        """Test formatting empty labels returns empty string."""
        assert _format_labels({}) == ""

    def test_single_label(self):
        """Test formatting single label."""
        result = _format_labels({"result": "success"})
        assert result == '{result="success"}'

    def test_multiple_labels_sorted(self):
        """Test that labels are sorted alphabetically."""
        result = _format_labels({"side": "buy", "result": "success"})
        assert result == '{result="success",side="buy"}'


class TestFormatPrometheus:
    """Tests for format_prometheus function."""

    def test_empty_summary(self):
        """Test formatting empty metrics summary."""
        result = format_prometheus({})
        assert result == ""

    def test_counter_formatting(self):
        """Test counter metric formatting."""
        summary = {
            "counters": {
                "gpt_trader_order_total{result=success,side=buy}": 10,
                "gpt_trader_order_total{result=failed,side=buy}": 2,
            }
        }
        result = format_prometheus(summary)

        assert "# TYPE gpt_trader_order_total counter" in result
        assert 'gpt_trader_order_total{result="failed",side="buy"} 2' in result
        assert 'gpt_trader_order_total{result="success",side="buy"} 10' in result
        # TYPE should only appear once
        assert result.count("# TYPE gpt_trader_order_total counter") == 1

    def test_gauge_formatting(self):
        """Test gauge metric formatting."""
        summary = {
            "gauges": {
                "gpt_trader_equity_dollars": 10500.5,
                "gpt_trader_ws_gap_count": 0,
            }
        }
        result = format_prometheus(summary)

        assert "# TYPE gpt_trader_equity_dollars gauge" in result
        assert "gpt_trader_equity_dollars 10500.5" in result
        assert "# TYPE gpt_trader_ws_gap_count gauge" in result
        assert "gpt_trader_ws_gap_count 0" in result

    def test_histogram_formatting(self):
        """Test histogram metric formatting with buckets."""
        summary = {
            "histograms": {
                "gpt_trader_cycle_duration_seconds{result=ok}": {
                    "count": 100,
                    "sum": 45.5,
                    "buckets": {"0.1": 20, "0.5": 65, "1.0": 13, "5.0": 2},
                }
            }
        }
        result = format_prometheus(summary)

        # Check TYPE declaration
        assert "# TYPE gpt_trader_cycle_duration_seconds histogram" in result

        # Check bucket format (cumulative counts)
        assert 'gpt_trader_cycle_duration_seconds_bucket{le="0.1",result="ok"} 20' in result
        assert 'gpt_trader_cycle_duration_seconds_bucket{le="0.5",result="ok"} 85' in result
        assert 'gpt_trader_cycle_duration_seconds_bucket{le="1.0",result="ok"} 98' in result
        assert 'gpt_trader_cycle_duration_seconds_bucket{le="5.0",result="ok"} 100' in result
        assert 'gpt_trader_cycle_duration_seconds_bucket{le="+Inf",result="ok"} 100' in result

        # Check sum and count
        assert 'gpt_trader_cycle_duration_seconds_sum{result="ok"} 45.5' in result
        assert 'gpt_trader_cycle_duration_seconds_count{result="ok"} 100' in result

    def test_combined_metrics(self):
        """Test formatting of combined counters, gauges, and histograms."""
        summary = {
            "counters": {"gpt_trader_order_total{result=success}": 5},
            "gauges": {"gpt_trader_equity_dollars": 1000.0},
            "histograms": {
                "gpt_trader_latency_seconds": {
                    "count": 10,
                    "sum": 1.5,
                    "buckets": {"0.1": 5, "1.0": 5},
                }
            },
        }
        result = format_prometheus(summary)

        # Verify all types present
        assert "# TYPE gpt_trader_order_total counter" in result
        assert "# TYPE gpt_trader_equity_dollars gauge" in result
        assert "# TYPE gpt_trader_latency_seconds histogram" in result

    def test_trailing_newline(self):
        """Test that non-empty output ends with newline."""
        summary = {"gauges": {"test_metric": 1.0}}
        result = format_prometheus(summary)
        assert result.endswith("\n")

    def test_metric_without_labels(self):
        """Test formatting metric without labels."""
        summary = {"counters": {"simple_counter": 42}}
        result = format_prometheus(summary)
        assert "simple_counter 42" in result


def _build_heartbeat_histogram_summary(
    bucket_counts: tuple[int, ...],
    total: float,
) -> dict[str, dict[str, object]]:
    """Helper for building heartbeat histogram summaries."""

    assert len(bucket_counts) == len(HEARTBEAT_LAG_BUCKETS)
    bucket_map = {
        str(bound): count
        for bound, count in zip(HEARTBEAT_LAG_BUCKETS, bucket_counts)
    }
    count = sum(bucket_counts)
    return {
        "histograms": {
            HEARTBEAT_LAG_METRIC: {
                "count": count,
                "sum": total,
                "buckets": bucket_map,
            }
        }
    }


def _expected_heartbeat_bucket_lines(bucket_counts: tuple[int, ...]) -> list[str]:
    """Expected bucket lines for heartbeat lag histograms."""

    expected: list[str] = []
    cumulative = 0
    for bound, bucket in zip(HEARTBEAT_LAG_BUCKETS, bucket_counts):
        cumulative += bucket
        expected.append(
            f'{HEARTBEAT_LAG_METRIC}_bucket{{le="{bound}"}} {cumulative}'
        )
    expected.append(
        f'{HEARTBEAT_LAG_METRIC}_bucket{{le="+Inf"}} {sum(bucket_counts)}'
    )
    return expected


class TestHeartbeatHistogramExport:
    """Heart beat histogram coverage for metric exporter."""

    def test_zero_data_remains_deterministic(self) -> None:
        """Zero-sample histogram emits deterministic bucket exposure."""

        bucket_counts = (0,) * len(HEARTBEAT_LAG_BUCKETS)
        summary = _build_heartbeat_histogram_summary(bucket_counts, total=0.0)
        result = format_prometheus(summary)

        assert "# TYPE gpt_trader_ws_heartbeat_lag_seconds histogram" in result
        bucket_lines = [
            line
            for line in result.splitlines()
            if line.startswith(f"{HEARTBEAT_LAG_METRIC}_bucket")
        ]
        assert len(bucket_lines) == len(HEARTBEAT_LAG_BUCKETS) + 1
        assert all(line.endswith(" 0") for line in bucket_lines)
        assert f"{HEARTBEAT_LAG_METRIC}_sum 0.0" in result
        assert f"{HEARTBEAT_LAG_METRIC}_count 0" in result

    def test_populated_data_exposes_bucket_boundaries(self) -> None:
        """Populated histogram highlights each bucket boundary and units."""

        bucket_counts = (1, 0, 2, 1, 0, 3, 0, 0, 2, 0, 0, 1, 0)
        total = 12.7
        summary = _build_heartbeat_histogram_summary(bucket_counts, total=total)
        result = format_prometheus(summary)

        assert "# TYPE gpt_trader_ws_heartbeat_lag_seconds histogram" in result
        for line in _expected_heartbeat_bucket_lines(bucket_counts):
            assert line in result
        assert f"{HEARTBEAT_LAG_METRIC}_sum {total}" in result
        assert f"{HEARTBEAT_LAG_METRIC}_count {sum(bucket_counts)}" in result

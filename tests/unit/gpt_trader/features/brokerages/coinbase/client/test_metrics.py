"""Tests for API Metrics Collection."""

import pytest

from gpt_trader.features.brokerages.coinbase.client.metrics import (
    EndpointMetrics,
)


class TestEndpointMetrics:
    """Tests for EndpointMetrics class."""

    def test_initial_values(self) -> None:
        """Test initial metric values."""
        metrics = EndpointMetrics()

        assert metrics.total_calls == 0
        assert metrics.total_errors == 0
        assert metrics.average_latency_ms == 0.0
        assert metrics.error_rate == 0.0

    def test_record_updates_metrics(self) -> None:
        """Test that recording updates all metrics."""
        metrics = EndpointMetrics()

        metrics.record(100.0)
        assert metrics.total_calls == 1
        assert metrics.total_latency_ms == 100.0
        assert metrics.last_latency_ms == 100.0
        assert metrics.max_latency_ms == 100.0
        assert metrics.min_latency_ms == 100.0

        metrics.record(200.0)
        assert metrics.total_calls == 2
        assert metrics.average_latency_ms == 150.0
        assert metrics.last_latency_ms == 200.0
        assert metrics.max_latency_ms == 200.0
        assert metrics.min_latency_ms == 100.0

    def test_error_tracking(self) -> None:
        """Test error rate calculation."""
        metrics = EndpointMetrics()

        metrics.record(100.0, error=False)
        metrics.record(100.0, error=True)
        metrics.record(100.0, error=False)

        assert metrics.total_calls == 3
        assert metrics.total_errors == 1
        assert metrics.error_rate == pytest.approx(1 / 3)

    def test_percentile_calculations(self) -> None:
        """Test latency percentile calculations."""
        metrics = EndpointMetrics()

        # Record latencies 1-100
        for i in range(1, 101):
            metrics.record(float(i))

        # P50 should be around 50
        assert 49 <= metrics.p50_latency_ms <= 51

        # P95 should be around 95
        assert 94 <= metrics.p95_latency_ms <= 96

        # P99 should be around 99
        assert 98 <= metrics.p99_latency_ms <= 100

    def test_to_dict(self) -> None:
        """Test dictionary serialization."""
        metrics = EndpointMetrics()
        metrics.record(100.0)
        metrics.record(200.0, error=True)

        result = metrics.to_dict()

        assert result["total_calls"] == 2
        assert result["total_errors"] == 1
        assert result["error_rate"] == 0.5
        assert result["avg_latency_ms"] == 150.0

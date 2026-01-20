"""Tests for EndpointMetrics and endpoint categorization."""

import pytest

from gpt_trader.features.brokerages.coinbase.client.metrics import (
    EndpointMetrics,
    categorize_endpoint,
)

# ============================================================================
# EndpointMetrics Tests
# ============================================================================


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

        for i in range(1, 101):
            metrics.record(float(i))

        assert 49 <= metrics.p50_latency_ms <= 51
        assert 94 <= metrics.p95_latency_ms <= 96
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

    def test_to_dict_zeroed_defaults(self) -> None:
        """Test dictionary serialization with default values."""
        metrics = EndpointMetrics()
        data = metrics.to_dict()

        assert data["total_calls"] == 0
        assert data["total_errors"] == 0
        assert data["error_rate"] == 0.0
        assert data["avg_latency_ms"] == 0.0
        assert data["last_latency_ms"] == 0.0
        assert data["min_latency_ms"] == 0.0
        assert data["max_latency_ms"] == 0.0
        assert data["p50_latency_ms"] == 0.0
        assert data["p95_latency_ms"] == 0.0
        assert data["p99_latency_ms"] == 0.0


# ============================================================================
# categorize_endpoint Tests
# ============================================================================


def test_categorize_endpoint_case_insensitive() -> None:
    """Test endpoint categorization is case insensitive."""
    assert categorize_endpoint("/API/V3/BROKERAGE/ORDERS") == "orders"
    assert categorize_endpoint("/api/v3/brokerage/MARKET/TICKER") == "market"
    assert categorize_endpoint("/api/v3/brokerage/positions") == "positions"
    assert categorize_endpoint("/api/v3/brokerage/unknown") == "other"

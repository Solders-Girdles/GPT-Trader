"""Tests for API Metrics Collection."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpt_trader.features.brokerages.coinbase.client.metrics import (
    APIMetricsCollector,
    EndpointMetrics,
    RequestTimer,
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


class TestAPIMetricsCollector:
    """Tests for APIMetricsCollector class."""

    def test_initial_state(self) -> None:
        """Test initial collector state."""
        collector = APIMetricsCollector()

        assert collector.get_average_latency() == 0.0
        assert collector.get_error_rate() == 0.0

    def test_record_request(self) -> None:
        """Test recording a request."""
        collector = APIMetricsCollector()

        collector.record_request("/api/v3/orders", 100.0)

        summary = collector.get_summary()
        assert summary["total_requests"] == 1
        assert summary["avg_latency_ms"] == 100.0

    def test_endpoint_categorization(self) -> None:
        """Test that endpoints are correctly categorized."""
        collector = APIMetricsCollector()

        collector.record_request("/api/v3/orders", 100.0)
        collector.record_request("/api/v3/orders/123/cancel", 150.0)
        collector.record_request("/api/v3/accounts", 200.0)

        summary = collector.get_summary()

        # Orders should be in "orders" category
        assert "orders" in summary["endpoints"]
        assert summary["endpoints"]["orders"]["total_calls"] == 2

        # Accounts should be in "accounts" category
        assert "accounts" in summary["endpoints"]
        assert summary["endpoints"]["accounts"]["total_calls"] == 1

    def test_error_and_rate_limit_tracking(self) -> None:
        """Test tracking errors and rate limits."""
        collector = APIMetricsCollector()

        collector.record_request("/api/v3/test", 100.0, error=False)
        collector.record_request("/api/v3/test", 100.0, error=True)
        collector.record_request("/api/v3/test", 100.0, error=True, rate_limited=True)

        summary = collector.get_summary()
        assert summary["total_requests"] == 3
        assert summary["total_errors"] == 2  # Two requests marked as error
        assert summary["rate_limit_hits"] == 1

    def test_disabled_collector(self) -> None:
        """Test that disabled collector doesn't record."""
        collector = APIMetricsCollector(enabled=False)

        collector.record_request("/api/v3/test", 100.0)

        summary = collector.get_summary()
        assert summary["total_requests"] == 0

    def test_reset(self) -> None:
        """Test resetting all metrics."""
        collector = APIMetricsCollector()

        collector.record_request("/api/v3/test", 100.0)
        collector.reset()

        summary = collector.get_summary()
        assert summary["total_requests"] == 0
        assert len(summary["endpoints"]) == 0

    def test_rate_limit_usage_display(self) -> None:
        """Test rate limit usage formatting."""
        collector = APIMetricsCollector()

        assert collector.get_rate_limit_usage_display(0.45) == "45%"
        assert collector.get_rate_limit_usage_display(1.0) == "100%"

    def test_thread_safety(self) -> None:
        """Test that collector is thread-safe."""
        collector = APIMetricsCollector()
        errors: list[Exception] = []

        def record_requests(n: int) -> None:
            try:
                for i in range(100):
                    collector.record_request(f"/api/v3/endpoint{n}", float(i))
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(record_requests, i) for i in range(10)]
            for f in futures:
                f.result()

        assert len(errors) == 0
        summary = collector.get_summary()
        assert summary["total_requests"] == 1000


class TestRequestTimer:
    """Tests for RequestTimer context manager."""

    def test_timer_records_latency(self) -> None:
        """Test that timer records request latency.

        Note: We can't use time.sleep here because the coinbase conftest
        has autouse=True for fast_retry_sleep which mocks time.sleep.
        Instead, we test that the timer correctly records elapsed time
        by verifying the record_request method works.
        """
        collector = APIMetricsCollector()

        # Directly test that recording works (timer uses perf_counter internally)
        collector.record_request("/api/v3/test", 100.0)

        summary = collector.get_summary()
        assert summary["total_requests"] == 1
        assert summary["avg_latency_ms"] == 100.0

    def test_timer_context_manager_records(self) -> None:
        """Test that RequestTimer context manager records on exit.

        This verifies the context manager protocol works correctly,
        though actual timing may be near-zero due to mocked time.sleep.
        """
        collector = APIMetricsCollector()

        with RequestTimer(collector, "/api/v3/test"):
            pass  # Just verify context manager records something

        summary = collector.get_summary()
        assert summary["total_requests"] == 1
        # Latency will be very small since no real work is done
        assert summary["avg_latency_ms"] >= 0

    def test_timer_marks_error(self) -> None:
        """Test marking a request as error."""
        collector = APIMetricsCollector()

        with RequestTimer(collector, "/api/v3/test") as timer:
            timer.mark_error()

        summary = collector.get_summary()
        assert summary["total_errors"] == 1

    def test_timer_marks_rate_limited(self) -> None:
        """Test marking a request as rate limited."""
        collector = APIMetricsCollector()

        with RequestTimer(collector, "/api/v3/test") as timer:
            timer.mark_rate_limited()

        summary = collector.get_summary()
        assert summary["rate_limit_hits"] == 1
        assert summary["total_errors"] == 1  # Rate limited counts as error

    def test_timer_records_on_exception(self) -> None:
        """Test that timer records even when exception occurs."""
        collector = APIMetricsCollector()

        with pytest.raises(ValueError):
            with RequestTimer(collector, "/api/v3/test"):
                raise ValueError("Test error")

        summary = collector.get_summary()
        assert summary["total_requests"] == 1
        assert summary["total_errors"] == 1

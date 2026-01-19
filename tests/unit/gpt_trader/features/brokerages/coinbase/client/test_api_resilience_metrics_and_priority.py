"""Tests for Coinbase API metrics and request priority behavior."""

from gpt_trader.features.brokerages.coinbase.client.metrics import APIMetricsCollector
from gpt_trader.features.brokerages.coinbase.client.priority import (
    PriorityManager,
    RequestPriority,
)


class TestAPIResilienceMetricsAndPriority:
    def test_priority_blocks_low_under_pressure(self) -> None:
        """Test that low priority requests are blocked under rate limit pressure."""
        priority_manager = PriorityManager(
            threshold_high=0.7,
            threshold_critical=0.85,
            enabled=True,
        )

        products_path = "/api/v3/products"
        orders_path = "/api/v3/orders"

        assert priority_manager.get_priority(products_path) == RequestPriority.LOW
        assert priority_manager.get_priority(orders_path) == RequestPriority.CRITICAL

        assert priority_manager.should_allow(products_path, 0.80) is False
        assert priority_manager.should_allow(orders_path, 0.80) is True

    def test_metrics_tracks_across_endpoints(self) -> None:
        """Test that metrics correctly aggregate across different endpoints."""
        metrics = APIMetricsCollector()

        metrics.record_request("/api/v3/orders", 100.0)
        metrics.record_request("/api/v3/orders/123", 150.0, error=True)
        metrics.record_request("/api/v3/accounts", 200.0)
        metrics.record_request("/api/v3/products", 50.0)

        summary = metrics.get_summary()

        assert summary["total_requests"] == 4
        assert summary["total_errors"] == 1
        assert summary["avg_latency_ms"] == 125.0

        assert summary["endpoints"]["orders"]["total_calls"] == 2
        assert summary["endpoints"]["orders"]["total_errors"] == 1
        assert summary["endpoints"]["accounts"]["total_calls"] == 1
        assert summary["endpoints"]["products"]["total_calls"] == 1

    def test_rate_limit_tracking(self) -> None:
        """Test that rate limit hits are tracked in metrics."""
        metrics = APIMetricsCollector(enabled=True)

        metrics.record_request("/api/v3/orders", 100.0)
        metrics.record_request("/api/v3/orders", 100.0)

        metrics.record_request("/api/v3/orders", 50.0, error=True, rate_limited=True)

        summary = metrics.get_summary()
        assert summary["total_requests"] == 3
        assert summary["total_errors"] == 1
        assert summary["rate_limit_hits"] == 1

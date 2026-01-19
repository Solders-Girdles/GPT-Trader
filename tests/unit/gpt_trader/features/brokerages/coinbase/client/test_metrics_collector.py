"""Tests for APIMetricsCollector."""

from concurrent.futures import ThreadPoolExecutor

from gpt_trader.features.brokerages.coinbase.client.metrics import APIMetricsCollector


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

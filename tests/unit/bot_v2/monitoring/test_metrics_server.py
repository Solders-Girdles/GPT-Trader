"""Unit tests for MetricsServer."""

from __future__ import annotations

import json
import socket
import time
from unittest.mock import MagicMock, patch

import pytest
import requests
from prometheus_client import CollectorRegistry

from bot_v2.monitoring.metrics_server import MetricsServer


def find_free_port() -> int:
    """Find a free port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture
def test_port() -> int:
    """Provide a free port for testing."""
    return find_free_port()


@pytest.fixture
def metrics_server(test_port: int) -> MetricsServer:
    """Create a MetricsServer instance with isolated registry."""
    registry = CollectorRegistry()
    server = MetricsServer(host="127.0.0.1", port=test_port, registry=registry)
    yield server
    if server.is_running:
        server.stop()


class TestMetricsServerStartStop:
    """Tests for server lifecycle management."""

    def test_server_starts_successfully(self, metrics_server: MetricsServer) -> None:
        """Server starts and becomes available."""
        assert not metrics_server.is_running

        metrics_server.start()
        time.sleep(0.1)  # Allow server thread to start

        assert metrics_server.is_running

    def test_server_stops_successfully(self, metrics_server: MetricsServer) -> None:
        """Server stops cleanly."""
        metrics_server.start()
        time.sleep(0.1)

        metrics_server.stop()

        assert not metrics_server.is_running

    def test_double_start_is_safe(self, metrics_server: MetricsServer) -> None:
        """Starting an already running server is safe."""
        metrics_server.start()
        time.sleep(0.1)

        # Should not raise
        metrics_server.start()

        assert metrics_server.is_running

    def test_double_stop_is_safe(self, metrics_server: MetricsServer) -> None:
        """Stopping an already stopped server is safe."""
        metrics_server.start()
        time.sleep(0.1)
        metrics_server.stop()

        # Should not raise
        metrics_server.stop()

        assert not metrics_server.is_running


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_endpoint_returns_200(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /metrics returns HTTP 200."""
        metrics_server.start()
        time.sleep(0.1)

        response = requests.get(f"http://127.0.0.1:{test_port}/metrics", timeout=2)

        assert response.status_code == 200

    def test_metrics_endpoint_content_type(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /metrics returns Prometheus text format."""
        metrics_server.start()
        time.sleep(0.1)

        response = requests.get(f"http://127.0.0.1:{test_port}/metrics", timeout=2)

        assert "text/plain" in response.headers["Content-Type"]
        assert "version=0.0.4" in response.headers["Content-Type"]

    def test_metrics_endpoint_includes_uptime(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /metrics includes bot_uptime_seconds metric."""
        metrics_server.start()
        time.sleep(0.1)

        # Update uptime
        metrics_server.update_uptime(profile="test")

        response = requests.get(f"http://127.0.0.1:{test_port}/metrics", timeout=2)

        assert "bot_uptime_seconds" in response.text

    def test_metrics_endpoint_includes_cycle_duration(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /metrics includes bot_cycle_duration_seconds histogram."""
        metrics_server.start()
        time.sleep(0.1)

        # Record some cycle durations
        metrics_server.record_cycle_duration(1.5, profile="test")
        metrics_server.record_cycle_duration(2.0, profile="test")

        response = requests.get(f"http://127.0.0.1:{test_port}/metrics", timeout=2)

        assert "bot_cycle_duration_seconds" in response.text

    def test_metrics_endpoint_includes_order_attempts(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /metrics includes bot_order_attempts_total counter."""
        metrics_server.start()
        time.sleep(0.1)

        # Record some order attempts
        metrics_server.record_order_attempt("attempted", profile="test")
        metrics_server.record_order_attempt("success", profile="test")

        response = requests.get(f"http://127.0.0.1:{test_port}/metrics", timeout=2)

        assert "bot_order_attempts_total" in response.text

    def test_metrics_endpoint_includes_errors(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /metrics includes bot_errors_total counter."""
        metrics_server.start()
        time.sleep(0.1)

        # Record some errors
        metrics_server.record_error("execution", "warning", profile="test")
        metrics_server.record_error("execution", "error", profile="test")

        response = requests.get(f"http://127.0.0.1:{test_port}/metrics", timeout=2)

        assert "bot_errors_total" in response.text


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_endpoint_returns_200(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /health returns HTTP 200."""
        metrics_server.start()
        time.sleep(0.1)

        response = requests.get(f"http://127.0.0.1:{test_port}/health", timeout=2)

        assert response.status_code == 200

    def test_health_endpoint_returns_json(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /health returns JSON content."""
        metrics_server.start()
        time.sleep(0.1)

        response = requests.get(f"http://127.0.0.1:{test_port}/health", timeout=2)

        assert response.headers["Content-Type"] == "application/json"

        # Should parse as valid JSON
        health_data = response.json()
        assert isinstance(health_data, dict)

    def test_health_endpoint_includes_status(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /health includes status field."""
        metrics_server.start()
        time.sleep(0.1)

        response = requests.get(f"http://127.0.0.1:{test_port}/health", timeout=2)
        health_data = response.json()

        assert "status" in health_data
        assert health_data["status"] == "ok"

    def test_health_endpoint_includes_uptime(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /health includes uptime_seconds field."""
        metrics_server.start()
        time.sleep(0.1)

        response = requests.get(f"http://127.0.0.1:{test_port}/health", timeout=2)
        health_data = response.json()

        assert "uptime_seconds" in health_data
        assert isinstance(health_data["uptime_seconds"], (int, float))
        assert health_data["uptime_seconds"] >= 0

    def test_health_endpoint_includes_background_tasks(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /health includes background_tasks_running field."""
        metrics_server.start()
        time.sleep(0.1)

        response = requests.get(f"http://127.0.0.1:{test_port}/health", timeout=2)
        health_data = response.json()

        assert "background_tasks_running" in health_data
        assert isinstance(health_data["background_tasks_running"], int)

    def test_health_endpoint_includes_last_cycle_timestamp(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /health includes last_cycle_timestamp field."""
        metrics_server.start()
        time.sleep(0.1)

        # Record a cycle
        metrics_server.record_cycle_duration(1.0, profile="test")

        response = requests.get(f"http://127.0.0.1:{test_port}/health", timeout=2)
        health_data = response.json()

        assert "last_cycle_timestamp" in health_data

    def test_health_endpoint_includes_streaming_status(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /health includes streaming_connected field."""
        metrics_server.start()
        time.sleep(0.1)

        response = requests.get(f"http://127.0.0.1:{test_port}/health", timeout=2)
        health_data = response.json()

        assert "streaming_connected" in health_data
        assert isinstance(health_data["streaming_connected"], bool)

    def test_health_endpoint_includes_streaming_details(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /health includes detailed streaming diagnostics."""
        metrics_server.start()
        time.sleep(0.1)

        metrics_server.set_streaming_context("test_stream", profile="test")
        metrics_server.update_streaming_status(True, profile="test", stream="test_stream")
        metrics_server.record_streaming_message(0.2, profile="test", stream="test_stream")
        metrics_server.record_streaming_reconnect(
            "attempt", attempt=1, profile="test", stream="test_stream"
        )

        response = requests.get(f"http://127.0.0.1:{test_port}/health", timeout=2)
        health_data = response.json()

        assert "streaming" in health_data
        streaming_info = health_data["streaming"]
        assert streaming_info["stream_name"] == "test_stream"
        assert streaming_info["connected"] is True
        assert "heartbeat_lag_seconds" in streaming_info
        assert "reconnect_attempts_total" in streaming_info
        assert streaming_info.get("fallback_active") is False

    def test_update_streaming_fallback_sets_gauge(self, metrics_server: MetricsServer) -> None:
        """Updating fallback state sets the gauge value."""

        metrics_server.set_streaming_context("test_stream", profile="test")
        metrics_server.update_streaming_fallback(True, profile="test", stream="test_stream")

        gauge_value = metrics_server.streaming_fallback_gauge.labels(
            profile="test", stream="test_stream"
        )._value.get()
        assert gauge_value == 1

        metrics_server.update_streaming_fallback(False, profile="test", stream="test_stream")
        gauge_value = metrics_server.streaming_fallback_gauge.labels(
            profile="test", stream="test_stream"
        )._value.get()
        assert gauge_value == 0


class TestMetricsRecording:
    """Tests for metric recording functionality."""

    def test_record_cycle_duration_updates_histogram(self, metrics_server: MetricsServer) -> None:
        """Recording cycle duration updates the histogram."""
        # Record multiple durations
        metrics_server.record_cycle_duration(1.5, profile="test")
        metrics_server.record_cycle_duration(2.0, profile="test")
        metrics_server.record_cycle_duration(0.5, profile="test")

        # Verify by checking the exported metrics contain the histogram
        from prometheus_client import generate_latest

        metrics_output = generate_latest(metrics_server.registry).decode("utf-8")

        # Should have histogram metrics
        assert "bot_cycle_duration_seconds_count" in metrics_output
        assert 'profile="test"' in metrics_output

    def test_update_background_tasks_sets_gauge(self, metrics_server: MetricsServer) -> None:
        """Updating background tasks sets the gauge value."""
        metrics_server.update_background_tasks("runtime_guards", 3, profile="test")

        gauge_value = metrics_server.background_tasks_gauge.labels(
            profile="test", task="runtime_guards"
        )._value.get()
        assert gauge_value == 3

    def test_record_order_attempt_increments_counter(self, metrics_server: MetricsServer) -> None:
        """Recording order attempts increments the counter."""
        initial_count = metrics_server.order_attempts_counter.labels(
            profile="test", status="success"
        )._value.get()

        metrics_server.record_order_attempt("success", profile="test")

        final_count = metrics_server.order_attempts_counter.labels(
            profile="test", status="success"
        )._value.get()
        assert final_count == initial_count + 1

    def test_record_error_increments_counter(self, metrics_server: MetricsServer) -> None:
        """Recording errors increments the counter."""
        initial_count = metrics_server.errors_counter.labels(
            component="execution", profile="test", severity="error"
        )._value.get()

        metrics_server.record_error("execution", "error", profile="test")

        final_count = metrics_server.errors_counter.labels(
            component="execution", profile="test", severity="error"
        )._value.get()
        assert final_count == initial_count + 1

    def test_update_streaming_status(self, metrics_server: MetricsServer) -> None:
        """Updating streaming status updates gauge and internal state."""
        metrics_server.set_streaming_context("test_stream", profile="test")

        metrics_server.update_streaming_status(True)

        gauge_value = metrics_server.streaming_connection_gauge.labels(
            profile="test", stream="test_stream"
        )._value.get()
        assert gauge_value == 1.0
        assert metrics_server._streaming_connected is True

        metrics_server.update_streaming_status(False)

        gauge_value = metrics_server.streaming_connection_gauge.labels(
            profile="test", stream="test_stream"
        )._value.get()
        assert gauge_value == 0.0
        assert metrics_server._streaming_connected is False

    def test_record_streaming_message_updates_metrics(self, metrics_server: MetricsServer) -> None:
        """Recording streaming message updates lag gauge and histogram."""
        metrics_server.set_streaming_context("test_stream", profile="test")

        metrics_server.record_streaming_message(0.25)

        lag_value = metrics_server.streaming_heartbeat_lag_gauge.labels(
            profile="test", stream="test_stream"
        )._value.get()
        assert lag_value == pytest.approx(0.25)

        histogram = metrics_server.streaming_message_gap_histogram
        bucket_count = histogram.labels(profile="test", stream="test_stream")._sum.get()
        assert bucket_count == pytest.approx(0.25)

    def test_record_streaming_reconnect_updates_counters(
        self, metrics_server: MetricsServer
    ) -> None:
        """Recording reconnect attempts increments counters and updates status."""
        metrics_server.set_streaming_context("test_stream", profile="test")

        metrics_server.record_streaming_reconnect("attempt", attempt=1)
        attempt_count = metrics_server.streaming_reconnect_counter.labels(
            profile="test", stream="test_stream", status="attempt"
        )._value.get()
        assert attempt_count == 1

        metrics_server.record_streaming_reconnect("success", attempt=1)
        success_count = metrics_server.streaming_reconnect_counter.labels(
            profile="test", stream="test_stream", status="success"
        )._value.get()
        assert success_count == 1

        gauge_value = metrics_server.streaming_connection_gauge.labels(
            profile="test", stream="test_stream"
        )._value.get()
        assert gauge_value == 1.0


class TestSystemMetrics:
    """Tests for system resource metrics."""

    @patch("psutil.Process")
    def test_update_system_metrics_sets_memory_gauge(
        self, mock_process_class: MagicMock, metrics_server: MetricsServer
    ) -> None:
        """Updating system metrics sets memory gauge."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=1024 * 1024 * 100)  # 100 MB
        mock_process.cpu_percent.return_value = 25.5
        metrics_server._process = mock_process

        metrics_server._update_system_metrics()

        memory_value = metrics_server.memory_gauge.labels(profile="default")._value.get()
        assert memory_value == 1024 * 1024 * 100

    @patch("psutil.Process")
    def test_update_system_metrics_sets_cpu_gauge(
        self, mock_process_class: MagicMock, metrics_server: MetricsServer
    ) -> None:
        """Updating system metrics sets CPU gauge."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=1024)
        mock_process.cpu_percent.return_value = 42.5
        metrics_server._process = mock_process

        metrics_server._update_system_metrics()

        cpu_value = metrics_server.cpu_gauge.labels(profile="default")._value.get()
        assert cpu_value == 42.5


class TestNotFoundEndpoint:
    """Tests for 404 handling."""

    def test_unknown_endpoint_returns_404(
        self, metrics_server: MetricsServer, test_port: int
    ) -> None:
        """GET /unknown returns HTTP 404."""
        metrics_server.start()
        time.sleep(0.1)

        response = requests.get(f"http://127.0.0.1:{test_port}/unknown", timeout=2)

        assert response.status_code == 404


class TestGuardMetrics:
    """Tests for guard rail metric integration."""

    def test_register_guard_manager_updates_metrics(self) -> None:
        from bot_v2.orchestration.guardrails import GuardRailManager

        registry = CollectorRegistry()
        server = MetricsServer(host="127.0.0.1", port=0, registry=registry)
        guards = GuardRailManager()
        guards.set_dry_run(True)

        server.register_guard_manager(guards, profile="test")

        guard_value = server.guard_state_gauge.labels(profile="test", guard="dry_run")._value.get()
        assert guard_value == 1.0

    def test_update_error_streak_sets_gauge(self) -> None:
        registry = CollectorRegistry()
        server = MetricsServer(host="127.0.0.1", port=0, registry=registry)
        server.update_error_streak(3, profile="test")
        value = server.error_streak_gauge.labels(profile="test")._value.get()
        assert value == 3

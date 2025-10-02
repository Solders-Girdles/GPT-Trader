from __future__ import annotations

from datetime import datetime

import pytest

from bot_v2.monitoring.system import collectors


@pytest.fixture(autouse=True)
def _patch_random(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_random_values = {
        "uniform": 1.0,
        "random": 0.2,
        "randint": 42,
    }

    monkeypatch.setattr(collectors.random, "uniform", lambda a, b=None: mock_random_values["uniform"])
    monkeypatch.setattr(collectors.random, "random", lambda: mock_random_values["random"])
    monkeypatch.setattr(collectors.random, "randint", lambda a, b=None: mock_random_values["randint"])


def test_resource_collector_collects_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMemory:
        percent = 23.0
        used = 50 * 1024 * 1024

    class DummyDisk:
        percent = 61.0
        used = 10 * 1024 * 1024 * 1024

    class DummyNetwork:
        bytes_sent = 1024 * 1024
        bytes_recv = 2 * 1024 * 1024

    class DummyProcess:
        def open_files(self):
            return [object()]

        def num_threads(self):
            return 7

    monkeypatch.setattr(collectors.psutil, "cpu_percent", lambda interval=0.0: 12.5)
    monkeypatch.setattr(collectors.psutil, "virtual_memory", lambda: DummyMemory())
    monkeypatch.setattr(collectors.psutil, "disk_usage", lambda _: DummyDisk())
    monkeypatch.setattr(collectors.psutil, "net_io_counters", lambda: DummyNetwork())
    monkeypatch.setattr(collectors.psutil, "Process", lambda: DummyProcess())

    resource = collectors.ResourceCollector().collect()

    assert resource.cpu_percent == 12.5
    assert resource.memory_percent == 23.0
    assert resource.disk_gb == pytest.approx(10.0)
    assert resource.network_sent_mb == pytest.approx(1.0, rel=1e-6)
    assert resource.open_files == 1
    assert resource.threads == 7


def test_resource_collector_handles_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_error(*_: object, **__: object) -> None:
        raise RuntimeError("psutil failure")

    monkeypatch.setattr(collectors.psutil, "cpu_percent", raise_error)
    monkeypatch.setattr(collectors.psutil, "virtual_memory", raise_error)
    monkeypatch.setattr(collectors.psutil, "disk_usage", raise_error)
    monkeypatch.setattr(collectors.psutil, "net_io_counters", raise_error)
    monkeypatch.setattr(collectors.psutil, "Process", raise_error)

    resource = collectors.ResourceCollector().collect()
    assert resource.cpu_percent == 0.0
    assert resource.memory_percent == 0.0
    assert resource.disk_percent == 0.0
    assert resource.network_recv_mb == 0.0
    assert resource.threads == 0


def test_performance_collector_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = collectors.PerformanceCollector()
    metrics = collector.collect()

    assert metrics.requests_per_second == 100.0
    assert metrics.avg_response_time_ms == 50.0
    assert metrics.p95_response_time_ms == 75.0
    assert metrics.error_rate == 0.01
    assert metrics.success_rate == 0.99

    for idx in range(1005):
        collector.record_request(duration_ms=float(idx), success=idx % 2 == 0)

    assert len(collector.request_times) == 1000
    assert collector.success_count > collector.error_count


def test_component_collector_returns_health(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = collectors.ComponentCollector()
    collector.component_start_times = {"DataProvider": datetime(2025, 1, 15, 12, 0)}

    components = collector.collect()

    assert set(components.keys()) == {
        "DataProvider",
        "Strategies",
        "RiskManager",
        "Executor",
        "Database",
        "API",
    }
    assert all(component.last_check.date() == datetime.now().date() for component in components.values())


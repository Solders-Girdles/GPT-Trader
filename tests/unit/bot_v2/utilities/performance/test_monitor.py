from contextlib import contextmanager

import pytest

from bot_v2.utilities.performance import monitor
from bot_v2.utilities.performance.metrics import PerformanceMetric


class StubCollector:
    def __init__(self) -> None:
        self.recorded: list[PerformanceMetric] = []
        self.summary = {"task": {"avg": 0.5, "max": 0.8}}
        self.recent_calls: list[tuple[str, int]] = []
        self.recent = [PerformanceMetric(name="task", value=0.1, unit="s", tags={})]

    def record(self, metric: PerformanceMetric) -> None:
        self.recorded.append(metric)

    def get_summary(self):
        return self.summary

    def get_recent_metrics(self, name: str, *, count: int = 10):
        self.recent_calls.append((name, count))
        return self.recent


class StubProfiler:
    pass


class StubResourceMonitor:
    def __init__(self, available: bool = True, snapshot=None, memory=None, cpu=None) -> None:
        self._available = available
        self._snapshot = snapshot or {}
        self._memory = memory or {}
        self._cpu = cpu or {}

    def is_available(self) -> bool:
        return self._available

    def get_system_info(self):
        return dict(self._snapshot)

    def get_memory_usage(self):
        return dict(self._memory)

    def get_cpu_usage(self):
        return dict(self._cpu)


@pytest.fixture
def stub_components(monkeypatch):
    collector = StubCollector()
    profiler = StubProfiler()
    resource = StubResourceMonitor()
    monkeypatch.setattr(monitor, "get_collector", lambda: collector)
    monkeypatch.setattr(monitor, "get_profiler", lambda: profiler)
    monkeypatch.setattr(monitor, "get_resource_monitor", lambda: resource)
    return collector, profiler, resource


def test_record_duration_creates_metric(stub_components):
    collector, profiler, resource = stub_components
    perf_monitor = monitor.PerformanceMonitor(
        collector=collector, profiler=profiler, resource_monitor=resource
    )

    perf_monitor.record_duration("db.query", 0.123, tags={"scope": "unit"})

    assert len(collector.recorded) == 1
    metric = collector.recorded[0]
    assert metric.name == "db.query"
    assert metric.value == pytest.approx(0.123)
    assert metric.tags == {"scope": "unit"}


def test_time_uses_measure_performance(monkeypatch, stub_components):
    collector, profiler, resource = stub_components
    captured = {}

    @contextmanager
    def fake_measure(name, tags, passed_collector):
        captured["name"] = name
        captured["tags"] = tags
        captured["collector"] = passed_collector
        yield "context"

    monkeypatch.setattr(monitor, "measure_performance", fake_measure)
    perf_monitor = monitor.PerformanceMonitor(
        collector=collector, profiler=profiler, resource_monitor=resource
    )

    with perf_monitor.time("task.run", tags={"env": "test"}) as ctx:
        assert ctx == "context"

    assert captured == {"name": "task.run", "tags": {"env": "test"}, "collector": collector}


def test_decorator_times_function(monkeypatch, stub_components):
    collector, profiler, resource = stub_components
    called = {"enter": 0, "exit": 0}

    @contextmanager
    def fake_measure(name, tags, passed_collector):
        called["enter"] += 1
        yield
        called["exit"] += 1

    monkeypatch.setattr(monitor, "measure_performance", fake_measure)
    perf_monitor = monitor.PerformanceMonitor(
        collector=collector, profiler=profiler, resource_monitor=resource
    )

    @perf_monitor.decorator("custom.op", tags={"region": "use1"})
    def sample(a, b):
        return a + b

    assert sample(2, 3) == 5
    assert called == {"enter": 1, "exit": 1}


def test_summary_and_recent_metrics_delegate(stub_components):
    collector, profiler, resource = stub_components
    perf_monitor = monitor.PerformanceMonitor(
        collector=collector, profiler=profiler, resource_monitor=resource
    )

    assert perf_monitor.summary() == collector.summary

    recent = perf_monitor.recent_metrics("task", count=5)
    assert recent == collector.recent
    assert collector.recent_calls == [("task", 5)]


def test_system_snapshot_unavailable(monkeypatch, stub_components):
    collector, profiler, _ = stub_components
    unavailable = StubResourceMonitor(available=False)
    perf_monitor = monitor.PerformanceMonitor(
        collector=collector, profiler=profiler, resource_monitor=unavailable
    )

    assert perf_monitor.system_snapshot() == {}


def test_system_snapshot_combines_data(monkeypatch, stub_components):
    collector, profiler, _ = stub_components
    resource = StubResourceMonitor(
        snapshot={"cpu_count": 4},
        memory={"rss_mb": 512},
        cpu={"cpu_percent": 42.0},
    )
    perf_monitor = monitor.PerformanceMonitor(
        collector=collector, profiler=profiler, resource_monitor=resource
    )

    snapshot = perf_monitor.system_snapshot()
    assert snapshot == {"cpu_count": 4, "rss_mb": 512, "cpu_percent": 42.0}

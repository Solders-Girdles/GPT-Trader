from __future__ import annotations

from decimal import Decimal

from bot_v2.utilities.performance import PerformanceMonitor, PerformanceMetric


def test_record_duration_and_summary() -> None:
    monitor = PerformanceMonitor()
    monitor.record_duration("op", 0.5)
    summary = monitor.summary()
    assert "op" in summary
    stats = summary["op"]
    assert stats["count"] == 1
    assert stats["total"] == 0.5


def test_time_context_records_metric() -> None:
    monitor = PerformanceMonitor()
    with monitor.time("timed_op"):
        pass
    metrics = monitor.recent_metrics("timed_op")
    assert metrics and isinstance(metrics[0], PerformanceMetric)


def test_profile_decorator_uses_profiler(monkeypatch) -> None:
    monitor = PerformanceMonitor()
    monitor.profiler.sample_rate = 1.0  # always sample
    calls: list[float] = []

    def record_call(func_name: str, duration: float) -> None:  # pragma: no cover - patched
        calls.append(duration)

    monkeypatch.setattr(monitor.profiler, "record_call", record_call)

    @monitor.profile(sample_rate=1.0)
    def work() -> int:
        return 42

    assert work() == 42
    assert calls


def test_system_snapshot_handles_unavailable(monkeypatch) -> None:
    monitor = PerformanceMonitor()
    monkeypatch.setattr(monitor.resource_monitor, "is_available", lambda: False)
    assert monitor.system_snapshot() == {}


def test_system_snapshot_merges_metrics(monkeypatch) -> None:
    monitor = PerformanceMonitor()
    monkeypatch.setattr(monitor.resource_monitor, "is_available", lambda: True)
    monkeypatch.setattr(
        monitor.resource_monitor,
        "get_system_info",
        lambda: {"cpu_count": 2},
    )
    monkeypatch.setattr(
        monitor.resource_monitor,
        "get_memory_usage",
        lambda: {"rss_mb": 128.0},
    )
    monkeypatch.setattr(
        monitor.resource_monitor,
        "get_cpu_usage",
        lambda: {"cpu_percent": 12.5},
    )

    snapshot = monitor.system_snapshot()
    assert snapshot == {"cpu_count": 2, "rss_mb": 128.0, "cpu_percent": 12.5}

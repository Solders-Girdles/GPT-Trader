import pytest

from gpt_trader.utilities.performance import health


class StubCollector:
    def __init__(self, summary):
        self._summary = summary

    def get_summary(self):
        return self._summary


class StubResourceMonitor:
    def __init__(self, *, available=True, memory=None, cpu=None):
        self._available = available
        self._memory = memory or {}
        self._cpu = cpu or {}

    def is_available(self):
        return self._available

    def get_memory_usage(self):
        return self._memory

    def get_cpu_usage(self):
        return self._cpu


@pytest.fixture(autouse=True)
def reset_singletons():
    # ensure module level singleton does not leak between tests
    health.get_resource_monitor.__globals__["_resource_monitor"] = None
    yield
    health.get_resource_monitor.__globals__["_resource_monitor"] = None


def test_health_check_with_no_metrics(monkeypatch):
    monkeypatch.setattr(health, "get_collector", lambda: StubCollector({}))
    monitor = StubResourceMonitor(available=False)
    monkeypatch.setattr(health, "get_resource_monitor", lambda: monitor)

    result = health.get_performance_health_check()

    assert result["status"] == "healthy"
    assert result["issues"] == []
    assert result["metrics"] == {
        "total_metrics": 0,
        "memory_usage_mb": 0,
        "cpu_usage_percent": 0,
    }


def test_health_check_flags_slow_operations(monkeypatch):
    summary = {
        "db_query": {"avg": 1.5, "max": 2.0},
        "report_generation": {"avg": 0.8, "max": 6.2},
    }
    monkeypatch.setattr(health, "get_collector", lambda: StubCollector(summary))
    monitor = StubResourceMonitor(
        memory={"rss_mb": 512, "percent": 72.5},
        cpu={"cpu_percent": 42.0},
    )
    monkeypatch.setattr(health, "get_resource_monitor", lambda: monitor)

    result = health.get_performance_health_check()

    assert result["status"] == "unhealthy"
    assert "Slow operation: db_query averaging 1.500s" in result["issues"]
    assert "Very slow operation: report_generation peaked at 6.200s" in result["issues"]
    assert result["metrics"]["total_metrics"] == 2
    assert result["metrics"]["memory_usage_mb"] == pytest.approx(512)
    assert result["metrics"]["cpu_usage_percent"] == pytest.approx(42.0)


def test_health_check_flags_high_resource_usage(monkeypatch):
    summary = {
        "task": {"avg": 0.2, "max": 0.4},
    }
    monkeypatch.setattr(health, "get_collector", lambda: StubCollector(summary))
    monitor = StubResourceMonitor(
        memory={"rss_mb": 1024, "percent": 85.0},
        cpu={"cpu_percent": 91.0},
    )
    monkeypatch.setattr(health, "get_resource_monitor", lambda: monitor)

    result = health.get_performance_health_check()

    assert result["status"] == "degraded"
    assert "High memory usage: 85.0%" in result["issues"]
    assert "High CPU usage: 91.0%" in result["issues"]
    assert result["metrics"]["memory_usage_mb"] == pytest.approx(1024)
    assert result["metrics"]["cpu_usage_percent"] == pytest.approx(91.0)

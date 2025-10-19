import builtins
import sys
from types import SimpleNamespace

import pytest
from src.bot_v2.utilities.performance import resource


class StubMemory:
    def __init__(self, *, used, total, available, percent):
        self.used = used
        self.total = total
        self.available = available
        self.percent = percent


class StubPsutil:
    def __init__(self, *, memory: StubMemory, cpu_percent: float, cpu_count: int):
        self._memory = memory
        self._cpu_percent = cpu_percent
        self._cpu_count = cpu_count

    def virtual_memory(self):
        return self._memory

    def cpu_percent(self):
        return self._cpu_percent

    def cpu_count(self):
        return self._cpu_count


@pytest.fixture(autouse=True)
def reset_singleton():
    resource._resource_monitor = None
    sys.modules.pop("bot_v2.utilities.performance_monitoring", None)
    yield
    resource._resource_monitor = None
    sys.modules.pop("bot_v2.utilities.performance_monitoring", None)


def test_resource_monitor_reports_usage(monkeypatch):
    memory = StubMemory(
        used=512 * 1024 * 1024, total=2048 * 1024 * 1024, available=1024 * 1024 * 1024, percent=37.5
    )
    stub = StubPsutil(memory=memory, cpu_percent=24.0, cpu_count=8)

    monitor = resource.ResourceMonitor()
    monitor._psutil = stub  # inject stub

    mem_usage = monitor.get_memory_usage()
    cpu_usage = monitor.get_cpu_usage()
    system = monitor.get_system_info()

    assert mem_usage == {
        "rss_mb": pytest.approx(512.0),
        "vms_mb": pytest.approx(2048.0),
        "percent": pytest.approx(37.5),
    }
    assert cpu_usage == {"cpu_percent": pytest.approx(24.0), "cpu_count": 8}
    assert system == {
        "cpu_count": 8,
        "memory_total_gb": pytest.approx(2.0),
        "memory_available_gb": pytest.approx(1.0),
        "memory_percent": pytest.approx(37.5),
    }


def test_resource_monitor_handles_unavailable(monkeypatch):
    monitor = resource.ResourceMonitor()
    monitor._psutil = None

    assert monitor.is_available() is False
    assert monitor.get_memory_usage() == {}
    assert monitor.get_cpu_usage() == {}
    assert monitor.get_system_info() == {}


def test_get_resource_monitor_memoizes_instance(monkeypatch):
    monitor1 = resource.get_resource_monitor()
    monitor2 = resource.get_resource_monitor()
    assert monitor1 is monitor2


def test_try_import_psutil_uses_legacy(monkeypatch):
    legacy_psutil = object()
    sys.modules["bot_v2.utilities.performance_monitoring"] = SimpleNamespace(psutil=legacy_psutil)

    monkeypatch.setattr(resource, "psutil", None)
    monitor = resource.ResourceMonitor()
    assert monitor._psutil is legacy_psutil


def test_try_import_psutil_handles_import_error(monkeypatch):
    monkeypatch.setattr(resource, "psutil", None)
    sys.modules["bot_v2.utilities.performance_monitoring"] = SimpleNamespace(psutil=None)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    monitor = resource.ResourceMonitor()
    assert monitor._psutil is None

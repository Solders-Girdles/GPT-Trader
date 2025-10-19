from pathlib import Path

from src.bot_v2.utilities.performance import reporter


class StubCollector:
    def __init__(self, summary):
        self._summary = summary

    def get_summary(self):
        return self._summary


class StubResourceMonitor:
    def __init__(self, available=True, memory=None, cpu=None):
        self._available = available
        self._memory = memory or {}
        self._cpu = cpu or {}

    def is_available(self) -> bool:
        return self._available

    def get_memory_usage(self):
        return self._memory

    def get_cpu_usage(self):
        return self._cpu


class StubProfiler:
    def __init__(self, data):
        self._data = data

    def get_profile_data(self):
        return self._data


def test_generate_report_with_metrics(monkeypatch):
    collector = StubCollector({"task": {"avg": 0.2, "max": 0.3}})
    monitor = StubResourceMonitor(
        memory={"rss_mb": 512.0, "percent": 75.0}, cpu={"cpu_percent": 41.0}
    )
    profiler = StubProfiler(
        {
            "module.func": {
                "call_count": 4,
                "avg_time": 0.012,
                "total_time": 0.048,
            }
        }
    )
    perf_reporter = reporter.PerformanceReporter(
        collector=collector, resource_monitor=monitor, profiler=profiler
    )

    report = perf_reporter.generate_report()

    assert "Performance Report" in report
    assert "task: {'avg': 0.2, 'max': 0.3}" in report
    assert "Memory: 512.0MB RSS, 75.0%" in report
    assert "CPU: 41.0%" in report
    assert "module.func: 4 calls, 0.012s avg, 0.048s total" in report


def test_generate_report_no_data(monkeypatch):
    collector = StubCollector({})
    monitor = StubResourceMonitor(available=False)
    profiler = StubProfiler({})
    perf_reporter = reporter.PerformanceReporter(
        collector=collector, resource_monitor=monitor, profiler=profiler
    )

    report = perf_reporter.generate_report()

    assert "No metrics recorded" in report
    assert "Resource monitoring not available" in report
    assert "No profiling data available" in report


def test_log_report(monkeypatch):
    collector = StubCollector({})
    monitor = StubResourceMonitor(available=False)
    profiler = StubProfiler({})
    perf_reporter = reporter.PerformanceReporter(
        collector=collector, resource_monitor=monitor, profiler=profiler
    )
    logs = []

    class StubLogger:
        def log(self, level, message, **kwargs):
            logs.append((level, message, kwargs))

        def info(self, message, **kwargs):
            logs.append(("INFO", message, kwargs))

    monkeypatch.setattr(reporter, "logger", StubLogger())

    perf_reporter.log_report(level=10)

    assert logs and logs[0][0] == 10  # level passed through
    assert "performance_report" in logs[0][1]
    assert "report" in logs[0][2]


def test_save_report(tmp_path: Path, monkeypatch):
    collector = StubCollector({})
    monitor = StubResourceMonitor(available=False)
    profiler = StubProfiler({})
    perf_reporter = reporter.PerformanceReporter(
        collector=collector, resource_monitor=monitor, profiler=profiler
    )
    written = {}

    class StubLogger:
        def log(self, level, message, **kwargs):
            pass

        def info(self, message, **kwargs):
            written["info"] = (message, kwargs)

    monkeypatch.setattr(reporter, "logger", StubLogger())

    target = tmp_path / "report.txt"
    perf_reporter.save_report(str(target))

    assert target.exists()
    contents = target.read_text()
    assert "Performance Report" in contents
    assert written["info"][1]["path"] == str(target)

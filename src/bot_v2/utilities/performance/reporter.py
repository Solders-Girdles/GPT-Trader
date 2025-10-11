"""Utilities for generating performance reports."""

from __future__ import annotations

from bot_v2.utilities.logging_patterns import get_logger

from .metrics import PerformanceCollector, get_collector
from .profiler import PerformanceProfiler, get_profiler
from .resource import ResourceMonitor, get_resource_monitor

logger = get_logger("performance", component="monitoring")


class PerformanceReporter:
    """Generate performance reports."""

    def __init__(
        self,
        collector: PerformanceCollector | None = None,
        resource_monitor: ResourceMonitor | None = None,
        profiler: PerformanceProfiler | None = None,
    ) -> None:
        self.collector = collector or get_collector()
        self.resource_monitor = resource_monitor or get_resource_monitor()
        self.profiler = profiler or get_profiler()

    def generate_report(self) -> str:
        lines = ["Performance Report", "=" * 50]

        lines.append("\nPerformance Metrics:")
        lines.append("-" * 20)
        summary = self.collector.get_summary()
        if summary:
            for name, stats in sorted(summary.items()):
                lines.append(f"{name}: {stats}")
        else:
            lines.append("No metrics recorded")

        lines.append("\nResource Usage:")
        lines.append("-" * 15)
        if self.resource_monitor.is_available():
            memory = self.resource_monitor.get_memory_usage()
            cpu = self.resource_monitor.get_cpu_usage()
            if memory:
                lines.append(f"Memory: {memory['rss_mb']:.1f}MB RSS, {memory['percent']:.1f}%")
            if cpu:
                lines.append(f"CPU: {cpu['cpu_percent']:.1f}%")
        else:
            lines.append("Resource monitoring not available")

        lines.append("\nProfiling Data:")
        lines.append("-" * 15)
        profile_data = self.profiler.get_profile_data()
        if profile_data:
            for func_name, data in sorted(
                profile_data.items(), key=lambda item: item[1]["total_time"], reverse=True
            ):
                lines.append(
                    f"{func_name}: {data['call_count']} calls, "
                    f"{data['avg_time']:.3f}s avg, {data['total_time']:.3f}s total"
                )
        else:
            lines.append("No profiling data available")

        return "\n".join(lines)

    def log_report(self, level: int = 20) -> None:
        report = self.generate_report()
        logger.log(level, "performance_report", report=report)

    def save_report(self, filepath: str) -> None:
        report = self.generate_report()
        with open(filepath, "w") as handle:
            handle.write(report)
        logger.info("performance_report_saved", path=filepath)


__all__ = ["PerformanceReporter"]

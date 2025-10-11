"""Performance-oriented health checks."""

from __future__ import annotations

from typing import Any

from .metrics import get_collector
from .resource import get_resource_monitor


def get_performance_health_check() -> dict[str, Any]:
    health: dict[str, Any] = {
        "status": "healthy",
        "issues": [],
        "metrics": {},
    }

    collector = get_collector()
    summary = collector.get_summary()

    for name, stats in summary.items():
        avg = stats.get("avg", 0.0) if isinstance(stats, dict) else getattr(stats, "avg", 0.0)
        peak = stats.get("max", 0.0) if isinstance(stats, dict) else getattr(stats, "max", 0.0)

        if avg > 1.0:
            health["issues"].append(f"Slow operation: {name} averaging {avg:.3f}s")
            health["status"] = "degraded"
        if peak > 5.0:
            health["issues"].append(f"Very slow operation: {name} peaked at {peak:.3f}s")
            health["status"] = "unhealthy"

    resource_monitor = get_resource_monitor()
    memory = {}
    cpu = {}
    if resource_monitor.is_available():
        memory = resource_monitor.get_memory_usage()
        if memory.get("percent", 0) > 80:
            health["issues"].append(f"High memory usage: {memory['percent']:.1f}%")
            health["status"] = "degraded"

        cpu = resource_monitor.get_cpu_usage()
        if cpu.get("cpu_percent", 0) > 80:
            health["issues"].append(f"High CPU usage: {cpu['cpu_percent']:.1f}%")
            health["status"] = "degraded"

    health["metrics"] = {
        "total_metrics": len(summary),
        "memory_usage_mb": memory.get("rss_mb", 0),
        "cpu_usage_percent": cpu.get("cpu_percent", 0),
    }

    return health


__all__ = ["get_performance_health_check"]

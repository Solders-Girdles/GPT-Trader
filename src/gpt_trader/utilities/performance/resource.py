"""System resource monitoring helpers."""

from __future__ import annotations

from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    psutil = None  # type: ignore

logger = get_logger("performance", component="monitoring")


class ResourceMonitor:
    """Monitor system resource usage."""

    def __init__(self) -> None:
        self._psutil: Any = None
        self._try_import_psutil()

    def _try_import_psutil(self) -> None:
        import sys

        legacy = sys.modules.get("gpt_trader.utilities.performance_monitoring")
        maybe = getattr(legacy, "psutil", None) if legacy is not None else None
        if maybe is not None:
            self._psutil = maybe
            return
        if psutil is not None:
            self._psutil = psutil
            return
        try:
            import psutil as psutil_mod  # type: ignore

            self._psutil = psutil_mod
        except ImportError:
            logger.debug("psutil not available, resource monitoring disabled")
            self._psutil = None

    def is_available(self) -> bool:
        return self._psutil is not None

    def get_memory_usage(self) -> dict[str, float]:
        if not self.is_available():
            return {}

        ps_module = self._psutil
        if ps_module is None:
            return {}

        memory = ps_module.virtual_memory()
        return {
            "rss_mb": memory.used / 1024 / 1024,
            "vms_mb": memory.total / 1024 / 1024,
            "percent": float(memory.percent),
        }

    def get_cpu_usage(self) -> dict[str, float]:
        if not self.is_available():
            return {}

        ps_module = self._psutil
        if ps_module is None:
            return {}

        return {
            "cpu_percent": float(ps_module.cpu_percent()),
            "cpu_count": ps_module.cpu_count(),
        }

    def get_system_info(self) -> dict[str, Any]:
        if not self.is_available():
            return {}

        ps_module = self._psutil
        if ps_module is None:
            return {}

        memory_info = ps_module.virtual_memory()
        return {
            "cpu_count": ps_module.cpu_count(),
            "memory_total_gb": memory_info.total / 1024 / 1024 / 1024,
            "memory_available_gb": memory_info.available / 1024 / 1024 / 1024,
            "memory_percent": float(memory_info.percent),
        }


_resource_monitor: ResourceMonitor | None = None


def get_resource_monitor() -> ResourceMonitor:
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor


__all__ = ["ResourceMonitor", "get_resource_monitor", "psutil"]

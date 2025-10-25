"""Core implementation for production logger."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any

from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.utilities import utc_now_iso

from .levels import LEVEL_MAP, LogLevel


class BaseProductionLogger:
    """Base logger providing structured JSON logging utilities."""

    def __init__(
        self,
        service_name: str = "bot_v2",
        enable_console: bool = True,
        *,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self.service_name = service_name
        self._settings = settings or load_runtime_settings()

        env_console = self._settings.raw_env.get("PERPS_JSON_CONSOLE")
        if env_console is not None:
            enable_console = env_console.strip().lower() in ("1", "true", "yes", "on")
        self.enable_console = enable_console

        self.correlation_ids = threading.local()
        self._log_count = 0
        self._total_log_time = 0.0
        self._recent_logs: list[dict[str, Any]] = []
        self._max_recent_logs = 1000
        self._lock = threading.Lock()

        self._min_level = (self._settings.raw_env.get("PERPS_MIN_LOG_LEVEL") or "info").lower()
        if (self._settings.raw_env.get("PERPS_DEBUG") or "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            self._min_level = "debug"

        self._py_json_logger = logging.getLogger(f"{self.service_name}.json")
        if not self._py_json_logger.handlers:
            self._py_json_logger = logging.getLogger("bot_v2.json")

    # ------------------------------------------------------------------
    def set_correlation_id(self, correlation_id: str | None = None) -> None:
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]
        self.correlation_ids.value = correlation_id

    def get_correlation_id(self) -> str:
        if not hasattr(self.correlation_ids, "value"):
            self.set_correlation_id()
        return str(self.correlation_ids.value)

    # ------------------------------------------------------------------
    def _create_log_entry(
        self,
        level: LogLevel,
        event_type: str,
        message: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        entry = {
            "timestamp": utc_now_iso(),
            "level": level.value,
            "service": self.service_name,
            "correlation_id": self.get_correlation_id(),
            "event_type": event_type,
            "message": message,
        }
        if kwargs:
            entry.update(kwargs)

        log_time = time.perf_counter() - start_time
        self._log_count += 1
        self._total_log_time += log_time
        return entry

    def _emit_log(self, entry: dict[str, Any]) -> None:
        try:
            if LEVEL_MAP.get(entry.get("level", "info"), logging.INFO) < LEVEL_MAP.get(
                self._min_level, logging.INFO
            ):
                return
        except Exception:
            pass

        with self._lock:
            self._recent_logs.append(entry)
            if len(self._recent_logs) > self._max_recent_logs:
                self._recent_logs.pop(0)

        if self.enable_console:
            print(json.dumps(entry, separators=(",", ":")))
        try:
            py_level = LEVEL_MAP.get(entry.get("level", "info"), logging.INFO)
            self._py_json_logger.log(py_level, json.dumps(entry, separators=(",", ":")))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def log_event(
        self,
        level: LogLevel,
        event_type: str,
        message: str,
        component: str | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(level, event_type, message, **kwargs)
        if component:
            entry["component"] = component
        self._emit_log(entry)

    # ------------------------------------------------------------------
    def get_recent_logs(self, count: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return (
                self._recent_logs[-count:]
                if count < len(self._recent_logs)
                else self._recent_logs.copy()
            )

    def get_performance_stats(self) -> dict[str, float | int]:
        if self._log_count == 0:
            return {"avg_log_time_ms": 0.0, "total_logs": 0}

        avg_time_ms = (self._total_log_time / self._log_count) * 1000
        return {
            "avg_log_time_ms": avg_time_ms,
            "total_logs": self._log_count,
            "total_log_time_ms": self._total_log_time * 1000,
        }


__all__ = ["BaseProductionLogger"]

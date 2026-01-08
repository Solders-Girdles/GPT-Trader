"""Metrics publishing helpers for SystemMonitor."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from gpt_trader.config.path_registry import RUNTIME_DATA_DIR
from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.telemetry import emit_metric

logger = get_logger(__name__, component="system_monitor_metrics")


class MetricsPublisher:
    """Emit cycle metrics to the event store and on-disk snapshots."""

    def __init__(
        self,
        *,
        event_store: Any,
        bot_id: str,
        profile: str,
        base_dir: Path = RUNTIME_DATA_DIR,
    ) -> None:
        self._event_store = event_store
        self._bot_id = bot_id
        self._profile = profile
        self._base_dir = base_dir

    def publish(self, metrics: dict[str, Any]) -> None:
        """Write metrics to the event store, JSON snapshot, and logger."""

        emit_metric(
            self._event_store,
            self._bot_id,
            {"event_type": "cycle_metrics", **metrics},
            logger=logger,
        )

        self._write_snapshot(metrics)
        self._log_update(metrics)

    def _target_dirs(self) -> list[Path]:
        return [self._base_dir / self._profile]

    # ------------------------------------------------------------------
    def _write_snapshot(self, metrics: dict[str, Any]) -> None:
        try:
            for target_dir in self._target_dirs():
                metrics_path = target_dir / "metrics.json"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("w") as fh:
                    json.dump(metrics, fh, indent=2)
        except Exception as exc:
            logger.debug(
                "Failed to write metrics snapshot",
                operation="system_monitor_metrics",
                stage="write_snapshot",
                error=str(exc),
                exc_info=True,
            )

    def _log_update(self, metrics: dict[str, Any]) -> None:
        try:
            # Local imports to avoid circular dependency with __init__.py
            from gpt_trader.monitoring.system import LogLevel
            from gpt_trader.monitoring.system import get_logger as _get_plog

            summary_fields = {
                key: value
                for key, value in metrics.items()
                if key not in {"positions", "decisions", "event_type"}
            }
            _get_plog().log_event(
                level=LogLevel.INFO,
                event_type="metrics_update",
                message="Cycle metrics updated",
                component="CoinbaseTrader",
                **summary_fields,
            )
        except Exception as exc:
            logger.debug(
                "Failed to emit metrics update event",
                operation="system_monitor_metrics",
                stage="log_update",
                error=str(exc),
                exc_info=True,
            )

    def write_health_status(self, ok: bool, message: str = "", error: str = "") -> None:
        """Write health status snapshot to disk."""

        status = {
            "ok": ok,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "error": error,
        }
        for target_dir in self._target_dirs():
            status_path = target_dir / "health.json"
            status_path.parent.mkdir(parents=True, exist_ok=True)
            with status_path.open("w") as fh:
                json.dump(status, fh, indent=2)


__all__ = ["MetricsPublisher"]

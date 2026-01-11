"""Configuration guardian coordinator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import ConfigurationMonitor
from .detector import DriftDetector
from .logging_utils import logger  # naming: allow
from .models import BaselineSnapshot, DriftEvent


class ConfigurationGuardian:
    """Coordinate configuration monitors and drift detection."""

    def __init__(
        self,
        baseline_snapshot: BaselineSnapshot,
        *,
        monitors: Sequence[ConfigurationMonitor] | None = None,
        detector: DriftDetector | None = None,
    ) -> None:
        self.baseline = baseline_snapshot
        self._monitors = list(monitors or [])
        self._detector = detector or DriftDetector(baseline_snapshot)

    def check(self) -> list[DriftEvent]:
        """Run monitors and record any drift events."""
        events: list[DriftEvent] = []
        for monitor in self._monitors:
            try:
                events.extend(monitor.check_changes())
            except Exception as exc:
                logger.warning(
                    "Configuration monitor failed",
                    operation="config_guardian",
                    stage="check",
                    monitor=monitor.monitor_name,
                    error=str(exc),
                    exc_info=True,
                )

        if events:
            self._detector.record_drift_events(events)
        return events

    def reset_baseline(
        self, baseline_snapshot: BaselineSnapshot, *, user_id: str | None = None
    ) -> None:
        """Update baseline snapshot and refresh monitors."""
        self.baseline = baseline_snapshot
        self._detector.update_baseline(baseline_snapshot)
        for monitor in self._monitors:
            if hasattr(monitor, "update_baseline"):
                monitor.update_baseline(baseline_snapshot)

        if user_id:
            logger.info(
                "Configuration baseline reset",
                operation="config_guardian",
                stage="reset_baseline",
                user_id=user_id,
            )

    def get_state(self) -> dict[str, Any]:
        """Return current guardian state for status reporting."""
        return {
            "baseline_timestamp": self.baseline.timestamp.isoformat(),
            "monitor_count": len(self._monitors),
            "drift_summary": self._detector.get_drift_summary(),
        }


__all__ = ["ConfigurationGuardian"]

"""Drift detector for configuration guardian."""

from __future__ import annotations

from typing import Any

from .base import ConfigurationMonitor
from .logging_utils import logger
from .models import BaselineSnapshot, DriftEvent


class DriftDetector(ConfigurationMonitor):
    """Detects configuration drift from baseline snapshot."""

    def __init__(self, baseline_snapshot: BaselineSnapshot):
        self.baseline = baseline_snapshot
        self.drift_history: list[DriftEvent] = []

    def update_baseline(self, baseline_snapshot: BaselineSnapshot) -> None:
        """Refresh baseline snapshot used for drift comparisons."""
        self.baseline = baseline_snapshot

    def check_changes(self) -> list[DriftEvent]:
        """Compare current state against baseline."""
        return self.drift_history

    def record_drift_events(self, events: list[DriftEvent]) -> None:
        """Record drift events for audit trail."""
        self.drift_history.extend(events)
        if events:
            logger.info(
                "Drift events recorded",
                operation="config_guardian",
                stage="record_events",
                event_count=len(events),
            )

    def get_drift_summary(self) -> dict[str, Any]:
        """Get summary of drift activity."""
        if not self.drift_history:
            return {
                "total_events": 0,
                "critical": 0,
                "high": 0,
                "last_event": None,
            }

        critical = sum(1 for event in self.drift_history if event.severity == "critical")
        high = sum(1 for event in self.drift_history if event.severity == "high")
        last_event = self.drift_history[-1]

        return {
            "total_events": len(self.drift_history),
            "critical": critical,
            "high": high,
            "last_event": {
                "timestamp": last_event.timestamp.isoformat(),
                "component": last_event.component,
                "drift_type": last_event.drift_type,
                "severity": last_event.severity,
            },
        }

    def get_current_state(self) -> dict[str, Any]:
        """Expose drift history state for monitoring."""
        return self.get_drift_summary()

    @property
    def monitor_name(self) -> str:
        return "drift_detector"


__all__ = ["DriftDetector"]

"""Drift detector for configuration guardian."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .base import ConfigurationMonitor
from .logging_utils import logger  # naming: allow
from .models import BaselineSnapshot, ConfigurationIssue, DriftEvent


class ConfigurationGuardianDetector:
    """Runs configuration validators and aggregates issues."""

    def __init__(
        self,
        validators: Sequence[Callable[[], list[ConfigurationIssue]] | Any] | None = None,
    ) -> None:
        self._validators: list[Callable[[], list[ConfigurationIssue]] | Any] = list(
            validators or []
        )

    def detect(self) -> list[ConfigurationIssue]:
        """Run validators, returning any issues discovered."""
        issues: list[ConfigurationIssue] = []

        for validator in self._validators:
            try:
                if hasattr(validator, "validate"):
                    result = validator.validate()
                else:
                    result = validator()
            except Exception as exc:
                logger.warning(
                    "Configuration validator failed",
                    operation="config_guardian",
                    stage="detect",
                    validator=self._validator_name(validator),
                    error=str(exc),
                    exc_info=True,
                )
                continue

            if result:
                issues.extend(result)

        return issues

    @staticmethod
    def build_response(issues: list[ConfigurationIssue]) -> dict[str, Any]:
        """Build response payload for detected issues."""
        return {"issues": [issue.to_dict() for issue in issues]}

    @staticmethod
    def _validator_name(validator: Any) -> str:
        name = getattr(validator, "name", None)
        if name:
            return str(name)
        return str(getattr(validator, "__name__", validator.__class__.__name__))


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


__all__ = ["ConfigurationGuardianDetector", "DriftDetector"]

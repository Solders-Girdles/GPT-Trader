"""Data models for configuration guardian monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from gpt_trader.utilities.datetime_helpers import utc_now


@dataclass
class BaselineSnapshot:
    """Baseline configuration snapshot for drift comparisons."""

    timestamp: datetime = field(default_factory=utc_now)
    profile: str | None = None
    active_symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftEvent:
    """Configuration drift event detected at runtime."""

    timestamp: datetime
    component: str
    drift_type: str
    severity: str
    details: dict[str, Any] = field(default_factory=dict)
    suggested_response: str | None = None
    applied_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the drift event for logging or transport."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "drift_type": self.drift_type,
            "severity": self.severity,
            "details": self.details,
            "suggested_response": self.suggested_response,
            "applied_response": self.applied_response,
        }


@dataclass
class ConfigurationIssue:
    """Issue surfaced by configuration validation checks."""

    category: str
    severity: str
    recommendation: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the issue for response payloads."""
        return {
            "category": self.category,
            "severity": self.severity,
            "recommendation": self.recommendation,
            "message": self.message,
            "details": self.details,
        }


__all__ = ["BaselineSnapshot", "ConfigurationIssue", "DriftEvent"]

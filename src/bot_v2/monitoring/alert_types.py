from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AlertSeverity(Enum):
    """Normalized alert severity levels shared across monitoring modules."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def numeric_level(self) -> int:
        """Approximate logging-style numeric level for comparisons."""
        mapping = {
            AlertSeverity.DEBUG: 10,
            AlertSeverity.INFO: 20,
            AlertSeverity.WARNING: 30,
            AlertSeverity.ERROR: 40,
            AlertSeverity.CRITICAL: 50,
        }
        return mapping[self]

    @classmethod
    def coerce(cls, value: AlertSeverity | str | int) -> AlertSeverity:
        """Coerce user-provided severity into an ``AlertSeverity`` enum."""
        if isinstance(value, AlertSeverity):
            return value
        if isinstance(value, int):
            numeric_map = {
                10: cls.DEBUG,
                20: cls.INFO,
                30: cls.WARNING,
                40: cls.ERROR,
                50: cls.CRITICAL,
            }
            # Round to nearest defined level when possible
            return numeric_map.get(value, cls.CRITICAL if value >= 50 else cls.DEBUG)
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown alert severity: {value!r}")


@dataclass(slots=True)
class Alert:
    """Canonical alert payload exchanged between producers and consumers."""

    severity: AlertSeverity
    title: str
    message: str
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source: str | None = None
    category: str | None = None
    component: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen_at: datetime | None = None
    occurrences: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    resolved_at: datetime | None = None
    acknowledged: bool = False
    created_at_override: InitVar[datetime | None] = None

    def __post_init__(self, created_at_override: datetime | None) -> None:
        if created_at_override is not None:
            self.created_at = created_at_override
        if self.last_seen_at is None:
            self.last_seen_at = self.created_at

    @property
    def id(self) -> str:
        """Backward compatible alias for :attr:`alert_id`."""
        return self.alert_id

    @property
    def timestamp(self) -> datetime:  # noqa: F811
        """Alias kept for callers that relied on ``timestamp`` field."""
        return self.created_at

    def touch(
        self,
        *,
        context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Increment the occurrence counter and update freshness markers."""
        self.occurrences += 1
        self.last_seen_at = datetime.utcnow()
        if context:
            self.context.update(context)
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the alert for structured logging/transport layers."""
        payload: dict[str, Any] = {
            "id": self.alert_id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "category": self.category,
            "component": self.component,
            "created_at": self.created_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "occurrences": self.occurrences,
            "metadata": self.metadata,
            "context": self.context,
            "details": self.details,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged": self.acknowledged,
        }
        return payload

    def mark_resolved(self) -> None:
        """Mark the alert as resolved at the current time."""
        self.resolved_at = datetime.utcnow()

    def acknowledge(self) -> None:
        """Mark the alert as acknowledged without resolving it."""
        self.acknowledged = True

    def is_active(self) -> bool:
        """Return ``True`` while the alert remains unresolved."""
        return self.resolved_at is None

    def age_minutes(self) -> float:
        """Age of the alert in minutes (resolved or now)."""
        end_time = self.resolved_at or datetime.utcnow()
        return (end_time - self.created_at).total_seconds() / 60


__all__ = ["AlertSeverity", "Alert"]

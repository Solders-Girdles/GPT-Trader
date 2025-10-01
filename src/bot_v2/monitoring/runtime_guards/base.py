"""
Base classes and types for Runtime Guard Alerting System.

Defines the foundational components used by all runtime guards:
- Alert data structures
- Guard configuration
- Guard status tracking
- Base guard evaluation logic
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, DecimalException, InvalidOperation
from enum import Enum
from typing import Any

# Import AlertSeverity from canonical alerts module
from bot_v2.monitoring.alerts import AlertSeverity

logger = logging.getLogger(__name__)


class GuardStatus(Enum):
    """Guard status states."""

    HEALTHY = "healthy"
    WARNING = "warning"
    BREACHED = "breached"
    DISABLED = "disabled"


@dataclass
class Alert:
    """Alert data structure."""

    timestamp: datetime
    guard_name: str
    severity: AlertSeverity
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "guard_name": self.guard_name,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class GuardConfig:
    """Configuration for a runtime guard."""

    name: str
    enabled: bool = True
    threshold: float = 0.0
    window_seconds: int = 60
    severity: AlertSeverity = AlertSeverity.WARNING
    auto_shutdown: bool = False
    cooldown_seconds: int = 300  # Prevent alert spam


class RuntimeGuard:
    """Base class for runtime guards."""

    def __init__(self, config: GuardConfig) -> None:
        self.config = config
        self.status: GuardStatus = GuardStatus.HEALTHY if config.enabled else GuardStatus.DISABLED
        self.last_check: datetime = datetime.now()
        self.last_alert: datetime | None = None
        self.breach_count: int = 0
        self.alerts: list[Alert] = []

    def check(self, context: dict[str, Any]) -> Alert | None:
        """
        Check guard condition and return alert if breached.

        Args:
            context: Current runtime context

        Returns:
            Alert if condition breached, None otherwise
        """
        if not self.config.enabled:
            return None

        # Perform guard-specific check
        is_breached, message = self._evaluate(context)

        # Check cooldown only for new alerts, not for status updates
        in_cooldown = False
        if self.last_alert:
            elapsed = (datetime.now() - self.last_alert).total_seconds()
            in_cooldown = elapsed < self.config.cooldown_seconds

        if is_breached:
            self.status = GuardStatus.BREACHED
            self.breach_count += 1

            # Only create alert if not in cooldown
            if not in_cooldown:
                alert = Alert(
                    timestamp=datetime.now(),
                    guard_name=self.config.name,
                    severity=self.config.severity,
                    message=message,
                    context=dict(context),
                )
                self.alerts.append(alert)
                self.last_alert = datetime.now()
                return alert
            return None

        # Check if we should downgrade from breached to warning
        if self.status == GuardStatus.BREACHED:
            self.status = GuardStatus.WARNING

        self.last_check = datetime.now()
        return None

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Generic evaluation hook used when no specialised guard exists.

        The default implementation inspects the context for a numeric value,
        compares it against a threshold, and emits a human-readable message
        when the configured comparison is breached. Subclasses can override
        this method for bespoke guard behaviour while still calling into the
        base implementation for convenience.
        """

        def _to_decimal(raw: Any) -> Decimal | None:
            try:
                return Decimal(str(raw))
            except (TypeError, ValueError, DecimalException, InvalidOperation):
                return None

        metric_key = context.get("metric_key")
        candidate_keys: list[str] = []
        if isinstance(metric_key, str):
            candidate_keys.append(metric_key)
        candidate_keys.extend(
            [
                "value",
                "metric",
                self.config.name,
                f"{self.config.name}_value",
            ]
        )

        raw_value = None
        for key in candidate_keys:
            if key in context:
                raw_value = context[key]
                break

        value = _to_decimal(raw_value)
        if value is None:
            return False, ""

        threshold_sources = (
            context.get("threshold_override"),
            context.get("threshold"),
            context.get("limit"),
            self.config.threshold,
        )
        threshold_raw = next((item for item in threshold_sources if item is not None), None)
        threshold = _to_decimal(threshold_raw)
        if threshold is None:
            return False, ""

        comparison = str(context.get("comparison", context.get("operator", "gt"))).lower()
        comparisons: dict[str, tuple[str, Callable[[Decimal, Decimal], bool], bool]] = {
            "gt": ("exceeded", lambda v, t: v > t, False),
            "+gt": ("exceeded", lambda v, t: v > t, False),
            ">": ("exceeded", lambda v, t: v > t, False),
            "ge": ("reached threshold", lambda v, t: v >= t, False),
            "gte": ("reached threshold", lambda v, t: v >= t, False),
            ">=": ("reached threshold", lambda v, t: v >= t, False),
            "lt": ("dropped below", lambda v, t: v < t, False),
            "lte": ("dropped to or below", lambda v, t: v <= t, False),
            "le": ("dropped to or below", lambda v, t: v <= t, False),
            "<": ("dropped below", lambda v, t: v < t, False),
            "<=": ("dropped to or below", lambda v, t: v <= t, False),
            "eq": ("matched", lambda v, t: v == t, False),
            "==": ("matched", lambda v, t: v == t, False),
            "ne": ("deviated from", lambda v, t: v != t, False),
            "!=": ("deviated from", lambda v, t: v != t, False),
            "abs_gt": (
                "exceeded absolute limit",
                lambda v, t: v.copy_abs() > t,
                True,
            ),
            "abs_gte": (
                "reached absolute limit",
                lambda v, t: v.copy_abs() >= t,
                True,
            ),
            "abs_ge": (
                "reached absolute limit",
                lambda v, t: v.copy_abs() >= t,
                True,
            ),
        }

        descriptor, comparator, use_absolute = comparisons.get(comparison, comparisons["gt"])

        evaluated_value = value.copy_abs() if use_absolute else value
        if comparator(value, threshold):
            label = context.get("label") or self.config.name.replace("_", " ")
            units = context.get("units")

            def _fmt(num: Decimal) -> str:
                text = format(num, "f")
                if "." in text:
                    text = text.rstrip("0").rstrip(".")
                return text

            suffix = f" {units}" if units else ""
            message_templates = {
                "matched": "{label} matched expected value {threshold}{suffix}",
                "deviated from": (
                    "{label} deviated from expected value {threshold}{suffix} "
                    "(current: {value}{suffix})"
                ),
                "exceeded absolute limit": (
                    "{label} exceeded absolute limit {threshold}{suffix} "
                    "(|current|: {value}{suffix})"
                ),
                "reached absolute limit": (
                    "{label} reached absolute limit {threshold}{suffix} "
                    "(|current|: {value}{suffix})"
                ),
            }
            template = message_templates.get(
                descriptor,
                "{label} {descriptor} {threshold}{suffix} (current: {value}{suffix})",
            )
            message = template.format(
                label=label,
                descriptor=descriptor,
                threshold=_fmt(threshold),
                value=_fmt(evaluated_value),
                suffix=suffix,
            )
            return True, message

        # Allow callers to express warning thresholds without triggering an alert
        warning_sources = (
            context.get("warning_threshold"),
            context.get("warning_limit"),
        )
        warning_raw = next((item for item in warning_sources if item is not None), None)
        if warning_raw is None and context.get("warning_ratio") is not None:
            ratio = _to_decimal(context.get("warning_ratio"))
            if ratio is not None:
                warning_raw = threshold * ratio

        warning_threshold = _to_decimal(warning_raw)
        if (
            warning_threshold is not None
            and self.status == GuardStatus.HEALTHY
            and comparator(value, warning_threshold)
        ):
            self.status = GuardStatus.WARNING

        return False, ""

    def reset(self) -> None:
        """Reset guard state."""
        self.status = GuardStatus.HEALTHY if self.config.enabled else GuardStatus.DISABLED
        self.breach_count = 0
        self.last_alert = None

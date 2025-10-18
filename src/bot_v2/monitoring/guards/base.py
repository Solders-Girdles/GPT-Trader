"""Base classes and primitives for runtime guards."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from bot_v2.monitoring.alert_types import AlertSeverity
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.validation import DecimalRule, RuleError

logger = get_logger(__name__, component="monitoring_guards_base")


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

    _DECIMAL_RULE = DecimalRule(allow_none=True)

    def __init__(self, config: GuardConfig) -> None:
        self.config = config
        self.status: GuardStatus = GuardStatus.HEALTHY if config.enabled else GuardStatus.DISABLED
        self.last_check: datetime = datetime.now()
        self.last_alert: datetime | None = None
        self.breach_count: int = 0
        self.alerts: list[Alert] = []

    def check(self, context: dict[str, Any]) -> Alert | None:
        """Check guard condition and return an alert if breached."""
        if not self.config.enabled:
            return None

        if self.last_alert:
            elapsed = (datetime.now() - self.last_alert).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                return None

        is_breached, message = self._evaluate(context)

        if is_breached:
            self.status = GuardStatus.BREACHED
            self.breach_count += 1
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

        if self.status == GuardStatus.BREACHED:
            self.status = GuardStatus.WARNING

        self.last_check = datetime.now()
        return None

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------
    def _coerce_decimal(self, raw: Any) -> Decimal | None:
        try:
            return self._DECIMAL_RULE(raw, "value")
        except RuleError:
            return None

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Default evaluation logic based on threshold comparisons."""
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

        value = self._coerce_decimal(raw_value)
        if value is None:
            return False, ""

        threshold_sources = (
            context.get("threshold_override"),
            context.get("threshold"),
            context.get("limit"),
            self.config.threshold,
        )
        threshold_raw = next((item for item in threshold_sources if item is not None), None)
        threshold = self._coerce_decimal(threshold_raw)
        if threshold is None:
            return False, ""

        comparison = str(context.get("comparison", context.get("operator", "gt"))).lower()
        comparisons: dict[str, tuple[str, Any, bool]] = {
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
            "abs_gt": ("exceeded absolute limit", lambda v, t: v.copy_abs() > t, True),
            "abs_gte": ("reached absolute limit", lambda v, t: v.copy_abs() >= t, True),
            "abs_ge": ("reached absolute limit", lambda v, t: v.copy_abs() >= t, True),
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

        warning_ratio = self._coerce_decimal(context.get("warning_ratio"))
        if (
            warning_ratio is not None
            and warning_ratio > Decimal("0")
            and warning_ratio < Decimal("1")
            and evaluated_value >= threshold * warning_ratio
            and self.status == GuardStatus.HEALTHY
        ):
            self.status = GuardStatus.WARNING

        return False, ""


__all__ = ["Alert", "GuardConfig", "GuardStatus", "RuntimeGuard"]

"""Base classes and primitives for runtime guards."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, cast

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.time_provider import SystemClock, TimeProvider
from gpt_trader.validation import DecimalRule, RuleError

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

    def __init__(self, config: GuardConfig, *, time_provider: TimeProvider | None = None) -> None:
        self.config = config
        self._time_provider = time_provider or SystemClock()
        self.status: GuardStatus = GuardStatus.HEALTHY if config.enabled else GuardStatus.DISABLED
        self.last_check: datetime = self._now()
        self.last_alert: datetime | None = None
        self._last_alerts_by_key: dict[str, datetime] = {}
        self._last_evaluation_evaluable: bool = True
        self.breach_count: int = 0
        self.alerts: list[Alert] = []

    def check(self, context: dict[str, Any]) -> Alert | None:
        """Check guard condition and return an alert if breached."""
        now = self._now()
        if not self.config.enabled:
            return None

        cooldown_key = self._resolve_cooldown_key(context)
        last_alert = self._last_alerts_by_key.get(cooldown_key)
        if last_alert:
            elapsed = (now - last_alert).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                return None

        self._last_evaluation_evaluable = True
        is_breached, message = self._evaluate(context)

        if not self._last_evaluation_evaluable:
            self.last_check = now
            return None

        if is_breached:
            self.status = GuardStatus.BREACHED
            self.breach_count += 1
            alert = Alert(
                timestamp=now,
                guard_name=self.config.name,
                severity=self.config.severity,
                message=message,
                context=dict(context),
            )
            self.alerts.append(alert)
            self.last_alert = now
            self._last_alerts_by_key[cooldown_key] = now
            self.last_check = now
            return alert

        if self.status == GuardStatus.BREACHED:
            self.status = GuardStatus.WARNING

        self.last_check = now
        return None

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------
    def _coerce_decimal(self, raw: Any) -> Decimal | None:
        try:
            return cast(Decimal, self._DECIMAL_RULE(raw, "value"))
        except RuleError:
            return None

    def _now(self) -> datetime:
        return self._time_provider.now_utc()

    def _resolve_cooldown_key(self, context: dict[str, Any]) -> str:
        cooldown_key = context.get("cooldown_key")
        if isinstance(cooldown_key, str) and cooldown_key:
            return cooldown_key

        metric_key = context.get("metric_key")
        if isinstance(metric_key, str) and metric_key:
            return metric_key

        return self.config.name

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
            self._last_evaluation_evaluable = False
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
            self._last_evaluation_evaluable = False
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

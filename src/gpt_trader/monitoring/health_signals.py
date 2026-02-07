"""
Health signal models for observability.

Provides structured health signals with status levels, thresholds,
and aggregation for alert readiness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HealthStatus(str, Enum):
    """Health status levels for signals."""

    OK = "OK"
    WARN = "WARN"
    CRIT = "CRIT"
    UNKNOWN = "UNKNOWN"


@dataclass
class HealthSignal:
    """Individual health signal with threshold-based status.

    Attributes:
        name: Signal identifier (e.g., "order_error_rate").
        status: Current status level (OK/WARN/CRIT).
        value: Current measured value.
        threshold_warn: Warning threshold (value >= threshold triggers WARN).
        threshold_crit: Critical threshold (value >= threshold triggers CRIT).
        unit: Unit of measurement (e.g., "percent", "ms", "count").
        details: Additional context for the signal.
    """

    name: str
    status: HealthStatus
    value: float
    threshold_warn: float
    threshold_crit: float
    unit: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(
        cls,
        name: str,
        value: float,
        threshold_warn: float,
        threshold_crit: float,
        unit: str = "",
        details: dict[str, Any] | None = None,
        *,
        higher_is_worse: bool = True,
    ) -> HealthSignal:
        """Create a signal with automatic status determination.

        Args:
            name: Signal identifier.
            value: Current measured value.
            threshold_warn: Warning threshold.
            threshold_crit: Critical threshold.
            unit: Unit of measurement.
            details: Additional context.
            higher_is_worse: If True, value >= threshold is bad.
                            If False, value <= threshold is bad (e.g., availability).

        Returns:
            HealthSignal with computed status.
        """
        if higher_is_worse:
            if value >= threshold_crit:
                status = HealthStatus.CRIT
            elif value >= threshold_warn:
                status = HealthStatus.WARN
            else:
                status = HealthStatus.OK
        else:
            # Lower is worse (e.g., availability percentage)
            if value <= threshold_crit:
                status = HealthStatus.CRIT
            elif value <= threshold_warn:
                status = HealthStatus.WARN
            else:
                status = HealthStatus.OK

        return cls(
            name=name,
            status=status,
            value=value,
            threshold_warn=threshold_warn,
            threshold_crit=threshold_crit,
            unit=unit,
            details=details or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "value": self.value,
            "threshold_warn": self.threshold_warn,
            "threshold_crit": self.threshold_crit,
            "unit": self.unit,
            "details": self.details,
        }


@dataclass
class HealthSummary:
    """Aggregated health summary from multiple signals.

    Attributes:
        status: Overall status (worst of all signals).
        signals: Individual health signals.
        message: Human-readable summary message.
    """

    status: HealthStatus
    signals: list[HealthSignal] = field(default_factory=list)
    message: str = ""

    @classmethod
    def from_signals(cls, signals: list[HealthSignal]) -> HealthSummary:
        """Create summary from a list of signals.

        Overall status is the worst status among all signals.
        """
        if not signals:
            return cls(
                status=HealthStatus.UNKNOWN,
                signals=[],
                message="No health signals available",
            )

        # Determine worst status
        status_priority = {
            HealthStatus.CRIT: 3,
            HealthStatus.WARN: 2,
            HealthStatus.OK: 1,
            HealthStatus.UNKNOWN: 0,
        }

        worst_status = max(signals, key=lambda s: status_priority.get(s.status, 0)).status

        # Build message
        crit_count = sum(1 for s in signals if s.status == HealthStatus.CRIT)
        warn_count = sum(1 for s in signals if s.status == HealthStatus.WARN)

        if worst_status == HealthStatus.CRIT:
            message = f"{crit_count} critical signal(s)"
            if warn_count:
                message += f", {warn_count} warning(s)"
        elif worst_status == HealthStatus.WARN:
            message = f"{warn_count} warning signal(s)"
        else:
            message = "All signals OK"

        return cls(
            status=worst_status,
            signals=signals,
            message=message,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "message": self.message,
            "signals": [s.to_dict() for s in self.signals],
        }


@dataclass
class HealthThresholds:
    """Configurable thresholds for health signals.

    All thresholds are optional with sensible defaults.
    """

    # Order submission error rate (failures / total in window)
    order_error_rate_warn: float = 0.05  # 5% warning
    order_error_rate_crit: float = 0.15  # 15% critical

    # Order retry rate (retries / total in window)
    order_retry_rate_warn: float = 0.10  # 10% warning
    order_retry_rate_crit: float = 0.25  # 25% critical

    # Broker API latency p95 (milliseconds)
    broker_latency_ms_warn: float = 1000.0  # 1 second warning
    broker_latency_ms_crit: float = 3000.0  # 3 seconds critical

    # WebSocket staleness (seconds since last message)
    ws_staleness_seconds_warn: float = 30.0  # 30 seconds warning
    ws_staleness_seconds_crit: float = 60.0  # 60 seconds critical

    # Market data feed staleness (seconds since last ticker update)
    market_data_staleness_seconds_warn: float = 10.0  # 10 seconds warning
    market_data_staleness_seconds_crit: float = 30.0  # 30 seconds critical

    # Guard trip frequency (trips in window)
    guard_trip_count_warn: int = 3  # 3 trips warning
    guard_trip_count_crit: int = 10  # 10 trips critical

    # Missing decision_id events (count)
    missing_decision_id_count_warn: int = 1  # 1 missing ID warning
    missing_decision_id_count_crit: int = 3  # 3 missing IDs critical


__all__ = [
    "HealthSignal",
    "HealthStatus",
    "HealthSummary",
    "HealthThresholds",
]

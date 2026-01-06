"""
Unified threshold constants for TUI performance and system widgets.

Provides consistent thresholds and helper functions for status indicators
across PerformanceDashboardWidget, SystemMonitorWidget, and RiskWidget.

Status levels:
- OK (green): Normal operation, within targets
- WARNING (yellow): Approaching limits, needs attention
- CRITICAL (red): Exceeds limits, requires action
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StatusLevel(Enum):
    """Status severity levels for metrics."""

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


# CSS class names for each status level
STATUS_CLASSES = {
    StatusLevel.OK: "status-ok",
    StatusLevel.WARNING: "status-warning",
    StatusLevel.CRITICAL: "status-critical",
}

# Status indicators (Unicode symbols)
STATUS_ICONS = {
    StatusLevel.OK: "✓",
    StatusLevel.WARNING: "⚠",
    StatusLevel.CRITICAL: "✗",
}

# Status colors for Rich markup
STATUS_COLORS = {
    StatusLevel.OK: "green",
    StatusLevel.WARNING: "yellow",
    StatusLevel.CRITICAL: "red",
}


@dataclass(frozen=True)
class PerformanceThresholds:
    """Unified thresholds for performance and system metrics.

    All thresholds define boundaries between OK/WARNING/CRITICAL states.
    Values are chosen for typical trading TUI workloads.
    """

    # === Latency (milliseconds) ===
    # Frame time / API response time
    latency_ok: float = 50.0  # Below = OK (responsive)
    latency_warn: float = 150.0  # Below = WARNING, above = CRITICAL

    # === CPU (percentage) ===
    cpu_ok: float = 50.0  # Below = OK (headroom available)
    cpu_warn: float = 80.0  # Below = WARNING, above = CRITICAL

    # === Memory (percentage of available) ===
    memory_ok: float = 60.0  # Below = OK
    memory_warn: float = 80.0  # Below = WARNING, above = CRITICAL

    # === FPS (frames per second) ===
    # Trading TUI is refresh-limited, not gaming
    fps_ok: float = 0.5  # Above = OK (2+ seconds between refreshes is fine)
    fps_warn: float = 0.2  # Above = WARNING, below = CRITICAL

    # === Rate Limit (percentage used) ===
    rate_ok: float = 50.0  # Below = OK
    rate_warn: float = 80.0  # Below = WARNING, above = CRITICAL

    # === Error Rate (percentage) ===
    error_ok: float = 1.0  # Below = OK
    error_warn: float = 5.0  # Below = WARNING, above = CRITICAL


@dataclass(frozen=True)
class RiskThresholds:
    """Unified thresholds for risk metrics.

    All thresholds define boundaries between OK/WARNING/CRITICAL states.
    Values represent percentage of limit used (0.0 - 1.0).
    """

    # === Daily Loss Ratio (current_loss / limit) ===
    # Ratio of current daily loss to daily loss limit
    loss_ratio_ok: float = 0.50  # Below = OK (< 50% of limit used)
    loss_ratio_warn: float = 0.75  # Below = WARNING, above = CRITICAL

    # === Leverage Utilization ===
    leverage_ok: float = 0.50  # Below = OK
    leverage_warn: float = 0.75  # Below = WARNING, above = CRITICAL

    # === Risk Score (aggregate from multiple factors) ===
    # Score-based thresholds for overall risk status
    risk_score_ok: int = 2  # Below = LOW risk
    risk_score_warn: int = 5  # Below = MEDIUM risk, above = HIGH risk


@dataclass(frozen=True)
class OrderThresholds:
    """Unified thresholds for order age metrics.

    All thresholds define boundaries between OK/WARNING/CRITICAL states.
    Values are in seconds since order creation.
    """

    # === Order Age (seconds) ===
    # Time since order was placed - used for UI coloring and alerts
    age_ok: float = 30.0  # Below = OK (normal)
    age_warn: float = 60.0  # Below = WARNING, at/above = CRITICAL (triggers alert)


# Default thresholds instances
DEFAULT_THRESHOLDS = PerformanceThresholds()
DEFAULT_RISK_THRESHOLDS = RiskThresholds()
DEFAULT_ORDER_THRESHOLDS = OrderThresholds()


def get_latency_status(
    latency_ms: float,
    thresholds: PerformanceThresholds = DEFAULT_THRESHOLDS,
) -> StatusLevel:
    """Get status level for latency value.

    Args:
        latency_ms: Latency in milliseconds.
        thresholds: Thresholds to use.

    Returns:
        StatusLevel for the latency value.
    """
    if latency_ms < thresholds.latency_ok:
        return StatusLevel.OK
    elif latency_ms < thresholds.latency_warn:
        return StatusLevel.WARNING
    return StatusLevel.CRITICAL


def get_cpu_status(
    cpu_percent: float,
    thresholds: PerformanceThresholds = DEFAULT_THRESHOLDS,
) -> StatusLevel:
    """Get status level for CPU usage.

    Args:
        cpu_percent: CPU usage as percentage (0-100).
        thresholds: Thresholds to use.

    Returns:
        StatusLevel for the CPU value.
    """
    if cpu_percent < thresholds.cpu_ok:
        return StatusLevel.OK
    elif cpu_percent < thresholds.cpu_warn:
        return StatusLevel.WARNING
    return StatusLevel.CRITICAL


def get_memory_status(
    memory_percent: float,
    thresholds: PerformanceThresholds = DEFAULT_THRESHOLDS,
) -> StatusLevel:
    """Get status level for memory usage.

    Args:
        memory_percent: Memory usage as percentage (0-100).
        thresholds: Thresholds to use.

    Returns:
        StatusLevel for the memory value.
    """
    if memory_percent < thresholds.memory_ok:
        return StatusLevel.OK
    elif memory_percent < thresholds.memory_warn:
        return StatusLevel.WARNING
    return StatusLevel.CRITICAL


def get_fps_status(
    fps: float,
    thresholds: PerformanceThresholds = DEFAULT_THRESHOLDS,
) -> StatusLevel:
    """Get status level for FPS.

    Args:
        fps: Frames per second.
        thresholds: Thresholds to use.

    Returns:
        StatusLevel for the FPS value.
    """
    if fps >= thresholds.fps_ok:
        return StatusLevel.OK
    elif fps >= thresholds.fps_warn:
        return StatusLevel.WARNING
    return StatusLevel.CRITICAL


def get_rate_limit_status(
    rate_percent: float,
    thresholds: PerformanceThresholds = DEFAULT_THRESHOLDS,
) -> StatusLevel:
    """Get status level for rate limit usage.

    Args:
        rate_percent: Rate limit usage as percentage (0-100).
        thresholds: Thresholds to use.

    Returns:
        StatusLevel for the rate limit value.
    """
    if rate_percent < thresholds.rate_ok:
        return StatusLevel.OK
    elif rate_percent < thresholds.rate_warn:
        return StatusLevel.WARNING
    return StatusLevel.CRITICAL


def get_error_rate_status(
    error_percent: float,
    thresholds: PerformanceThresholds = DEFAULT_THRESHOLDS,
) -> StatusLevel:
    """Get status level for error rate.

    Args:
        error_percent: Error rate as percentage (0-100).
        thresholds: Thresholds to use.

    Returns:
        StatusLevel for the error rate value.
    """
    if error_percent < thresholds.error_ok:
        return StatusLevel.OK
    elif error_percent < thresholds.error_warn:
        return StatusLevel.WARNING
    return StatusLevel.CRITICAL


def get_status_class(status: StatusLevel) -> str:
    """Get CSS class for a status level.

    Args:
        status: The status level.

    Returns:
        CSS class name (e.g., "status-ok", "status-warning", "status-critical").
    """
    return STATUS_CLASSES[status]


def get_status_icon(status: StatusLevel) -> str:
    """Get icon for a status level.

    Args:
        status: The status level.

    Returns:
        Unicode status icon.
    """
    return STATUS_ICONS[status]


def get_status_color(status: StatusLevel) -> str:
    """Get Rich color name for a status level.

    Args:
        status: The status level.

    Returns:
        Color name for Rich markup.
    """
    return STATUS_COLORS[status]


def format_status_label(status: StatusLevel, include_icon: bool = True) -> str:
    """Format a status label with optional icon.

    Args:
        status: The status level.
        include_icon: Whether to include the status icon.

    Returns:
        Formatted label like "✓ OK" or "OK".
    """
    name = status.value.upper()
    if include_icon:
        return f"{STATUS_ICONS[status]} {name}"
    return name


# === Risk-specific helper functions ===


def get_loss_ratio_status(
    current_loss_pct: float,
    limit_pct: float,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
) -> StatusLevel:
    """Get status level for daily loss ratio.

    IMPORTANT: Uses abs(current_loss_pct) to correctly handle negative losses.
    A loss of -5% against a 10% limit = 50% utilization, not -50%.

    Args:
        current_loss_pct: Current daily loss as percentage (can be negative for losses).
        limit_pct: Daily loss limit as percentage.
        thresholds: Thresholds to use.

    Returns:
        StatusLevel for the loss ratio.
    """
    if limit_pct <= 0:
        return StatusLevel.OK  # No limit configured

    # Use abs() to correctly handle negative loss values
    loss_ratio = abs(current_loss_pct) / limit_pct

    if loss_ratio < thresholds.loss_ratio_ok:
        return StatusLevel.OK
    elif loss_ratio < thresholds.loss_ratio_warn:
        return StatusLevel.WARNING
    return StatusLevel.CRITICAL


def get_risk_score_status(
    risk_score: int,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
) -> StatusLevel:
    """Get status level for aggregate risk score.

    Args:
        risk_score: Aggregate risk score from multiple factors.
        thresholds: Thresholds to use.

    Returns:
        StatusLevel for the risk score (maps to LOW/MEDIUM/HIGH).
    """
    if risk_score < thresholds.risk_score_ok:
        return StatusLevel.OK  # LOW risk
    elif risk_score < thresholds.risk_score_warn:
        return StatusLevel.WARNING  # MEDIUM risk
    return StatusLevel.CRITICAL  # HIGH risk


def get_risk_status_label(status: StatusLevel) -> str:
    """Get human-readable risk status label.

    Maps StatusLevel to risk-specific labels (LOW/MEDIUM/HIGH).

    Args:
        status: The status level.

    Returns:
        Risk status label.
    """
    return {
        StatusLevel.OK: "LOW",
        StatusLevel.WARNING: "MEDIUM",
        StatusLevel.CRITICAL: "HIGH",
    }[status]


# === Confidence-specific helper functions ===


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Unified thresholds for strategy confidence display.

    Confidence is a 0-1 value representing signal strength.
    Higher is better (opposite of risk thresholds).
    """

    # === Confidence Levels ===
    # Boundaries for LOW/MEDIUM/HIGH display
    confidence_low: float = 0.4  # Below = LOW confidence
    confidence_high: float = 0.7  # Above = HIGH confidence, between = MEDIUM


DEFAULT_CONFIDENCE_THRESHOLDS = ConfidenceThresholds()


def get_confidence_status(
    confidence: float,
    thresholds: ConfidenceThresholds = DEFAULT_CONFIDENCE_THRESHOLDS,
) -> StatusLevel:
    """Get status level for confidence value.

    Note: For confidence, higher is better, so mapping is inverted:
    - HIGH confidence (>= 0.7) = OK (green)
    - MEDIUM confidence (0.4-0.7) = WARNING (yellow)
    - LOW confidence (< 0.4) = CRITICAL (red)

    Args:
        confidence: Confidence value (0.0 to 1.0).
        thresholds: Thresholds to use.

    Returns:
        StatusLevel for the confidence value.
    """
    if confidence >= thresholds.confidence_high:
        return StatusLevel.OK  # HIGH confidence = good
    elif confidence >= thresholds.confidence_low:
        return StatusLevel.WARNING  # MEDIUM confidence
    return StatusLevel.CRITICAL  # LOW confidence = concerning


def get_confidence_label(status: StatusLevel) -> str:
    """Get human-readable confidence label.

    Maps StatusLevel to confidence-specific labels (HIGH/MED/LOW).

    Args:
        status: The status level.

    Returns:
        Confidence label.
    """
    return {
        StatusLevel.OK: "HIGH",
        StatusLevel.WARNING: "MED",
        StatusLevel.CRITICAL: "LOW",
    }[status]


def format_confidence_with_badge(confidence: float) -> tuple[str, str]:
    """Format confidence value with badge for display.

    Args:
        confidence: Confidence value (0.0 to 1.0).

    Returns:
        Tuple of (formatted string like "0.75 HIGH", css class).
    """
    status = get_confidence_status(confidence)
    label = get_confidence_label(status)
    css_class = get_status_class(status)
    return f"{confidence:.2f} {label}", css_class


# === Order-specific helper functions ===


def get_order_age_status(
    age_seconds: float,
    thresholds: OrderThresholds = DEFAULT_ORDER_THRESHOLDS,
) -> StatusLevel:
    """Get status level for order age.

    Args:
        age_seconds: Order age in seconds.
        thresholds: Thresholds to use.

    Returns:
        StatusLevel for the order age.
    """
    if age_seconds < thresholds.age_ok:
        return StatusLevel.OK
    elif age_seconds < thresholds.age_warn:
        return StatusLevel.WARNING
    return StatusLevel.CRITICAL


def get_order_status_level(status: str) -> StatusLevel:
    """Get status level for order status string.

    Maps order status to visual severity for at-a-glance readability:
    - OK (green): Active orders working normally (OPEN, PENDING, FILLED)
    - WARNING (yellow): Needs attention (PARTIAL, EXPIRED)
    - CRITICAL (red): Error states (REJECTED, FAILED)

    CANCELLED is mapped to OK since it's user-initiated, not an error.

    Args:
        status: Order status string (e.g., "OPEN", "REJECTED").

    Returns:
        StatusLevel for the order status.
    """
    status_upper = status.upper()

    # Error states - red
    if status_upper in {"REJECTED", "FAILED"}:
        return StatusLevel.CRITICAL

    # Attention needed - yellow
    if status_upper in {"PARTIAL", "EXPIRED"}:
        return StatusLevel.WARNING

    # Normal states - green (OPEN, PENDING, FILLED, CANCELLED)
    return StatusLevel.OK


# Legacy aliases for backward compatibility
# These map old class names to new StatusLevel-based classes
LEGACY_CLASS_MAP = {
    "good": "status-ok",
    "warning": "status-warning",
    "bad": "status-critical",
    "risk-status-low": "status-ok",
    "risk-status-medium": "status-warning",
    "risk-status-high": "status-critical",
}

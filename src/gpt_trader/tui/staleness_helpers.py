"""Shared staleness and data freshness helpers for TUI widgets.

Provides unified thresholds, copy, and severity mapping for consistent
data trust signals across all widgets.

Thresholds:
    - Fresh: <10s since last data fetch
    - Stale (warning): 10-30s since last data fetch
    - Critical (error): >30s OR connection_healthy is False
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

# Staleness thresholds in seconds
FRESH_THRESHOLD_SECONDS = 10
STALE_THRESHOLD_SECONDS = 30

# Execution health thresholds
EXEC_SUCCESS_RATE_WARNING = 95.0  # Below this shows warning
EXEC_SUCCESS_RATE_CRITICAL = 80.0  # Below this shows critical
EXEC_RETRY_RATE_WARNING = 0.5  # Above this shows warning

Severity = Literal["fresh", "stale", "critical"]


def get_data_age_seconds(state: TuiState) -> int | None:
    """Get the age of data in seconds since last fetch.

    Args:
        state: TuiState instance.

    Returns:
        Age in seconds, or None if no data has been fetched yet.
    """
    if not state.data_available or state.last_data_fetch <= 0:
        return None
    return int(time.time() - state.last_data_fetch)


def get_staleness_severity(state: TuiState) -> Severity:
    """Determine staleness severity based on data age and connection health.

    Args:
        state: TuiState instance.

    Returns:
        "fresh" if data is recent (<10s),
        "stale" if data is aging (10-30s),
        "critical" if data is old (>30s) or connection unhealthy.
    """
    # Connection unhealthy is always critical
    if not state.connection_healthy:
        return "critical"

    age = get_data_age_seconds(state)
    if age is None:
        # No data yet - not stale, just missing
        return "fresh"

    if age >= STALE_THRESHOLD_SECONDS:
        return "critical"
    if age >= FRESH_THRESHOLD_SECONDS:
        return "stale"
    return "fresh"


def format_freshness_label(age_seconds: int) -> str:
    """Format data age as a human-readable relative time string.

    Args:
        age_seconds: Age of data in seconds.

    Returns:
        Formatted string like "5s ago", "2m ago", "1h ago".
    """
    if age_seconds < 60:
        return f"{age_seconds}s ago"
    if age_seconds < 3600:
        return f"{age_seconds // 60}m ago"
    return f"{age_seconds // 3600}h ago"


def get_freshness_display(state: TuiState) -> tuple[str, str] | None:
    """Get freshness indicator text with Rich markup for color coding.

    Args:
        state: TuiState instance.

    Returns:
        Tuple of (formatted_text, css_class) or None if no data yet.
        css_class is one of: "fresh", "stale", "critical".
    """
    if state.data_fetching:
        return "[cyan]syncing[/cyan]", "fresh"

    age = get_data_age_seconds(state)
    if age is None:
        return None

    label = format_freshness_label(age)
    severity = get_staleness_severity(state)

    if severity == "critical":
        return f"[red]{label}[/red]", "critical"
    if severity == "stale":
        return f"[yellow]{label}[/yellow]", "stale"
    return f"[dim]{label}[/dim]", "fresh"


def get_execution_health_banner(state: TuiState) -> tuple[str, str] | None:
    """Get banner text for execution health issues.

    Checks execution telemetry for degraded states:
    - Circuit breaker open
    - Low success rate
    - High retry rate

    Args:
        state: TuiState instance.

    Returns:
        Tuple of (banner_text, severity) or None if execution is healthy.
    """
    try:
        # Check resilience data for circuit breaker
        if state.resilience_data.any_circuit_open:
            return "Circuit breaker OPEN — execution paused", "error"

        # Check execution metrics
        exec_data = state.execution_data
        if exec_data.submissions_total == 0:
            return None  # No executions yet

        # Critical: success rate below 80%
        if exec_data.success_rate < EXEC_SUCCESS_RATE_CRITICAL:
            return f"Execution degraded: {exec_data.success_rate:.0f}% success", "error"

        # Warning: success rate below 95%
        if exec_data.success_rate < EXEC_SUCCESS_RATE_WARNING:
            return f"Execution warning: {exec_data.success_rate:.0f}% success", "warning"

        # Warning: high retry rate
        if exec_data.retry_rate > EXEC_RETRY_RATE_WARNING:
            return f"High retry rate: {exec_data.retry_rate:.1f}x", "warning"

    except (AttributeError, TypeError):
        pass  # Execution data not available

    return None


def get_staleness_banner(state: TuiState) -> tuple[str, str] | None:
    """Get banner text and severity for staleness/degraded states.

    Checks conditions in priority order:
    1. Data fetching (reconnecting)
    2. Degraded mode (StatusReporter unavailable)
    3. Execution health (circuit breaker, success rate)
    4. Critical staleness (>30s or connection unhealthy)
    5. Stale data (10-30s)

    Args:
        state: TuiState instance.

    Returns:
        Tuple of (banner_text, severity) where severity is "info", "warning",
        or "error". Returns None if no banner should be shown.
    """
    # Priority 1: Currently reconnecting/fetching
    if state.data_fetching:
        return "Reconnecting...", "info"

    # Priority 2: Degraded mode
    if state.degraded_mode:
        reason = state.degraded_reason or "Status reporter unavailable"
        return f"Degraded: {reason}", "warning"

    # Priority 3: Execution health issues (circuit breaker, low success rate)
    exec_banner = get_execution_health_banner(state)
    if exec_banner:
        return exec_banner

    # Priority 4: Connection state banners (handled separately by widgets
    # for connection-specific copy like "Connecting...")

    # Priority 5: Staleness based on data age
    age = get_data_age_seconds(state)
    if age is None:
        return None

    severity = get_staleness_severity(state)

    if severity == "critical":
        # Critical: >30s or connection unhealthy
        if not state.connection_healthy:
            return f"Data stale ({age}s) — press R to reconnect", "error"
        return f"Data stale ({age}s) — press R to reconnect", "error"

    if severity == "stale":
        # Warning: 10-30s
        return f"Data stale ({age}s)", "warning"

    return None


def get_connection_banner(
    connection_status: str,
    bot_running: bool,
    degraded_mode: bool = False,
) -> tuple[str, str] | None:
    """Get banner text for connection-specific states.

    This handles connection states that are independent of data staleness,
    like CONNECTING, DISCONNECTED, etc.

    Args:
        connection_status: Connection status string from system_data.
        bot_running: Whether the bot is currently running.
        degraded_mode: Whether in degraded mode.

    Returns:
        Tuple of (banner_text, severity) or None if no connection banner needed.
    """
    # Don't show connection banners when bot is intentionally stopped
    if not bot_running and not degraded_mode:
        return None

    status = connection_status.upper() if connection_status else ""

    # Transitional states - no banner (handled by empty state or spinner)
    if status in ("", "UNKNOWN", "CONNECTING", "RECONNECTING", "SYNCING", "--"):
        return None

    # Healthy states - no banner
    if status in ("CONNECTED", "OK", "HEALTHY"):
        return None

    # Error states
    if status in ("DISCONNECTED", "ERROR", "FAILED"):
        return "Connection lost", "error"

    # Other non-healthy states
    return f"Broker: {status}", "warning"


# Standard empty state configurations
EMPTY_STATE_STOPPED = {
    "title": "Awaiting Start",
    "subtitle": "Start the bot to see live data",
    "icon": "◉",
    "actions": ["[S] Start Bot", "[C] Config"],
}

EMPTY_STATE_STOPPED_READ_ONLY = {
    "title": "Feed Paused",
    "subtitle": "Start the data feed to see live data",
    "icon": "◉",
    "actions": ["[S] Start Feed", "[C] Config"],
}

EMPTY_STATE_CONNECTION_FAILED = {
    "title": "Connection Failed",
    "subtitle": "Check credentials and network",
    "icon": "⚠",
    "actions": ["[R] Reconnect", "[C] Config"],
}

EMPTY_STATE_CONNECTING = {
    "title": "Connecting",
    "subtitle": "Establishing connection...",
    "icon": "○",
    "actions": [],
}


def get_empty_state_config(
    data_type: str,
    bot_running: bool,
    data_source_mode: str,
    connection_status: str,
) -> dict[str, str | list[str]]:
    """Get standardized empty state configuration based on current state.

    Args:
        data_type: Type of data (e.g., "Market", "Position", "Strategy").
        bot_running: Whether the bot is currently running.
        data_source_mode: Current mode ("demo", "paper", "live", "read_only").
        connection_status: Connection status string.

    Returns:
        Dict with keys: title, subtitle, icon, actions.
    """
    status = connection_status.upper() if connection_status else ""

    # Bot not running
    if not bot_running:
        if data_source_mode == "read_only":
            return EMPTY_STATE_STOPPED_READ_ONLY.copy()
        return EMPTY_STATE_STOPPED.copy()

    # Connection failure
    if status in ("DISCONNECTED", "ERROR", "FAILED"):
        return EMPTY_STATE_CONNECTION_FAILED.copy()

    # Connecting/transitional
    if status in ("CONNECTING", "RECONNECTING", "SYNCING", "UNKNOWN", "--", ""):
        return {
            "title": "Connecting",
            "subtitle": f"Waiting for {data_type.lower()} feed...",
            "icon": "○",
            "actions": [],
        }

    # Running but no data yet
    return {
        "title": f"No {data_type} Yet",
        "subtitle": f"Waiting for {data_type.lower()} data...",
        "icon": "◇",
        "actions": ["[R] Refresh"],
    }

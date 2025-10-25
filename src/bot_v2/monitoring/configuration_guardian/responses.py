"""Drift response constants."""


class DriftResponse:
    """Response strategies for configuration drift detection."""

    STICKY = "sticky"  # Keep old config, warn, continue
    REDUCE_ONLY = "reduce_only"  # Switch to reduce-only mode
    EMERGENCY_SHUTDOWN = "emergency_shutdown"  # Stop all trading immediately


__all__ = ["DriftResponse"]

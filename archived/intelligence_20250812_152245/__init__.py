from __future__ import annotations

"""
Phase 1 Intelligence utilities: safety rails, metrics, CI tools, and simulators.

These modules are intentionally dependency-light and can be integrated
incrementally across live and backtest flows without impacting existing tests.
"""

__all__ = [
    "safety_rails",
    "selection_metrics",
    "transition_metrics",
    "confidence_intervals",
    "order_simulator",
    "observability",
]

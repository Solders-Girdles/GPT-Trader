"""Order metrics tracking and reporting for execution engines.

This module provides centralized metrics tracking for order placement,
rejections, fills, and cancellations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.features.brokerages.core.interfaces import Order
    from bot_v2.monitoring.metrics_collector import MetricsCollector


@dataclass
class OrderMetrics:
    """Order execution metrics.

    Tracks counts of various order lifecycle events.
    """

    placed: int = 0
    filled: int = 0
    cancelled: int = 0
    rejected: int = 0
    post_only_rejected: int = 0


class OrderMetricsReporter:
    """Centralized order metrics tracking and reporting.

    Responsibilities:
    - Track order placements, fills, cancellations, rejections
    - Categorize rejection reasons
    - Provide metrics summary

    This class consolidates all metrics tracking logic that was previously
    scattered across the AdvancedExecutionEngine.
    """

    def __init__(self) -> None:
        """Initialize metrics reporter."""
        self.metrics = OrderMetrics()
        self.rejections_by_reason: dict[str, int] = {}

    def record_placement(self, order: Order) -> None:
        """Record successful order placement.

        Args:
            order: The placed order
        """
        self.metrics.placed += 1

    def record_rejection(self, reason: str, *, post_only: bool = False) -> None:
        """Record order rejection with reason.

        Args:
            reason: Rejection reason (e.g., "risk", "position_sizing", "spec_violation")
            post_only: Whether this was a post-only rejection
        """
        self.metrics.rejected += 1

        if post_only:
            self.metrics.post_only_rejected += 1

        if reason:
            self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1

    def record_fill(self, order: Order) -> None:
        """Record order fill.

        Args:
            order: The filled order
        """
        self.metrics.filled += 1

    def record_cancellation(self, order: Order) -> None:
        """Record order cancellation.

        Args:
            order: The cancelled order
        """
        self.metrics.cancelled += 1

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary.

        Returns:
            Dictionary containing all metrics and rejection reasons
        """
        return {
            "placed": self.metrics.placed,
            "filled": self.metrics.filled,
            "cancelled": self.metrics.cancelled,
            "rejected": self.metrics.rejected,
            "post_only_rejected": self.metrics.post_only_rejected,
            "rejections_by_reason": dict(self.rejections_by_reason),
        }

    def get_metrics_dict(self) -> dict[str, int]:
        """Get metrics as a simple dictionary (for backward compatibility).

        Returns:
            Dictionary with metric counts
        """
        return {
            "placed": self.metrics.placed,
            "filled": self.metrics.filled,
            "cancelled": self.metrics.cancelled,
            "rejected": self.metrics.rejected,
            "post_only_rejected": self.metrics.post_only_rejected,
        }

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.metrics = OrderMetrics()
        self.rejections_by_reason.clear()

    def export_to_collector(self, collector: MetricsCollector, prefix: str = "orders") -> None:
        """Export metrics to MetricsCollector for telemetry.

        Args:
            collector: MetricsCollector instance to export to
            prefix: Metric name prefix (default: "orders")

        Example:
            >>> reporter = OrderMetricsReporter()
            >>> from bot_v2.monitoring.metrics_collector import get_metrics_collector
            >>> collector = get_metrics_collector()
            >>> reporter.export_to_collector(collector)
            # Metrics exported as:
            # - orders.placed
            # - orders.filled
            # - orders.cancelled
            # - orders.rejected
            # - orders.post_only_rejected
            # - orders.rejection.{reason}
        """
        # Export aggregate counters
        collector.record_gauge(f"{prefix}.placed", float(self.metrics.placed))
        collector.record_gauge(f"{prefix}.filled", float(self.metrics.filled))
        collector.record_gauge(f"{prefix}.cancelled", float(self.metrics.cancelled))
        collector.record_gauge(f"{prefix}.rejected", float(self.metrics.rejected))
        collector.record_gauge(
            f"{prefix}.post_only_rejected", float(self.metrics.post_only_rejected)
        )

        # Export rejection reasons as individual metrics
        for reason, count in self.rejections_by_reason.items():
            # Sanitize reason for metric name (replace spaces/special chars with underscores)
            safe_reason = reason.replace(" ", "_").replace("-", "_")
            collector.record_gauge(f"{prefix}.rejection.{safe_reason}", float(count))

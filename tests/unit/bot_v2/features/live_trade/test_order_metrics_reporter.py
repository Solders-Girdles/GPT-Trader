"""Unit tests for OrderMetricsReporter."""

from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.core.interfaces import Order
from bot_v2.features.live_trade.order_metrics_reporter import (
    OrderMetrics,
    OrderMetricsReporter,
)


@pytest.fixture
def reporter():
    """Create metrics reporter instance."""
    return OrderMetricsReporter()


@pytest.fixture
def mock_order():
    """Create mock order."""
    order = Mock(spec=Order)
    order.id = "order-123"
    return order


class TestOrderMetrics:
    """Test OrderMetrics dataclass."""

    def test_default_values(self):
        """Test that metrics initialize to zero."""
        metrics = OrderMetrics()
        assert metrics.placed == 0
        assert metrics.filled == 0
        assert metrics.cancelled == 0
        assert metrics.rejected == 0
        assert metrics.post_only_rejected == 0


class TestOrderMetricsReporter:
    """Test OrderMetricsReporter functionality."""

    def test_initial_state(self, reporter):
        """Test reporter initializes with zero metrics."""
        assert reporter.metrics.placed == 0
        assert reporter.metrics.filled == 0
        assert reporter.metrics.cancelled == 0
        assert reporter.metrics.rejected == 0
        assert reporter.metrics.post_only_rejected == 0
        assert len(reporter.rejections_by_reason) == 0

    def test_record_placement(self, reporter, mock_order):
        """Test recording order placement."""
        reporter.record_placement(mock_order)

        assert reporter.metrics.placed == 1
        assert reporter.metrics.rejected == 0

    def test_record_multiple_placements(self, reporter, mock_order):
        """Test recording multiple placements."""
        reporter.record_placement(mock_order)
        reporter.record_placement(mock_order)
        reporter.record_placement(mock_order)

        assert reporter.metrics.placed == 3

    def test_record_rejection_basic(self, reporter):
        """Test recording basic rejection."""
        reporter.record_rejection("risk")

        assert reporter.metrics.rejected == 1
        assert reporter.metrics.post_only_rejected == 0
        assert reporter.rejections_by_reason["risk"] == 1

    def test_record_rejection_post_only(self, reporter):
        """Test recording post-only rejection."""
        reporter.record_rejection("post_only_cross", post_only=True)

        assert reporter.metrics.rejected == 1
        assert reporter.metrics.post_only_rejected == 1
        assert reporter.rejections_by_reason["post_only_cross"] == 1

    def test_record_multiple_rejection_reasons(self, reporter):
        """Test tracking multiple rejection reasons."""
        reporter.record_rejection("risk")
        reporter.record_rejection("position_sizing")
        reporter.record_rejection("risk")
        reporter.record_rejection("spec_violation")

        assert reporter.metrics.rejected == 4
        assert reporter.rejections_by_reason["risk"] == 2
        assert reporter.rejections_by_reason["position_sizing"] == 1
        assert reporter.rejections_by_reason["spec_violation"] == 1

    def test_record_fill(self, reporter, mock_order):
        """Test recording order fill."""
        reporter.record_fill(mock_order)

        assert reporter.metrics.filled == 1
        assert reporter.metrics.placed == 0

    def test_record_cancellation(self, reporter, mock_order):
        """Test recording order cancellation."""
        reporter.record_cancellation(mock_order)

        assert reporter.metrics.cancelled == 1
        assert reporter.metrics.placed == 0

    def test_get_summary(self, reporter, mock_order):
        """Test getting comprehensive summary."""
        # Record various events
        reporter.record_placement(mock_order)
        reporter.record_placement(mock_order)
        reporter.record_rejection("risk")
        reporter.record_rejection("position_sizing", post_only=True)
        reporter.record_fill(mock_order)
        reporter.record_cancellation(mock_order)

        summary = reporter.get_summary()

        assert summary["placed"] == 2
        assert summary["filled"] == 1
        assert summary["cancelled"] == 1
        assert summary["rejected"] == 2
        assert summary["post_only_rejected"] == 1
        assert summary["rejections_by_reason"]["risk"] == 1
        assert summary["rejections_by_reason"]["position_sizing"] == 1

    def test_get_metrics_dict(self, reporter, mock_order):
        """Test getting simple metrics dict (backward compatibility)."""
        reporter.record_placement(mock_order)
        reporter.record_rejection("risk")

        metrics = reporter.get_metrics_dict()

        assert metrics["placed"] == 1
        assert metrics["rejected"] == 1
        assert "rejections_by_reason" not in metrics

    def test_reset(self, reporter, mock_order):
        """Test resetting all metrics."""
        # Record some events
        reporter.record_placement(mock_order)
        reporter.record_rejection("risk")
        reporter.record_fill(mock_order)

        # Reset
        reporter.reset()

        # Verify all metrics are zero
        assert reporter.metrics.placed == 0
        assert reporter.metrics.filled == 0
        assert reporter.metrics.cancelled == 0
        assert reporter.metrics.rejected == 0
        assert reporter.metrics.post_only_rejected == 0
        assert len(reporter.rejections_by_reason) == 0

    def test_rejection_reason_empty_string(self, reporter):
        """Test that empty rejection reason increments rejected count but not reason dict."""
        reporter.record_rejection("")

        assert reporter.metrics.rejected == 1
        # Empty reason should not be added to rejections_by_reason
        assert "" not in reporter.rejections_by_reason

    def test_concurrent_rejection_tracking(self, reporter):
        """Test rejection tracking with multiple concurrent reasons."""
        # Simulate complex rejection scenario
        reporter.record_rejection("risk")
        reporter.record_rejection("risk")
        reporter.record_rejection("position_sizing")
        reporter.record_rejection("spec_violation")
        reporter.record_rejection("post_only_cross", post_only=True)
        reporter.record_rejection("post_only_cross", post_only=True)

        assert reporter.metrics.rejected == 6
        assert reporter.metrics.post_only_rejected == 2
        assert reporter.rejections_by_reason["risk"] == 2
        assert reporter.rejections_by_reason["position_sizing"] == 1
        assert reporter.rejections_by_reason["spec_violation"] == 1
        assert reporter.rejections_by_reason["post_only_cross"] == 2

    def test_full_lifecycle(self, reporter, mock_order):
        """Test tracking full order lifecycle."""
        # Place orders
        reporter.record_placement(mock_order)
        reporter.record_placement(mock_order)
        reporter.record_placement(mock_order)

        # Some get filled
        reporter.record_fill(mock_order)
        reporter.record_fill(mock_order)

        # One gets cancelled
        reporter.record_cancellation(mock_order)

        # Some get rejected
        reporter.record_rejection("risk")
        reporter.record_rejection("position_sizing")

        summary = reporter.get_summary()

        assert summary["placed"] == 3
        assert summary["filled"] == 2
        assert summary["cancelled"] == 1
        assert summary["rejected"] == 2
        assert summary["post_only_rejected"] == 0

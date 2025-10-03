"""Integration tests for OrderMetricsReporter telemetry export."""

from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.core.interfaces import Order
from bot_v2.features.live_trade.order_metrics_reporter import OrderMetricsReporter
from bot_v2.monitoring.metrics_collector import MetricsCollector


@pytest.fixture
def reporter():
    """Create OrderMetricsReporter instance."""
    return OrderMetricsReporter()


@pytest.fixture
def collector():
    """Create MetricsCollector instance."""
    return MetricsCollector()


@pytest.fixture
def mock_order():
    """Create mock order."""
    order = Mock(spec=Order)
    order.id = "order-123"
    return order


def test_export_to_collector_empty_metrics(reporter, collector):
    """Test exporting empty metrics to collector."""
    reporter.export_to_collector(collector, prefix="test_orders")

    # Verify all metrics are exported as 0
    assert collector.gauges.get("test_orders.placed") == 0.0
    assert collector.gauges.get("test_orders.filled") == 0.0
    assert collector.gauges.get("test_orders.cancelled") == 0.0
    assert collector.gauges.get("test_orders.rejected") == 0.0
    assert collector.gauges.get("test_orders.post_only_rejected") == 0.0


def test_export_to_collector_with_data(reporter, collector, mock_order):
    """Test exporting metrics with actual data."""
    # Record some events
    reporter.record_placement(mock_order)
    reporter.record_placement(mock_order)
    reporter.record_placement(mock_order)
    reporter.record_fill(mock_order)
    reporter.record_rejection("risk")
    reporter.record_rejection("position_sizing", post_only=True)
    reporter.record_cancellation(mock_order)

    # Export to collector
    reporter.export_to_collector(collector, prefix="orders")

    # Verify metrics
    assert collector.gauges.get("orders.placed") == 3.0
    assert collector.gauges.get("orders.filled") == 1.0
    assert collector.gauges.get("orders.cancelled") == 1.0
    assert collector.gauges.get("orders.rejected") == 2.0
    assert collector.gauges.get("orders.post_only_rejected") == 1.0

    # Verify rejection reasons
    assert collector.gauges.get("orders.rejection.risk") == 1.0
    assert collector.gauges.get("orders.rejection.position_sizing") == 1.0


def test_export_rejection_reason_sanitization(reporter, collector):
    """Test that rejection reasons are sanitized for metric names."""
    reporter.record_rejection("post-only-cross")
    reporter.record_rejection("spec violation")
    reporter.record_rejection("risk-limit-exceeded")

    reporter.export_to_collector(collector, prefix="orders")

    # Verify sanitized metric names (hyphens/spaces replaced with underscores)
    assert collector.gauges.get("orders.rejection.post_only_cross") == 1.0
    assert collector.gauges.get("orders.rejection.spec_violation") == 1.0
    assert collector.gauges.get("orders.rejection.risk_limit_exceeded") == 1.0


def test_export_with_custom_prefix(reporter, collector, mock_order):
    """Test exporting with custom metric prefix."""
    reporter.record_placement(mock_order)
    reporter.record_fill(mock_order)

    reporter.export_to_collector(collector, prefix="execution.live")

    # Verify custom prefix used
    assert collector.gauges.get("execution.live.placed") == 1.0
    assert collector.gauges.get("execution.live.filled") == 1.0


def test_export_updates_existing_metrics(reporter, collector, mock_order):
    """Test that re-exporting updates metric values."""
    # First export
    reporter.record_placement(mock_order)
    reporter.export_to_collector(collector, prefix="orders")
    assert collector.gauges.get("orders.placed") == 1.0

    # Record more events
    reporter.record_placement(mock_order)
    reporter.record_placement(mock_order)

    # Second export should update values
    reporter.export_to_collector(collector, prefix="orders")
    assert collector.gauges.get("orders.placed") == 3.0


def test_multiple_rejection_reasons_accumulate(reporter, collector):
    """Test that multiple rejections of same reason accumulate correctly."""
    reporter.record_rejection("risk")
    reporter.record_rejection("risk")
    reporter.record_rejection("risk")
    reporter.record_rejection("position_sizing")

    reporter.export_to_collector(collector, prefix="orders")

    assert collector.gauges.get("orders.rejection.risk") == 3.0
    assert collector.gauges.get("orders.rejection.position_sizing") == 1.0

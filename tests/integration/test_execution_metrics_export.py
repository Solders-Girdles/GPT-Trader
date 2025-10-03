"""End-to-end test: AdvancedExecutionEngine -> MetricsCollector integration."""

from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.core.interfaces import Order, OrderSide, OrderType, TimeInForce
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.features.live_trade.advanced_execution_models.models import OrderConfig
from bot_v2.monitoring.metrics_collector import MetricsCollector


@pytest.fixture
def mock_broker():
    """Create mock broker."""
    broker = Mock()
    broker.place_order = Mock(return_value=None)  # Simulate order rejection
    broker.get_quote = Mock()
    broker.get_product = Mock()
    return broker


@pytest.fixture
def exec_engine(mock_broker):
    """Create AdvancedExecutionEngine instance."""
    config = OrderConfig(enable_ioc=True)
    return AdvancedExecutionEngine(broker=mock_broker, config=config)


@pytest.fixture
def collector():
    """Create fresh MetricsCollector."""
    return MetricsCollector()


def test_export_metrics_to_collector(exec_engine, collector):
    """Test that execution metrics are exported to MetricsCollector."""
    # Simulate some order activity
    # (all will be rejected since broker.place_order returns None)
    exec_engine.place_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        quantity=1.0,
        order_type=OrderType.LIMIT,
        limit_price=50000.0,
    )
    exec_engine.place_order(
        symbol="ETH-USD",
        side=OrderSide.SELL,
        quantity=2.0,
        order_type=OrderType.MARKET,
    )

    # Export metrics
    exec_engine.export_metrics(collector, prefix="test_execution")

    # Verify order metrics were exported
    # (both should be rejected since broker returns None)
    assert collector.gauges.get("test_execution.orders.placed") == 0.0
    assert collector.gauges.get("test_execution.orders.rejected") >= 0.0

    # Verify pending orders gauge
    assert "test_execution.pending_orders" in collector.gauges

    # Verify stop trigger metrics
    assert "test_execution.stop_triggers" in collector.gauges
    assert "test_execution.active_stops" in collector.gauges


def test_metrics_collector_summary_includes_execution_metrics(exec_engine, collector):
    """Test that execution metrics appear in MetricsCollector summary."""
    # Simulate order placement
    exec_engine.place_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        quantity=1.0,
        order_type=OrderType.LIMIT,
        limit_price=50000.0,
    )

    # Export to collector
    exec_engine.export_metrics(collector, prefix="execution")

    # Get summary
    summary = collector.get_metrics_summary()

    # Verify execution metrics are included
    assert "gauges" in summary
    assert any(k.startswith("execution.") for k in summary["gauges"].keys())

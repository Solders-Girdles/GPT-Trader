from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from tests.unit.gpt_trader.tui.services.alert_manager_test_utils import (  # naming: allow
    create_alert_manager,
    create_mock_app,
    create_sample_state,
)

from gpt_trader.tui.services.alert_manager import (
    AlertCategory,
    AlertManager,
    AlertSeverity,
)


@pytest.fixture
def mock_app() -> MagicMock:
    return create_mock_app()


@pytest.fixture
def alert_manager(mock_app: MagicMock) -> AlertManager:
    return create_alert_manager(mock_app)


@pytest.fixture
def sample_state():
    return create_sample_state()


class TestTradeAlertsStaleOrders:
    """Trade alerts: stale open orders."""

    def test_stale_orders_rules_registered(self, alert_manager):
        """Test that trade alert rules are registered."""
        status = alert_manager.get_rule_status()
        assert "stale_open_orders" in status
        assert "failed_orders" in status
        assert "expired_orders" in status

    def test_stale_orders_alert_triggers(self, alert_manager, mock_app, sample_state):
        """Test stale orders alert triggers when order is old enough."""
        import time

        from gpt_trader.tui.thresholds import DEFAULT_ORDER_THRESHOLDS
        from gpt_trader.tui.types import ActiveOrders, Order

        stale_age = DEFAULT_ORDER_THRESHOLDS.age_warn + 10
        old_order = Order(
            order_id="stale-1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            status="OPEN",
            creation_time=time.time() - stale_age,
        )
        sample_state.order_data = ActiveOrders(orders=[old_order])

        alerts = alert_manager.check_alerts(sample_state)

        stale_alerts = [a for a in alerts if a.rule_id == "stale_open_orders"]
        assert len(stale_alerts) == 1
        assert stale_alerts[0].severity == AlertSeverity.WARNING
        assert stale_alerts[0].category == AlertCategory.TRADE
        assert "BTC-USD" in stale_alerts[0].message

    def test_stale_orders_no_alert_when_fresh(self, alert_manager, mock_app, sample_state):
        """Test no stale orders alert when orders are fresh."""
        import time

        from gpt_trader.tui.types import ActiveOrders, Order

        fresh_order = Order(
            order_id="fresh-1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            status="OPEN",
            creation_time=time.time() - 5,
        )
        sample_state.order_data = ActiveOrders(orders=[fresh_order])

        alerts = alert_manager.check_alerts(sample_state)

        stale_alerts = [a for a in alerts if a.rule_id == "stale_open_orders"]
        assert len(stale_alerts) == 0

    def test_stale_orders_ignores_filled_orders(self, alert_manager, mock_app, sample_state):
        """Test stale orders alert ignores non-open orders."""
        import time

        from gpt_trader.tui.thresholds import DEFAULT_ORDER_THRESHOLDS
        from gpt_trader.tui.types import ActiveOrders, Order

        stale_age = DEFAULT_ORDER_THRESHOLDS.age_warn + 100
        filled_order = Order(
            order_id="filled-1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            status="FILLED",
            creation_time=time.time() - stale_age,
        )
        sample_state.order_data = ActiveOrders(orders=[filled_order])

        alerts = alert_manager.check_alerts(sample_state)

        stale_alerts = [a for a in alerts if a.rule_id == "stale_open_orders"]
        assert len(stale_alerts) == 0

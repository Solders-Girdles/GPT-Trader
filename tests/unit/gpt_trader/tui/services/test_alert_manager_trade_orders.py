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


class TestTradeAlertsOrders:
    """Trade alerts: failed and expired orders."""

    def test_failed_orders_alert_triggers(self, alert_manager, mock_app, sample_state):
        """Test failed orders alert triggers on REJECTED status."""
        from gpt_trader.tui.types import ActiveOrders, Order

        rejected_order = Order(
            order_id="rejected-1",
            symbol="ETH-USD",
            side="SELL",
            quantity=Decimal("1.0"),
            price=Decimal("3000"),
            status="REJECTED",
        )
        sample_state.order_data = ActiveOrders(orders=[rejected_order])

        alerts = alert_manager.check_alerts(sample_state)

        failed_alerts = [a for a in alerts if a.rule_id == "failed_orders"]
        assert len(failed_alerts) == 1
        assert failed_alerts[0].severity == AlertSeverity.ERROR
        assert failed_alerts[0].category == AlertCategory.TRADE
        assert "ETH-USD" in failed_alerts[0].message
        assert "rejected" in failed_alerts[0].message.lower()

    def test_failed_orders_alert_triggers_on_failed_status(
        self, alert_manager, mock_app, sample_state
    ):
        """Test failed orders alert triggers on FAILED status."""
        from gpt_trader.tui.types import ActiveOrders, Order

        failed_order = Order(
            order_id="failed-1",
            symbol="SOL-USD",
            side="BUY",
            quantity=Decimal("10.0"),
            price=Decimal("100"),
            status="FAILED",
        )
        sample_state.order_data = ActiveOrders(orders=[failed_order])

        alerts = alert_manager.check_alerts(sample_state)

        failed_alerts = [a for a in alerts if a.rule_id == "failed_orders"]
        assert len(failed_alerts) == 1
        assert "SOL-USD" in failed_alerts[0].message

    def test_failed_orders_no_alert_on_cancelled(self, alert_manager, mock_app, sample_state):
        """Test failed orders alert does NOT trigger on CANCELLED (user-initiated)."""
        from gpt_trader.tui.types import ActiveOrders, Order

        cancelled_order = Order(
            order_id="cancelled-1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.5"),
            price=Decimal("50000"),
            status="CANCELLED",
        )
        sample_state.order_data = ActiveOrders(orders=[cancelled_order])

        alerts = alert_manager.check_alerts(sample_state)

        failed_alerts = [a for a in alerts if a.rule_id == "failed_orders"]
        assert len(failed_alerts) == 0

    def test_expired_orders_alert_triggers(self, alert_manager, mock_app, sample_state):
        """Test expired orders alert triggers on EXPIRED status."""
        from gpt_trader.tui.types import ActiveOrders, Order

        expired_order = Order(
            order_id="expired-1",
            symbol="BTC-USD",
            side="SELL",
            quantity=Decimal("0.2"),
            price=Decimal("50000"),
            status="EXPIRED",
        )
        sample_state.order_data = ActiveOrders(orders=[expired_order])

        alerts = alert_manager.check_alerts(sample_state)

        expired_alerts = [a for a in alerts if a.rule_id == "expired_orders"]
        assert len(expired_alerts) == 1
        assert expired_alerts[0].severity == AlertSeverity.WARNING
        assert expired_alerts[0].category == AlertCategory.TRADE
        assert "BTC-USD" in expired_alerts[0].message
        assert "expired" in expired_alerts[0].message.lower()

    def test_expired_orders_no_alert_when_none_expired(self, alert_manager, mock_app, sample_state):
        """Test no expired orders alert when no orders are expired."""
        from gpt_trader.tui.types import ActiveOrders, Order

        open_order = Order(
            order_id="open-1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            status="OPEN",
        )
        sample_state.order_data = ActiveOrders(orders=[open_order])

        alerts = alert_manager.check_alerts(sample_state)

        expired_alerts = [a for a in alerts if a.rule_id == "expired_orders"]
        assert len(expired_alerts) == 0

    def test_multiple_expired_orders_message(self, alert_manager, mock_app, sample_state):
        """Test expired orders alert message when multiple orders expired."""
        from gpt_trader.tui.types import ActiveOrders, Order

        expired_orders = [
            Order(
                order_id="exp-1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                status="EXPIRED",
            ),
            Order(
                order_id="exp-2",
                symbol="ETH-USD",
                side="SELL",
                quantity=Decimal("1.0"),
                price=Decimal("3000"),
                status="EXPIRED",
            ),
        ]
        sample_state.order_data = ActiveOrders(orders=expired_orders)

        alerts = alert_manager.check_alerts(sample_state)

        expired_alerts = [a for a in alerts if a.rule_id == "expired_orders"]
        assert len(expired_alerts) == 1
        assert "2 orders expired" in expired_alerts[0].message

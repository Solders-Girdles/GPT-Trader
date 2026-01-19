"""Basic behavior tests for CFMBalanceWidget."""

from decimal import Decimal
from unittest.mock import MagicMock

from tests.unit.gpt_trader.tui.widgets.cfm_balance_test_helpers import make_widget_with_mock_labels
from textual.widgets import Label

from gpt_trader.core.account import CFMBalance
from gpt_trader.tui.widgets.cfm_balance import CFMBalanceWidget


class TestCFMBalanceWidgetBasic:
    def test_initial_state_no_cfm(self):
        """Test widget starts with has_cfm=False."""
        widget = CFMBalanceWidget()
        assert widget.has_cfm is False

    def test_update_balance_with_none(self):
        """Test that None balance sets has_cfm to False."""
        widget = CFMBalanceWidget()
        widget.has_cfm = True  # Pre-set to True

        widget.query_one = MagicMock(return_value=MagicMock(spec=Label))

        widget.update_balance(None)
        assert widget.has_cfm is False

    def test_update_balance_with_valid_data(self):
        """Test that valid CFMBalance updates the widget correctly."""
        widget, mock_labels = make_widget_with_mock_labels()

        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("10000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("4000.00"),
            initial_margin=Decimal("1000.00"),
            unrealized_pnl=Decimal("150.50"),
            daily_realized_pnl=Decimal("75.25"),
            liquidation_threshold=Decimal("500.00"),
            liquidation_buffer_amount=Decimal("4500.00"),
            liquidation_buffer_percentage=90.0,
        )

        widget.update_balance(cfm_balance)

        assert widget.has_cfm is True
        for label in mock_labels.values():
            label.update.assert_called()

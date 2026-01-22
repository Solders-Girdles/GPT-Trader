"""Tests for CFMBalanceWidget."""

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


class TestCFMBalanceWidgetColorCoding:
    def test_margin_utilization_color_coding_low(self):
        """Test margin utilization shows green for low usage."""
        widget, mock_labels = make_widget_with_mock_labels()

        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("10000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("4500.00"),
            initial_margin=Decimal("500.00"),
            unrealized_pnl=Decimal("0"),
            daily_realized_pnl=Decimal("0"),
            liquidation_threshold=Decimal("250.00"),
            liquidation_buffer_amount=Decimal("4750.00"),
            liquidation_buffer_percentage=95.0,
        )

        widget.update_balance(cfm_balance)

        call_args = mock_labels["#cfm-margin-used"].update.call_args[0][0]
        assert "[green]" in call_args

    def test_margin_utilization_color_coding_high(self):
        """Test margin utilization shows red for high usage."""
        widget, mock_labels = make_widget_with_mock_labels()

        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("1000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("500.00"),
            initial_margin=Decimal("4500.00"),
            unrealized_pnl=Decimal("0"),
            daily_realized_pnl=Decimal("0"),
            liquidation_threshold=Decimal("4000.00"),
            liquidation_buffer_amount=Decimal("1000.00"),
            liquidation_buffer_percentage=25.0,
        )

        widget.update_balance(cfm_balance)

        call_args = mock_labels["#cfm-margin-used"].update.call_args[0][0]
        assert "[red]" in call_args

    def test_pnl_positive_color_coding(self):
        """Test positive P&L shows green with + prefix."""
        widget, mock_labels = make_widget_with_mock_labels()

        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("10000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("4000.00"),
            initial_margin=Decimal("1000.00"),
            unrealized_pnl=Decimal("250.00"),
            daily_realized_pnl=Decimal("0"),
            liquidation_threshold=Decimal("500.00"),
            liquidation_buffer_amount=Decimal("4500.00"),
            liquidation_buffer_percentage=90.0,
        )

        widget.update_balance(cfm_balance)

        call_args = mock_labels["#cfm-pnl"].update.call_args[0][0]
        assert "[green]" in call_args
        assert "+" in call_args

    def test_pnl_negative_color_coding(self):
        """Test negative P&L shows red."""
        widget, mock_labels = make_widget_with_mock_labels()

        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("10000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("4000.00"),
            initial_margin=Decimal("1000.00"),
            unrealized_pnl=Decimal("-150.00"),
            daily_realized_pnl=Decimal("0"),
            liquidation_threshold=Decimal("500.00"),
            liquidation_buffer_amount=Decimal("4500.00"),
            liquidation_buffer_percentage=90.0,
        )

        widget.update_balance(cfm_balance)

        call_args = mock_labels["#cfm-pnl"].update.call_args[0][0]
        assert "[red]" in call_args

    def test_liquidation_buffer_critical(self):
        """Test liquidation buffer shows red bold for critical level."""
        widget, mock_labels = make_widget_with_mock_labels()

        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("1000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("500.00"),
            initial_margin=Decimal("4500.00"),
            unrealized_pnl=Decimal("0"),
            daily_realized_pnl=Decimal("0"),
            liquidation_threshold=Decimal("4000.00"),
            liquidation_buffer_amount=Decimal("1000.00"),
            liquidation_buffer_percentage=15.0,
        )

        widget.update_balance(cfm_balance)

        call_args = mock_labels["#cfm-liq-buffer"].update.call_args[0][0]
        assert "[red bold]" in call_args


class TestCFMBalanceWidgetUnavailable:
    def test_show_unavailable(self):
        """Test show_unavailable method sets has_cfm to False."""
        widget = CFMBalanceWidget()
        widget.has_cfm = True

        mock_label = MagicMock(spec=Label)
        widget.query_one = MagicMock(return_value=mock_label)

        widget.show_unavailable("Test unavailable message")

        assert widget.has_cfm is False
        mock_label.update.assert_called_once()
        call_args = mock_label.update.call_args[0][0]
        assert "Test unavailable message" in call_args

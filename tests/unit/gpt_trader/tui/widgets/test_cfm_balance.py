"""Tests for CFMBalanceWidget."""

from decimal import Decimal
from unittest.mock import MagicMock

from textual.widgets import Label

from gpt_trader.core.account import CFMBalance
from gpt_trader.tui.widgets.cfm_balance import CFMBalanceWidget


class TestCFMBalanceWidget:
    """Test suite for CFMBalanceWidget."""

    def test_initial_state_no_cfm(self):
        """Test widget starts with has_cfm=False."""
        widget = CFMBalanceWidget()
        assert widget.has_cfm is False

    def test_update_balance_with_none(self):
        """Test that None balance sets has_cfm to False."""
        widget = CFMBalanceWidget()
        widget.has_cfm = True  # Pre-set to True

        # Mock query_one to avoid composition issues in test
        widget.query_one = MagicMock(return_value=MagicMock(spec=Label))

        widget.update_balance(None)
        assert widget.has_cfm is False

    def test_update_balance_with_valid_data(self):
        """Test that valid CFMBalance updates the widget correctly."""
        widget = CFMBalanceWidget()

        # Create mock labels for each field
        mock_labels = {
            "#cfm-balance": MagicMock(spec=Label),
            "#cfm-buying-power": MagicMock(spec=Label),
            "#cfm-avail-margin": MagicMock(spec=Label),
            "#cfm-margin-used": MagicMock(spec=Label),
            "#cfm-pnl": MagicMock(spec=Label),
            "#cfm-liq-buffer": MagicMock(spec=Label),
        }

        def mock_query_one(selector, *args):
            if selector in mock_labels:
                return mock_labels[selector]
            return MagicMock(spec=Label)

        widget.query_one = MagicMock(side_effect=mock_query_one)

        # Create CFMBalance with test data
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

        # Verify has_cfm is True
        assert widget.has_cfm is True

        # Verify labels were updated with formatted values
        mock_labels["#cfm-balance"].update.assert_called()
        mock_labels["#cfm-buying-power"].update.assert_called()
        mock_labels["#cfm-avail-margin"].update.assert_called()
        mock_labels["#cfm-margin-used"].update.assert_called()
        mock_labels["#cfm-pnl"].update.assert_called()
        mock_labels["#cfm-liq-buffer"].update.assert_called()

    def test_margin_utilization_color_coding_low(self):
        """Test margin utilization shows green for low usage."""
        widget = CFMBalanceWidget()

        mock_labels = {
            "#cfm-balance": MagicMock(spec=Label),
            "#cfm-buying-power": MagicMock(spec=Label),
            "#cfm-avail-margin": MagicMock(spec=Label),
            "#cfm-margin-used": MagicMock(spec=Label),
            "#cfm-pnl": MagicMock(spec=Label),
            "#cfm-liq-buffer": MagicMock(spec=Label),
        }

        widget.query_one = MagicMock(side_effect=lambda s, *a: mock_labels.get(s, MagicMock()))

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

        # Check margin used label - should be green for low margin (10%)
        call_args = mock_labels["#cfm-margin-used"].update.call_args[0][0]
        assert "[green]" in call_args  # Low margin = green

    def test_margin_utilization_color_coding_high(self):
        """Test margin utilization shows red for high usage."""
        widget = CFMBalanceWidget()

        mock_labels = {
            "#cfm-balance": MagicMock(spec=Label),
            "#cfm-buying-power": MagicMock(spec=Label),
            "#cfm-avail-margin": MagicMock(spec=Label),
            "#cfm-margin-used": MagicMock(spec=Label),
            "#cfm-pnl": MagicMock(spec=Label),
            "#cfm-liq-buffer": MagicMock(spec=Label),
        }

        widget.query_one = MagicMock(side_effect=lambda s, *a: mock_labels.get(s, MagicMock()))

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

        # Check margin used label - should be red for high margin (90%)
        call_args = mock_labels["#cfm-margin-used"].update.call_args[0][0]
        assert "[red]" in call_args  # High margin = red

    def test_pnl_positive_color_coding(self):
        """Test positive P&L shows green with + prefix."""
        widget = CFMBalanceWidget()

        mock_labels = {
            "#cfm-balance": MagicMock(spec=Label),
            "#cfm-buying-power": MagicMock(spec=Label),
            "#cfm-avail-margin": MagicMock(spec=Label),
            "#cfm-margin-used": MagicMock(spec=Label),
            "#cfm-pnl": MagicMock(spec=Label),
            "#cfm-liq-buffer": MagicMock(spec=Label),
        }

        widget.query_one = MagicMock(side_effect=lambda s, *a: mock_labels.get(s, MagicMock()))

        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("10000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("4000.00"),
            initial_margin=Decimal("1000.00"),
            unrealized_pnl=Decimal("250.00"),  # Positive
            daily_realized_pnl=Decimal("0"),
            liquidation_threshold=Decimal("500.00"),
            liquidation_buffer_amount=Decimal("4500.00"),
            liquidation_buffer_percentage=90.0,
        )

        widget.update_balance(cfm_balance)

        # Check P&L label - should be green with +
        call_args = mock_labels["#cfm-pnl"].update.call_args[0][0]
        assert "[green]" in call_args
        assert "+" in call_args

    def test_pnl_negative_color_coding(self):
        """Test negative P&L shows red."""
        widget = CFMBalanceWidget()

        mock_labels = {
            "#cfm-balance": MagicMock(spec=Label),
            "#cfm-buying-power": MagicMock(spec=Label),
            "#cfm-avail-margin": MagicMock(spec=Label),
            "#cfm-margin-used": MagicMock(spec=Label),
            "#cfm-pnl": MagicMock(spec=Label),
            "#cfm-liq-buffer": MagicMock(spec=Label),
        }

        widget.query_one = MagicMock(side_effect=lambda s, *a: mock_labels.get(s, MagicMock()))

        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("10000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("4000.00"),
            initial_margin=Decimal("1000.00"),
            unrealized_pnl=Decimal("-150.00"),  # Negative
            daily_realized_pnl=Decimal("0"),
            liquidation_threshold=Decimal("500.00"),
            liquidation_buffer_amount=Decimal("4500.00"),
            liquidation_buffer_percentage=90.0,
        )

        widget.update_balance(cfm_balance)

        # Check P&L label - should be red
        call_args = mock_labels["#cfm-pnl"].update.call_args[0][0]
        assert "[red]" in call_args

    def test_liquidation_buffer_critical(self):
        """Test liquidation buffer shows red bold for critical level."""
        widget = CFMBalanceWidget()

        mock_labels = {
            "#cfm-balance": MagicMock(spec=Label),
            "#cfm-buying-power": MagicMock(spec=Label),
            "#cfm-avail-margin": MagicMock(spec=Label),
            "#cfm-margin-used": MagicMock(spec=Label),
            "#cfm-pnl": MagicMock(spec=Label),
            "#cfm-liq-buffer": MagicMock(spec=Label),
        }

        widget.query_one = MagicMock(side_effect=lambda s, *a: mock_labels.get(s, MagicMock()))

        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("1000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("500.00"),
            initial_margin=Decimal("4500.00"),
            unrealized_pnl=Decimal("0"),
            daily_realized_pnl=Decimal("0"),
            liquidation_threshold=Decimal("4000.00"),
            liquidation_buffer_amount=Decimal("1000.00"),
            liquidation_buffer_percentage=15.0,  # Critical - below 25%
        )

        widget.update_balance(cfm_balance)

        # Check liquidation buffer label - should be red bold
        call_args = mock_labels["#cfm-liq-buffer"].update.call_args[0][0]
        assert "[red bold]" in call_args

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
